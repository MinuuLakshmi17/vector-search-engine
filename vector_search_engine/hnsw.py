"""From-scratch Hierarchical Navigable Small World (HNSW) index.

Implements random level assignment, greedy descent, layer-wise best-first search,
heuristic neighbor selection, reciprocal links, tombstone deletion, and rebuild.
The implementation favors clarity and inspectability over maximum throughput.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
import threading
from typing import Iterable
import numpy as np
from .distances import pairwise_distance, validate_metric


@dataclass(frozen=True, slots=True)
class SearchResult:
    id: str
    distance: float
    metadata: dict | None = None


class HNSWIndex:
    FORMAT_VERSION = 1

    def __init__(self, dimension: int, metric: str = "cosine", m: int = 16,
                 ef_construction: int = 200, ef_search: int = 64, seed: int = 42):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if m < 2:
            raise ValueError("m must be >= 2")
        if ef_construction < m or ef_search < 1:
            raise ValueError("ef_construction must be >= m and ef_search >= 1")
        self.dimension = dimension
        self.metric = validate_metric(metric)
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._rng = np.random.default_rng(seed)
        self.vectors: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict] = {}
        self.levels: dict[str, int] = {}
        self.graph: dict[str, list[list[str]]] = {}
        self.deleted: set[str] = set()
        self.entry_point: str | None = None
        self.max_level = -1
        self._lock = threading.RLock()

    @property
    def count(self) -> int:
        return len(self.vectors) - len(self.deleted)

    def _vector(self, vector: Iterable[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self.dimension:
            raise ValueError(f"vector must be one-dimensional with dimension {self.dimension}")
        if not np.isfinite(arr).all():
            raise ValueError("vector values must be finite")
        return arr.copy()

    def _distance(self, a: str | np.ndarray, b: str | np.ndarray) -> float:
        va = self.vectors[a] if isinstance(a, str) else a
        vb = self.vectors[b] if isinstance(b, str) else b
        return pairwise_distance(va, vb, self.metric)

    def _random_level(self) -> int:
        return int(-math.log(max(float(self._rng.random()), 1e-12)) / math.log(self.m))

    def _greedy(self, query: np.ndarray, entry: str, level: int) -> str:
        current = entry
        current_dist = self._distance(query, current)
        changed = True
        while changed:
            changed = False
            if level >= len(self.graph[current]):
                break
            for neighbor in self.graph[current][level]:
                distance = self._distance(query, neighbor)
                if (distance, neighbor) < (current_dist, current):
                    current, current_dist, changed = neighbor, distance, True
        return current

    def _search_layer(self, query: np.ndarray, entries: list[str], ef: int, level: int) -> list[tuple[float, str]]:
        visited: set[str] = set()
        candidates: list[tuple[float, str]] = []  # min-heap
        best: list[tuple[float, str]] = []        # max-heap encoded as negative distance
        for entry in entries:
            if entry in visited or level >= len(self.graph.get(entry, [])):
                continue
            visited.add(entry)
            distance = self._distance(query, entry)
            heapq.heappush(candidates, (distance, entry))
            heapq.heappush(best, (-distance, entry))
        while candidates:
            distance, current = heapq.heappop(candidates)
            worst = -best[0][0] if best else float("inf")
            if len(best) >= ef and distance > worst:
                break
            for neighbor in self.graph[current][level]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                nd = self._distance(query, neighbor)
                worst = -best[0][0] if best else float("inf")
                if len(best) < ef or nd < worst:
                    heapq.heappush(candidates, (nd, neighbor))
                    heapq.heappush(best, (-nd, neighbor))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted([(-neg_dist, node) for neg_dist, node in best], key=lambda x: (x[0], x[1]))

    def _select_neighbors(self, node: str, candidates: list[tuple[float, str]], limit: int) -> list[str]:
        selected: list[str] = []
        # HNSW diversity heuristic: prefer candidates not occluded by a closer selected neighbor.
        for distance, candidate in candidates:
            if candidate == node or candidate in self.deleted:
                continue
            if all(self._distance(candidate, chosen) >= distance for chosen in selected):
                selected.append(candidate)
            if len(selected) >= limit:
                break
        if len(selected) < min(limit, len(candidates)):
            for _, candidate in candidates:
                if candidate != node and candidate not in self.deleted and candidate not in selected:
                    selected.append(candidate)
                    if len(selected) >= limit:
                        break
        return selected

    def add(self, vector_id: str, vector: Iterable[float], metadata: dict | None = None) -> None:
        if not vector_id or not isinstance(vector_id, str):
            raise ValueError("id must be a non-empty string")
        arr = self._vector(vector)
        with self._lock:
            if vector_id in self.vectors and vector_id not in self.deleted:
                raise ValueError(f"id already exists: {vector_id}")
            # Reusing a deleted id is supported by replacing its old graph node via rebuild.
            if vector_id in self.vectors:
                self.vectors[vector_id] = arr
                self.metadata[vector_id] = dict(metadata or {})
                self.deleted.discard(vector_id)
                self._rebuild_locked()
                return
            self.vectors[vector_id] = arr
            self.metadata[vector_id] = dict(metadata or {})
            level = self._random_level()
            self.levels[vector_id] = level
            self.graph[vector_id] = [[] for _ in range(level + 1)]
            if self.entry_point is None:
                self.entry_point, self.max_level = vector_id, level
                return
            entry = self.entry_point
            for layer in range(self.max_level, level, -1):
                entry = self._greedy(arr, entry, layer)
            for layer in range(min(level, self.max_level), -1, -1):
                candidates = self._search_layer(arr, [entry], self.ef_construction, layer)
                limit = self.m0 if layer == 0 else self.m
                neighbors = self._select_neighbors(vector_id, candidates, limit)
                self.graph[vector_id][layer] = neighbors
                for neighbor in neighbors:
                    self.graph[neighbor][layer].append(vector_id)
                    if len(self.graph[neighbor][layer]) > limit:
                        scored = sorted(((self._distance(neighbor, other), other)
                                         for other in self.graph[neighbor][layer]), key=lambda x: (x[0], x[1]))
                        self.graph[neighbor][layer] = self._select_neighbors(neighbor, scored, limit)
                if candidates:
                    entry = candidates[0][1]
            if level > self.max_level:
                self.entry_point, self.max_level = vector_id, level

    def add_many(self, items: Iterable[tuple[str, Iterable[float], dict | None]]) -> int:
        count = 0
        for vector_id, vector, metadata in items:
            self.add(vector_id, vector, metadata)
            count += 1
        return count

    def search(self, query: Iterable[float], k: int = 10, ef_search: int | None = None) -> list[SearchResult]:
        if k < 1:
            raise ValueError("k must be >= 1")
        arr = self._vector(query)
        with self._lock:
            if self.count == 0 or self.entry_point is None:
                return []
            entry = self.entry_point
            for layer in range(self.max_level, 0, -1):
                entry = self._greedy(arr, entry, layer)
            ef = max(k, ef_search or self.ef_search)
            candidates = self._search_layer(arr, [entry], ef, 0)
            results = []
            for distance, node in candidates:
                if node not in self.deleted:
                    results.append(SearchResult(node, distance, dict(self.metadata.get(node, {}))))
                if len(results) >= k:
                    break
            # Tombstones can consume ef candidates; exact fallback only when too few live nodes found.
            if len(results) < min(k, self.count):
                live = [(self._distance(arr, node), node) for node in self.vectors if node not in self.deleted]
                live.sort(key=lambda x: (x[0], x[1]))
                results = [SearchResult(node, dist, dict(self.metadata.get(node, {}))) for dist, node in live[:k]]
            return results

    def delete(self, vector_id: str) -> bool:
        with self._lock:
            if vector_id not in self.vectors or vector_id in self.deleted:
                return False
            self.deleted.add(vector_id)
            return True

    def stats(self) -> dict:
        with self._lock:
            edge_count = sum(len(neighbors) for layers in self.graph.values() for neighbors in layers)
            return {"dimension": self.dimension, "metric": self.metric, "count": self.count,
                    "stored_nodes": len(self.vectors), "deleted_nodes": len(self.deleted),
                    "layers": self.max_level + 1 if self.entry_point else 0,
                    "edges_directed": edge_count, "m": self.m,
                    "ef_construction": self.ef_construction, "ef_search": self.ef_search}

    def _rebuild_locked(self) -> None:
        live = [(key, self.vectors[key].copy(), dict(self.metadata.get(key, {})))
                for key in self.vectors if key not in self.deleted]
        self.vectors, self.metadata, self.levels, self.graph = {}, {}, {}, {}
        self.deleted, self.entry_point, self.max_level = set(), None, -1
        for key, vector, metadata in live:
            self.add(key, vector, metadata)

    def compact(self) -> None:
        with self._lock:
            self._rebuild_locked()

    def to_dict(self) -> dict:
        with self._lock:
            return {"format_version": self.FORMAT_VERSION, "dimension": self.dimension,
                    "metric": self.metric, "m": self.m, "ef_construction": self.ef_construction,
                    "ef_search": self.ef_search, "rng_state": self._rng.bit_generator.state,
                    "vectors": {k: v.tolist() for k, v in self.vectors.items()},
                    "metadata": self.metadata, "levels": self.levels, "graph": self.graph,
                    "deleted": sorted(self.deleted), "entry_point": self.entry_point,
                    "max_level": self.max_level}

    @classmethod
    def from_dict(cls, data: dict) -> "HNSWIndex":
        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported index format version")
        index = cls(data["dimension"], data["metric"], data["m"], data["ef_construction"], data["ef_search"])
        index.vectors = {k: index._vector(v) for k, v in data["vectors"].items()}
        index.metadata = {k: dict(v) for k, v in data["metadata"].items()}
        index.levels = {k: int(v) for k, v in data["levels"].items()}
        index.graph = {k: [[str(n) for n in layer] for layer in layers] for k, layers in data["graph"].items()}
        index.deleted = set(data["deleted"])
        index.entry_point = data["entry_point"]
        index.max_level = int(data["max_level"])
        if "rng_state" in data:
            index._rng.bit_generator.state = data["rng_state"]
        index._validate_graph()
        return index

    def _validate_graph(self) -> None:
        keys = set(self.vectors)
        if set(self.graph) != keys or set(self.levels) != keys:
            raise ValueError("corrupt index: node tables do not match")
        if not self.deleted.issubset(keys):
            raise ValueError("corrupt index: deleted ids not present")
        for node, layers in self.graph.items():
            if len(layers) != self.levels[node] + 1:
                raise ValueError(f"corrupt index: invalid layer count for {node}")
            for layer in layers:
                if any(n not in keys for n in layer):
                    raise ValueError(f"corrupt index: dangling edge from {node}")
