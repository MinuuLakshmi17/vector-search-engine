"""Exact nearest-neighbour baseline; intended for validation and benchmarks."""
from __future__ import annotations
import numpy as np
from .distances import pairwise_distance, validate_metric


def brute_force(query: np.ndarray, vectors: dict[str, np.ndarray], k: int, metric: str = "cosine",
                excluded: set[str] | None = None) -> list[tuple[str, float]]:
    metric = validate_metric(metric)
    excluded = excluded or set()
    scored = [(key, pairwise_distance(query, vector, metric))
              for key, vector in vectors.items() if key not in excluded]
    scored.sort(key=lambda item: (item[1], item[0]))
    return scored[:max(0, k)]


def recall_at_k(approx_ids: list[str], exact_ids: list[str], k: int) -> float:
    if k <= 0:
        return 1.0
    expected = set(exact_ids[:k])
    if not expected:
        return 1.0
    return len(set(approx_ids[:k]) & expected) / len(expected)
