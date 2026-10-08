"""Reproducible ANN-vs-exact benchmark; writes machine-readable JSON and CSV."""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from .brute_force import brute_force, recall_at_k
from .hnsw import HNSWIndex


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def run_benchmark(n: int = 2000, dimension: int = 64, queries: int = 100, k: int = 10,
                  metric: str = "cosine", seed: int = 7, m: int = 16,
                  ef_construction: int = 200, ef_search: int = 64) -> dict:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, dimension)).astype(np.float32)
    index = HNSWIndex(dimension, metric, m, ef_construction, ef_search, seed)
    started = time.perf_counter()
    for i, vector in enumerate(data):
        index.add(f"v{i:08d}", vector)
    build_seconds = time.perf_counter() - started
    qdata = rng.normal(size=(queries, dimension)).astype(np.float32)
    latency, recalls = [], []
    exact_vectors = index.vectors
    for q in qdata:
        t0 = time.perf_counter()
        approx = index.search(q, k)
        latency.append((time.perf_counter() - t0) * 1000)
        exact = brute_force(q, exact_vectors, k, metric)
        recalls.append(recall_at_k([x.id for x in approx], [x[0] for x in exact], k))
    total = sum(latency) / 1000
    return {"n": n, "dimension": dimension, "queries": queries, "k": k, "metric": metric,
            "seed": seed, "m": m, "ef_construction": ef_construction, "ef_search": ef_search,
            "build_seconds": build_seconds, "queries_per_second": queries / total if total else 0,
            "latency_ms_p50": percentile(latency, 50), "latency_ms_p95": percentile(latency, 95),
            "latency_ms_p99": percentile(latency, 99), "recall_at_k": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)), "index_stats": index.stats()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--output", default="benchmark-results")
    args = parser.parse_args()
    if args.n < 1 or args.queries < 1 or args.dimension < 1 or args.k < 1:
        parser.error("n, queries, dimension and k must be positive")
    result = run_benchmark(args.n, args.dimension, args.queries, args.k, args.metric)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k, v in result.items() if not isinstance(v, dict)])
        writer.writeheader()
        writer.writerow({k: v for k, v in result.items() if not isinstance(v, dict)})
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
