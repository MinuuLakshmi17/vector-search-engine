"""Distance functions used by the index and exact-search baseline."""
from __future__ import annotations
import numpy as np

SUPPORTED_METRICS = ("cosine", "l2")


def validate_metric(metric: str) -> str:
    metric = metric.lower().strip()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {SUPPORTED_METRICS}, got {metric!r}")
    return metric


def pairwise_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "l2":
        return float(np.linalg.norm(a - b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 0.0
    if na == 0.0 or nb == 0.0:
        return 1.0
    return float(1.0 - np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
