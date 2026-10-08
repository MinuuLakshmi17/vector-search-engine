import numpy as np
import pytest
from vector_search_engine.hnsw import HNSWIndex
from vector_search_engine.brute_force import brute_force, recall_at_k


def test_insert_search_and_metadata():
    index = HNSWIndex(3, seed=1)
    index.add("origin", [1, 0, 0], {"label": "x"})
    index.add("near", [0.99, 0.01, 0], {"label": "near"})
    index.add("far", [0, 1, 0])
    result = index.search([1, 0, 0], 2)
    assert result[0].id == "origin"
    assert result[1].id == "near"
    assert result[0].metadata == {"label": "x"}


def test_l2_metric_and_validation():
    index = HNSWIndex(2, metric="l2", seed=2)
    index.add("a", [0, 0])
    index.add("b", [3, 4])
    assert index.search([0, 0], 1)[0].id == "a"
    with pytest.raises(ValueError):
        index.add("bad", [1])
    with pytest.raises(ValueError):
        index.search([float("nan"), 0], 1)


def test_delete_compact_and_reuse_id():
    index = HNSWIndex(2, seed=3)
    for i in range(20):
        index.add(str(i), [float(i), 1.0])
    assert index.delete("5") is True
    assert index.delete("5") is False
    assert "5" not in [r.id for r in index.search([5, 1], 5)]
    index.compact()
    index.add("5", [5, 1])
    assert "5" in [r.id for r in index.search([5, 1], 1)]


def test_recall_reasonable_on_small_clustered_data():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(250, 12)).astype(np.float32)
    index = HNSWIndex(12, m=12, ef_construction=100, ef_search=100, seed=42)
    for i, vector in enumerate(data):
        index.add(str(i), vector)
    recalls = []
    for q in data[:20]:
        approx = index.search(q, 5)
        exact = brute_force(q, index.vectors, 5)
        recalls.append(recall_at_k([x.id for x in approx], [x[0] for x in exact], 5))
    assert np.mean(recalls) >= 0.75


def test_bad_configuration():
    with pytest.raises(ValueError):
        HNSWIndex(0)
    with pytest.raises(ValueError):
        HNSWIndex(3, m=1)
