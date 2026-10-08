import numpy as np
import pytest
from vector_search_engine.hnsw import HNSWIndex
from vector_search_engine.store import load_index, save_index


def test_round_trip(tmp_path):
    path = tmp_path / "index.json"
    index = HNSWIndex(4, metric="l2", seed=9)
    for i in range(25):
        index.add(f"id-{i}", np.full(4, i, dtype=np.float32), {"i": i})
    index.delete("id-3")
    save_index(index, path)
    loaded = load_index(path)
    assert loaded.stats()["count"] == 24
    assert loaded.metric == "l2"
    assert loaded.search([7, 7, 7, 7], 1)[0].id == "id-7"


def test_corrupt_format_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"format_version": 999}', encoding="utf-8")
    with pytest.raises((KeyError, ValueError)):
        load_index(path)
