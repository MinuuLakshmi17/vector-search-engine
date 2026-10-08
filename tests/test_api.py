from fastapi.testclient import TestClient
import vector_search_engine.api as api


def test_api_crud_batch_query_and_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "INDEX_PATH", tmp_path / "index.json")
    monkeypatch.setattr(api, "DIMENSION", 3)
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        batch = client.post("/vectors/batch", json={"vectors": [
            {"id": "a", "vector": [1, 0, 0], "metadata": {"kind": "unit"}},
            {"id": "b", "vector": [0, 1, 0], "metadata": {}}]})
        assert batch.status_code == 201, batch.text
        result = client.post("/query", json={"vector": [1, 0, 0], "k": 1})
        assert result.status_code == 200, result.text
        assert result.json()[0]["id"] == "a"
        assert client.get("/stats").json()["count"] == 2
        assert client.delete("/vectors/b").status_code == 200
        assert client.delete("/vectors/missing").status_code == 404
        assert client.post("/query", json={"vector": [1, 0], "k": 1}).status_code == 422
    assert (tmp_path / "index.json").exists()
