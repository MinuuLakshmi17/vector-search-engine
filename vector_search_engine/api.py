"""FastAPI REST interface for the HNSW index."""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .hnsw import HNSWIndex
from .store import load_index, save_index

INDEX_PATH = Path(os.getenv("VECTOR_INDEX_PATH", "data/index.json"))
DIMENSION = int(os.getenv("VECTOR_DIMENSION", "128"))
METRIC = os.getenv("VECTOR_METRIC", "cosine")
M = int(os.getenv("HNSW_M", "16"))
EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "64"))


class VectorInput(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchInput(BaseModel):
    vectors: list[VectorInput] = Field(min_length=1, max_length=1000)


class QueryInput(BaseModel):
    vector: list[float]
    k: int = Field(default=10, ge=1, le=1000)
    ef_search: int | None = Field(default=None, ge=1, le=10000)


class SearchOutput(BaseModel):
    id: str
    distance: float
    metadata: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    if INDEX_PATH.exists():
        app.state.index = load_index(INDEX_PATH)
        if app.state.index.dimension != DIMENSION:
            raise RuntimeError(f"persisted index dimension {app.state.index.dimension} != configured {DIMENSION}")
    else:
        app.state.index = HNSWIndex(DIMENSION, METRIC, M, EF_CONSTRUCTION, EF_SEARCH)
    app.state.lock = RLock()
    yield
    with app.state.lock:
        save_index(app.state.index, INDEX_PATH)


app = FastAPI(title="Vector Search Engine", version="1.0.0",
              description="From-scratch HNSW approximate nearest-neighbour search API.", lifespan=lifespan)


def _index() -> HNSWIndex:
    return app.state.index


def _persist() -> None:
    save_index(_index(), INDEX_PATH)


def _validate_vector(vector: list[float]) -> None:
    if len(vector) != _index().dimension:
        raise HTTPException(422, f"vector dimension must be {_index().dimension}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_count": _index().count}


@app.get("/stats")
def stats() -> dict:
    return _index().stats()


@app.post("/vectors", status_code=201)
def insert(item: VectorInput) -> dict:
    _validate_vector(item.vector)
    try:
        with app.state.lock:
            _index().add(item.id, item.vector, item.metadata)
            _persist()
    except ValueError as exc:
        raise HTTPException(409 if "already exists" in str(exc) else 422, str(exc)) from exc
    return {"id": item.id, "inserted": True}


@app.post("/vectors/batch", status_code=201)
def batch_insert(batch: BatchInput) -> dict:
    ids = [item.id for item in batch.vectors]
    if len(ids) != len(set(ids)):
        raise HTTPException(422, "batch contains duplicate ids")
    for item in batch.vectors:
        _validate_vector(item.vector)
    with app.state.lock:
        existing = set(_index().vectors) - _index().deleted
        duplicate = existing.intersection(ids)
        if duplicate:
            raise HTTPException(409, f"ids already exist: {sorted(duplicate)[:5]}")
        try:
            _index().add_many((item.id, item.vector, item.metadata) for item in batch.vectors)
            _persist()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    return {"inserted": len(batch.vectors)}


@app.post("/query", response_model=list[SearchOutput])
def query(body: QueryInput) -> list[SearchOutput]:
    _validate_vector(body.vector)
    try:
        return [SearchOutput(id=r.id, distance=r.distance, metadata=r.metadata or {})
                for r in _index().search(body.vector, body.k, body.ef_search)]
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.delete("/vectors/{vector_id}")
def delete(vector_id: str) -> dict:
    with app.state.lock:
        if not _index().delete(vector_id):
            raise HTTPException(404, "vector not found")
        _persist()
    return {"id": vector_id, "deleted": True}


@app.post("/compact")
def compact() -> dict:
    with app.state.lock:
        _index().compact()
        _persist()
    return {"compacted": True, "stats": _index().stats()}
