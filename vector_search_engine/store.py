"""Safe JSON persistence with atomic replacement (no pickle deserialization)."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
from .hnsw import HNSWIndex


def save_index(index: HNSWIndex, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(index.to_dict(), separators=(",", ":"), allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_index(path: str | Path) -> HNSWIndex:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return HNSWIndex.from_dict(data)
