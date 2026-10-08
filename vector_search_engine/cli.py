from __future__ import annotations
import argparse
import json
from pathlib import Path
from .hnsw import HNSWIndex
from .store import load_index, save_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or query a persisted HNSW index")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("stats")
    inspect.add_argument("index", type=Path)
    query = sub.add_parser("query")
    query.add_argument("index", type=Path)
    query.add_argument("--vector", required=True, help="JSON array of floats")
    query.add_argument("-k", type=int, default=10)
    args = parser.parse_args()
    index = load_index(args.index)
    if args.command == "stats":
        print(json.dumps(index.stats(), indent=2))
    else:
        vector = json.loads(args.vector)
        print(json.dumps([{"id": r.id, "distance": r.distance, "metadata": r.metadata}
                          for r in index.search(vector, args.k)], indent=2))

if __name__ == "__main__":
    main()
