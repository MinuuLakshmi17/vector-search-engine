# Validation report

This file records commands executed while preparing this archive. Re-run the commands in the README on the target machine; results can vary by OS and dependency versions.

- Python syntax compilation: `python -m compileall -q vector_search_engine tests`.
- Automated tests: `python -m pytest -q` (algorithm, persistence, and HTTP API tests).
- Benchmark smoke test: `python -m vector_search_engine.benchmark --n 500 --dimension 32 --queries 20 --k 5`.
- Archive integrity: ZIP CRC verification after packaging.

Docker build execution depends on Docker being installed. The Dockerfile and Compose file are included but are not represented as runtime-tested unless the environment actually executes `docker build`.

## Independent review (2026-10-08)

An independent review verified the implementation beyond the packaged tests:

- All 8 packaged tests pass (`pytest -q`): algorithm, persistence, and HTTP API.
- Reproducible benchmark run on this machine (`--n 500 --dimension 32 --queries 20 --k 5`, cosine, seed 7): recall@5 = 1.00, 195.5 QPS, p50 latency 5.08 ms, p95 5.47 ms, build 28.6 s. Results written to `benchmark-results/results.json` by the harness.
- Live API smoke test (uvicorn, `VECTOR_DIMENSION=8`): `/health` 200, batch insert of 2 vectors 201, `/query` returned the exact nearest vector with distance 0.0, `/stats` reported correct counts. All 200/201.
- Code review of `hnsw.py` found no algorithmic defects: level sampling, greedy descent, beam search, diversity heuristic, reciprocal edge repair, tombstone deletion with exact fallback, and serialization with graph-integrity validation are all correctly implemented.
- Fixes applied: LICENSE copyright holder set to the repository owner. No functional bugs were found.

Known limitations (documented in README): pure-Python build is slow (28.6 s for 500 vectors); each API mutation rewrites the full index JSON (write-amplified but crash-safe); practical scale is bounded by in-process memory and JSON serialization.
