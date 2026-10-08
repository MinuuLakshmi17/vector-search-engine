.PHONY: install test lint run benchmark
install:
	python -m pip install -e '.[test]'
test:
	python -m pytest -q
lint:
	python -m compileall -q vector_search_engine tests scripts
run:
	uvicorn vector_search_engine.api:app --reload
benchmark:
	python -m vector_search_engine.benchmark --n 2000 --dimension 64 --queries 100 --k 10
