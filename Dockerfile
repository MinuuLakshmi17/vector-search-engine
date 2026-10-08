FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VECTOR_INDEX_PATH=/app/data/index.json
WORKDIR /app
COPY pyproject.toml README.md ./
COPY vector_search_engine ./vector_search_engine
RUN pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1
CMD ["uvicorn", "vector_search_engine.api:app", "--host", "0.0.0.0", "--port", "8000"]
