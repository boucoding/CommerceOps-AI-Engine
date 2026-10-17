# syntax=docker/dockerfile:1
# CommerceOps AI Engine: multi-stage image.
#   runtime (default): API server / batch runs, non-root, no tests or dev tools
#   test:              runtime + dev tools + test suite

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

FROM base AS builder
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM builder AS dev-builder
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

FROM base AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app
WORKDIR /app
COPY src/ src/
COPY data/ data/
COPY fixtures/ fixtures/
COPY evals/gold_labels.json evals/gold_labels.json
RUN mkdir -p exports && chown app:app exports
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["python", "-m", "src.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test
COPY --from=dev-builder /opt/venv /opt/venv
ENV COVERAGE_FILE=/tmp/.coverage
COPY pyproject.toml .env.example ./
COPY tests/ tests/
HEALTHCHECK NONE
CMD ["pytest", "-q", "-p", "no:cacheprovider"]
