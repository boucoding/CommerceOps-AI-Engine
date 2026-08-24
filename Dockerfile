# ==============================================================================
# CommerceOps AI Engine - Ultra-Slim Multi-Stage Container
# ==============================================================================

# Stage 1: Build Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Minimal Runtime
FROM python:3.11-slim AS runner

WORKDIR /app

# Install only minimal runtime postgres client library and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache

# Copy pre-built wheels and packages from builder
COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Copy application source code and datasets
COPY src/ /app/src/
COPY data/ /app/data/
COPY tests/ /app/tests/
COPY init_db.sql /app/init_db.sql
COPY run_pipeline.py /app/run_pipeline.py

CMD ["python", "run_pipeline.py"]
