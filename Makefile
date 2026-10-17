# CommerceOps AI Engine: common tasks. Run `make help` for the list.
PY ?= python
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)
export HOST_UID HOST_GID

.PHONY: help install dev lint format typecheck test test-integration check eval eval-offline demo run record \
        fixtures serve up down logs docker-run docker-test clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

dev:  ## Install runtime + development dependencies
	$(PY) -m pip install -r requirements-dev.txt

lint:  ## Lint and check formatting
	ruff check . && ruff format --check .

format:  ## Auto-format and apply safe lint fixes
	ruff format . && ruff check --fix .

typecheck:  ## Static type checking
	mypy

test:  ## Unit tests (integration tests run when TEST_DATABASE_URL is set)
	$(PY) -m pytest --cov --cov-report=term-missing

test-integration:  ## PostgreSQL integration tests only (needs TEST_DATABASE_URL)
	$(PY) -m pytest -m integration

check: lint typecheck test  ## Everything CI runs locally

eval:  ## Score replay fixtures against the gold labels
	$(PY) -m src.cli eval --mode replay --output evals/latest_replay.md

eval-offline:  ## Score the deterministic rules against the gold labels
	$(PY) -m src.cli eval --mode offline --output evals/latest_offline.md

demo:  ## Process the sample emails with replay fixtures (no API key needed) -> exports/
	$(PY) -m src.cli run --mode replay

run:  ## Process the email queue with the configured LLM_MODE -> exports/
	$(PY) -m src.cli run

record:  ## Re-record fixtures from the live model (needs OPENAI_API_KEY), then score them
	STORAGE_BACKEND=memory $(PY) -m src.cli run --mode record --out exports
	$(PY) -m src.cli eval --mode replay --output evals/latest_replay.md

fixtures:  ## Rebuild hand-authored replay fixtures after changing prompts or retrieval
	$(PY) -m scripts.build_reference_fixtures

serve:  ## Run the API locally on :8000
	$(PY) -m src.cli serve

up:  ## Start PostgreSQL + API in Docker
	docker compose up -d --build

down:  ## Stop the Docker stack
	docker compose down

logs:  ## Follow Docker logs
	docker compose logs -f

docker-run:  ## Run the batch pipeline in Docker -> ./exports
	docker compose --profile batch run --rm pipeline

docker-test:  ## Run unit + integration tests in Docker
	docker compose --profile test run --rm --build tests

clean:  ## Remove caches and local evaluation output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov evals/latest_*.md
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
