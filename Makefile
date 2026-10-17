# CommerceOps AI Engine: common tasks. Run `make help` for the list.
PY ?= python
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)
export HOST_UID HOST_GID

.PHONY: help install dev lint format typecheck test check record eval run migrate serve up down logs \
        docker-run docker-test clean

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

test:  ## Tests (database tests need TEST_DATABASE_URL; see README)
	$(PY) -m pytest --cov --cov-report=term-missing

check: lint typecheck test  ## Everything CI runs locally

record:  ## Run the agent live on the sample emails and save the recording (needs OPENAI_API_KEY)
	$(PY) -m src.cli eval --mode record --output evals/latest_live.md

eval:  ## Score the recorded run against the gold labels (no API key needed)
	$(PY) -m src.cli eval --mode replay --output evals/latest_replay.md

run:  ## Process data/inbound_customer_emails.csv with the configured LLM_MODE -> exports/
	$(PY) -m src.cli run

migrate:  ## Apply database migrations
	$(PY) -m src.cli migrate

serve:  ## Run the API locally on :8000
	$(PY) -m src.cli serve

up:  ## Start PostgreSQL + API in Docker (needs .env)
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
