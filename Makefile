.PHONY: build up down test run logs clean shell

build:
	docker compose build

up:
	docker compose up -d postgres-db

down:
	docker compose down

test:
	docker compose run --rm test-runner

run:
	docker compose run --rm commerceops-app

shell:
	docker compose run --rm commerceops-app /bin/bash

logs:
	docker compose logs -f

clean:
	docker compose down -v
	rm -rf __pycache__ .pytest_cache .coverage Solving_Business_Problems_Output.xlsx
