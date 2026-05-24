.PHONY: sync lock fmt lint type test test-fast test-contract test-arch up down logs migrate api ui ui-web demo bootstrap clean

# Auto-source .env for every recipe so DATABASE_URL_SYNC, OLLAMA_*, etc. are
# available without manual `set -a; source .env`.
ifneq (,$(wildcard .env))
	include .env
	export
endif

sync:
	uv sync --all-packages --all-extras

lock:
	uv lock

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

type:
	uv run pyright

test:
	uv run pytest -q

test-fast:
	uv run pytest -q -m "not integration and not slow"

test-contract:
	uv run pytest tests/contract -q

test-arch:
	uv run pytest tests/architecture -q

up:
	docker compose up -d qdrant postgres

down:
	docker compose down

logs:
	docker compose logs -f

migrate:
	uv run alembic -c packages/rie-persistence/alembic.ini upgrade head

api:
	uv run uvicorn rie_api.main:app --host 0.0.0.0 --port 8080 --reload

ui:
	uv run streamlit run packages/rie-ui/src/rie_ui/app.py --server.port 8501

ui-web:
	cd packages/rie-ui-web && npm install && npm run dev

ui-web-build:
	cd packages/rie-ui-web && npm install && npm run build

demo:
	uv run python scripts/run_jurisdiction.py --pillars 6,7 --jurisdiction SAMPLE

bootstrap:
	bash scripts/bootstrap.sh

clean:
	rm -rf .pytest_cache .ruff_cache .pyright dist build data/cache/* data/outputs/*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
