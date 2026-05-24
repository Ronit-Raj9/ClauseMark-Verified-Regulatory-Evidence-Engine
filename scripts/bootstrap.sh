#!/usr/bin/env bash
# Bootstrap the dev environment end-to-end.
#   - uv sync (workspace)
#   - docker compose up Postgres + Qdrant
#   - alembic upgrade head
#   - smoke test
#
# Idempotent. Re-runnable.

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "WARN: docker not found — Postgres/Qdrant will not start. Install Docker first."
fi

echo "▶ uv sync (workspace)"
uv sync --all-packages --all-extras

if command -v docker >/dev/null 2>&1; then
  echo "▶ docker compose up postgres + qdrant"
  docker compose up -d postgres qdrant
  echo "▶ wait for Postgres ready"
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-rie}" >/dev/null 2>&1; then
      echo "  Postgres is ready."
      break
    fi
    sleep 1
  done

  if [[ -f .env ]]; then set -a; source .env; set +a; fi

  echo "▶ alembic upgrade head"
  uv run alembic -c packages/rie-persistence/alembic.ini upgrade head
fi

echo "▶ pytest (fast suite)"
uv run pytest -q -m "not integration and not slow"

echo "✅ bootstrap complete."
