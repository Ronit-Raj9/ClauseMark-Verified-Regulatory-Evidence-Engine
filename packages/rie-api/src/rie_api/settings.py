"""Application settings loaded from environment / `.env`.

All settings live here. Routers / deps read these via the `Settings` instance
injected through FastAPI's lifespan, never via module-level globals.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed environment configuration.

    Values are sourced (in order) from constructor args, environment variables,
    and a project-root `.env` file. Unknown keys are ignored so we tolerate the
    shared workspace `.env` containing keys for other packages.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── App ───────────────────────────────────────────────────────────────
    env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Where the pillar / source / gold YAML lives.
    repo_root: Path = Field(default_factory=lambda: Path.cwd())
    pillars_dir: str = "pillars"
    gold_dir: str = "gold"
    sources_dir: str = "sources"
    data_dir: str = "./data"

    # ── Postgres ──────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "rie"
    postgres_user: str = "rie"
    postgres_password: str = "rie_dev_password"
    database_url_sync: str | None = None

    # ── Qdrant ────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection: str = "rie_clauses"

    # ── LLM / serving ─────────────────────────────────────────────────────
    llm_backend: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M"
    ollama_verifier_model: str = "mistral:7b-instruct-q4_K_M"

    # In-process test/dev short-circuit. When True, the app uses an in-memory
    # SQLite DocumentRepository so tests can run without Postgres.
    use_in_memory_db: bool = False

    def sync_dsn(self) -> str:
        if self.database_url_sync:
            return self.database_url_sync
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def qdrant_base_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


def get_settings() -> Settings:
    """Construct a fresh `Settings` instance.

    The app's lifespan stores ONE instance on `app.state`; tests override deps
    rather than touching environment.
    """
    return Settings()
