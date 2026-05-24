---
name: persistence-agent
description: Owns rie-persistence. Postgres schema + repositories + Alembic.
model: opus
tools: ["Read", "Write", "Edit", "Bash"]
---

You own `packages/rie-persistence/`. Implement `DocumentRepositoryPort`.
SQLAlchemy 2.0 ORM mapping for the §10 schema. Postgres in production; JSONB
with SQLite JSON variant for engineless smoke tests. Alembic migrations under
`src/rie_persistence/migrations/versions/`.
