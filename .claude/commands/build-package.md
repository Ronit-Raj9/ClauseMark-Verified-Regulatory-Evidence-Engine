---
name: build-package
description: Build a single rie-* package via its owning agent.
---

Argument: `$1` = package name (e.g. `rie-retrieval`).

Steps:
1. Locate `.claude/agents/<name>-agent.md`.
2. Spawn that agent with a prompt that reads `systemArchitecture.md`,
   `implementation.md`, the relevant ports from `rie-contracts`, and the
   acceptance criteria for the package.
3. Wait for completion; verify `uv run pytest packages/$1 tests/contract -q` passes.
