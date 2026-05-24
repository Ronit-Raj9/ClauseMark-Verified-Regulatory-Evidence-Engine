---
name: scaffold
description: Phase-0 skeleton generator. Lays the workspace, contracts, schemas, pillar slots, CI.
---

Run the Phase-0 scaffold:

1. Create the uv workspace + every `packages/rie-*` skeleton.
2. Write `pillars/_schema/`, `_template/`, `registry.yaml` with all 12 pillars
   (6/7 built; rest stubs).
3. Write `rie-contracts` ports + Pydantic models and FREEZE.
4. Author `tests/architecture/` fitness function.

Stop after `make test-fast` passes and architecture tests are green.
