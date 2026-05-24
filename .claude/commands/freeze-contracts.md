---
name: freeze-contracts
description: Run contract tests + the architecture fitness function, then tag.
---

Steps:
1. `uv run pytest packages/rie-contracts/tests tests/contract tests/architecture -q`
2. Verify all pass.
3. `git tag -a contracts-frozen-v0.1.0 -m "rie-contracts frozen"`
4. Refuse any subsequent edit to `packages/rie-contracts/`.
