# Regulatory Intelligence Engine (RIE)

Evidence-extraction and verification engine for RDTII digital-governance indicators.

> Maps clauses of real legal text to indicators with exact span-level citations and deterministic
> authority handling, verifies every claim, reasons honestly about coverage, and produces a
> human-reviewable audit package with a **recommended** score.
> **Reviewers retain final scoring authority.**

## Quick start

```bash
# 1. Bootstrap: uv sync + docker services + migrations
make bootstrap

# 2. Start infrastructure (Postgres + Qdrant)
make up
make migrate

# 3. Run pipeline on sample laws
make demo

# 4. Audit UI
make ui            # http://localhost:8501

# 5. API
make api           # http://localhost:8080
```

## Architecture

Hexagonal. `rie-contracts` (ports + Pydantic models) is frozen first; every other package
implements an adapter behind one of those ports. Dependency direction is enforced by
`tests/architecture/`.

```
ingest → extract (+ structure graph) → retrieval (parent-doc + RRF + rerank)
       → classify (constrained decoding) → verify (4-gate) → coverage (3-state)
       → output (L1 evidence package + L2 recommended score)
       → audit viewer (HITL)
```

Orchestrated by **LangGraph** with Postgres checkpointer + interrupts for human review.

## Packages

| Package | Owns |
|---|---|
| `rie-contracts` | Ports + Pydantic models. **Frozen** — never edited after Phase 0. |
| `rie-domain` | Pure business logic. Imports only `rie-contracts`. |
| `rie-config` | Loads + validates `pillars/`, `sources/`, `gold/`. |
| `rie-profiles` | Document-profile strategies (statutory / tabular / treaty). |
| `rie-ingest` | Sample-law loader, source-registry parser, provenance. |
| `rie-extract` | Docling + PyMuPDF + VLM-OCR + structure graph. |
| `rie-retrieval` | BGE-M3 + Qdrant hybrid + reranker + parent-doc. |
| `rie-classify` | LLM classification + constrained decoding + regime assembly. |
| `rie-verify` | 4-gate verification + ID-replacement citations. |
| `rie-coverage` | 3-state absence reasoning. |
| `rie-orchestration` | LangGraph wiring. |
| `rie-persistence` | Postgres + Alembic. |
| `rie-api` | FastAPI driver. |
| `rie-ui` | Streamlit audit viewer. |
| `rie-eval` | Gold-set runner, RAGAS metrics, ablations. |

## Pillars

All 12 RDTII pillars exist as data from commit one. Pillars 6 & 7 built to depth;
others are validated stubs (`status: stub` in `pillars/registry.yaml`).

**Add a pillar:** copy `pillars/_template/pillar_NN_template.yaml`, register in
`registry.yaml`, drop a gold set in `gold/pillar_NN/`, flip `status: built`. See
[docs/guides/adding_a_pillar.md](docs/guides/adding_a_pillar.md).

## Non-negotiables

- **The LLM never authors a fact** — it selects from enums and emits span IDs.
- **Pillar behaviour lives in `pillars/`, never in `.py`.**
- **All 12 pillars as data from commit one** — depth varies.
- **Contracts frozen before parallel work.**
- **uv only** — never `pip`, never hand-edit `uv.lock`.

See [docs/architecture_v3.md](systemArchitecture.md) for the full architecture.
