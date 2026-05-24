# Regulatory Intelligence Engine — Implementation Plan v2
### All-Opus multi-agent build · uv workspace · hexagonal · built for all 12 pillars from day one

> v2 changes two things from v1: **(1)** every Claude Code agent runs **Opus 4.7 at high reasoning effort** — with a Max subscription, capability is the only axis that matters, so there is no model tiering; **(2)** the file structure is fully expanded so that **adding any of the 12 RDTII pillars later is a configuration task, not a code change.** The structure ships with all 12 pillar slots, a profile-strategy system, per-pillar gold sets, and a pillar registry — present from the first commit.

---

## 0. Two principles, restated

**Principle A — contracts before parallelism.** Multi-agent builds collapse when agents edit the same files. The fix: a frozen contract layer (`rie-contracts`: ports + Pydantic models) written *before* any agent spawns. Once frozen, N agents build behind it blind to each other, because the only shared surface has stopped moving.

**Principle B — the pillar is data, never code.** The engine contains zero pillar-specific logic. A pillar is: one YAML file in `pillars/`, an entry in `pillars/registry.yaml`, a gold set in `gold/`, and — only if it needs a new *document profile* — one strategy class. Pillars 6 and 7 are built to depth; the other ten ship as validated stubs from day one, so "incorporate all pillars later" means *fill in a YAML*, not *re-architect*.

These two principles drive every structural decision below.

---

## 1. The pillar-extensibility model — read this first

This is the heart of v2. Three layers of abstraction make all 12 pillars trivial to add.

### Layer 1 — the pillar config (what an indicator *is*)
Every pillar is one YAML file validated against a JSON schema. The engine reads indicator definitions, scoring criteria, keywords, clause patterns, and few-shot examples at runtime. **No engine code references "Pillar 6" or "cross-border data."**

### Layer 2 — the document profile (how the evidence is *shaped*)
Not all pillars live in the same kind of document. v2 makes this explicit with a `document_profile` field on every pillar. Each profile is a pluggable **strategy** (ingest + extract + chunk behaviour) registered in `rie-profiles`:

| `document_profile` | Pillars | Evidence shape | Status |
|---|---|---|---|
| `statutory_legal_text` | 6, 7, 8, 9, 12 (digital governance) | Acts, statutes, amendments — prose | **MVP — built** |
| `structured_tabular` | 1, 10, 11 (tariffs, NTMs, standards) | Tariff schedules, HS-code tables | Stub strategy — Phase 2 |
| `mixed_regulatory` | 2, 3, 4, 5 (procurement, FDI, IP, telecom) | Statute + registries + notices | Stub strategy — Phase 2 |
| `treaty_membership` | indicators 6.5, 12.10–12.13, etc. | Membership lists — a lookup, not extraction | Stub strategy — Phase 2 |

Adding the tariff pillars later = implement the `structured_tabular` strategy **once**; retrieval, classification, and verification never change.

### Layer 3 — the evaluation method (how an indicator is *answered*)
Some indicators are clause extraction; some are binary treaty lookups. Each indicator declares an `evaluation_method` (`clause_extraction` | `tabular_lookup` | `treaty_lookup`) so the orchestrator routes per-indicator, not just per-pillar.

**The net effect:** the 12 pillars are present as data structures from commit one. Pillars 6/7 are fully populated; the rest are schema-valid stubs. "Incorporate a pillar" = populate its YAML + gold set, and (if a new profile) implement one strategy. The walkthrough is §6.

---

## 2. Repository layout — the full tree

A uv **virtual workspace** (root has only workspace config, no app code). Everything pillar-related lives in top-level data directories so it is never tangled with engine code.

```
regulatory-intelligence-engine/
│
├── pyproject.toml                  # virtual workspace root: [tool.uv.workspace] only
├── uv.lock                         # single committed lockfile
├── .python-version                 # 3.12
├── .env.example
├── docker-compose.yml              # qdrant + postgres + model-serving + api + ui
├── Makefile                        # thin wrappers over uv commands
├── CLAUDE.md                       # repo-wide rules for every agent (§7)
├── ruff.toml  ·  pyrightconfig.json  ·  .pre-commit-config.yaml
│
├── .claude/
│   ├── agents/                     # 14 agent definitions, one .md each (§5)
│   ├── commands/                   # slash-command orchestration pipelines (§5)
│   └── settings.json               # shared permissions, hooks
│
├── docs/
│   ├── architecture_v3.md
│   ├── implementation_plan_v2.md   # this file
│   ├── adr/                        # architecture decision records, numbered
│   └── guides/
│       ├── adding_a_pillar.md      # ← the §6 walkthrough, kept current
│       └── adding_a_document_profile.md
│
├── pillars/                        # ═══ THE PILLAR KNOWLEDGE BASE ═══
│   ├── registry.yaml               # master index: every pillar → cluster, profile, status
│   ├── _schema/
│   │   ├── pillar.schema.json       # JSON schema; CI validates every pillar file against it
│   │   ├── indicator.schema.json
│   │   └── README.md                # the contract a pillar author must satisfy
│   ├── _template/
│   │   └── pillar_NN_template.yaml   # copy-to-create a new pillar
│   ├── digital_governance/         # cluster 1 — MVP depth
│   │   ├── pillar_06_cross_border_data.yaml
│   │   ├── pillar_07_domestic_data_protection.yaml
│   │   ├── pillar_08_intermediary_liability.yaml
│   │   ├── pillar_09_content_access.yaml
│   │   └── pillar_12_online_transactions.yaml
│   ├── traditional_trade/          # cluster 2 — stubs
│   │   ├── pillar_01_tariffs_trade_defense.yaml
│   │   ├── pillar_10_non_technical_ntms.yaml
│   │   └── pillar_11_standards_procedures.yaml
│   └── other_domestic/             # cluster 3 — stubs
│       ├── pillar_02_public_procurement.yaml
│       ├── pillar_03_foreign_direct_investment.yaml
│       ├── pillar_04_intellectual_property.yaml
│       └── pillar_05_telecom_competition.yaml
│
├── sources/                        # source registries — authoritative-source seeds
│   ├── _schema/source_registry.schema.json
│   └── jurisdictions/
│       ├── _template.yaml
│       └── country_a.yaml          # per-jurisdiction: gazettes, DPA portals, tiers
│
├── gold/                           # gold sets — one folder per pillar
│   ├── _schema/gold_item.schema.json
│   ├── pillar_06/  ·  pillar_07/    # populated
│   └── pillar_08/ … pillar_12/      # empty dirs, ready
│
├── data/                           # gitignored runtime data
│   ├── samples/                    # organiser-provided sample laws
│   ├── cache/                      # parsed-doc cache, keyed by sha256
│   └── outputs/                    # exported JSON/CSV result packages
│
├── packages/                       # ═══ THE ENGINE — uv workspace members ═══
│   ├── rie-contracts/              # ports + Pydantic models. Frozen first.
│   ├── rie-domain/                 # pure business logic. Imports only rie-contracts.
│   ├── rie-config/                 # loads + validates pillars/, sources/, gold/
│   ├── rie-profiles/               # ⭐ document-profile strategies — the all-pillar seam
│   ├── rie-ingest/                 # adapter: sample-law + source-registry loading
│   ├── rie-extract/                # adapter: Docling / PyMuPDF / VLM-OCR + structure graph
│   ├── rie-retrieval/              # adapter: BGE-M3 + Qdrant hybrid + reranker
│   ├── rie-classify/               # adapter: LLM classification + constrained decoding
│   ├── rie-verify/                 # adapter: 4-gate verification + citation builder
│   ├── rie-coverage/               # 3-state absence reasoning
│   ├── rie-orchestration/          # LangGraph graph wiring all stages
│   ├── rie-persistence/            # adapter: Postgres schema + repositories
│   ├── rie-api/                    # FastAPI driver adapter
│   ├── rie-ui/                     # Streamlit audit viewer
│   └── rie-eval/                   # gold-set runner, RAGAS metrics, ablations
│
├── tests/
│   ├── contract/                   # one test per port — the spec for parallel agents
│   ├── integration/                # full pipeline on sample laws
│   ├── architecture/               # fitness functions: dependency-direction enforcement
│   └── fixtures/                   # sample PDFs (born-digital, scanned), per-pillar
│
└── scripts/
    ├── bootstrap.sh                # one-shot: uv sync + docker compose up + migrate
    ├── new_pillar.py               # scaffolds a pillar from _template (§6)
    └── run_jurisdiction.py         # CLI entrypoint for a full run
```

### Internal structure of an engine package (hexagonal, identical for every package)

Every `packages/rie-*` follows the same shape, so an agent learns it once:

```
packages/rie-extract/
├── pyproject.toml                  # declares workspace deps (rie-contracts, rie-profiles)
├── README.md                       # what this package owns, its port, its seam
└── src/rie_extract/
    ├── __init__.py
    ├── service.py                  # the public entrypoint — implements the port
    ├── adapters/                    # concrete external-tool wrappers
    │   ├── docling_extractor.py
    │   ├── pymupdf_extractor.py
    │   └── vlm_ocr_engine.py
    ├── structure/                   # structure-graph builder (cross-refs, provisos)
    │   ├── graph_builder.py
    │   └── legal_numbering.py        # deterministic article/section grammar
    ├── router.py                    # picks adapter by document type
    └── errors.py
└── tests/
    ├── unit/
    └── conftest.py
```

The `rie-profiles` package is the exception worth detailing — it is the all-pillar seam:

```
packages/rie-profiles/src/rie_profiles/
├── base.py                         # DocumentProfileStrategy protocol
├── registry.py                     # profile-name → strategy class lookup
└── strategies/
    ├── statutory_legal_text.py      # BUILT — pillars 6,7,8,9,12
    ├── structured_tabular.py        # STUB — pillars 1,10,11
    ├── mixed_regulatory.py          # STUB — pillars 2,3,4,5
    └── treaty_membership.py         # STUB — treaty-lookup indicators
```

Adding a profile = a new file in `strategies/` + one line in `registry.py`. Nothing else in the engine moves.

---

## 3. The pillar config schema (what makes a pillar "data")

Every file in `pillars/` validates against `pillars/_schema/pillar.schema.json`. CI rejects a malformed pillar. The schema (described, not coded):

- `pillar_id`, `pillar_name`, `cluster`, `document_profile`, `status` (`built` | `stub` | `roadmap`).
- `indicators[]`, each with:
  - `indicator_id`, `name`, `definition`
  - `evaluation_method` — `clause_extraction` | `tabular_lookup` | `treaty_lookup`
  - `clause_pattern` — `obligation` | `prohibition` | `conditional_regime` | `exemption`
  - `scoring_criteria` — the 0 / 0.5 / 1 rubric strings
  - `positive_keywords`, `negative_cues` (multilingual-ready: keyed by language)
  - `few_shot_examples[]` — text + decomposition + label + score + rationale
  - `authority_hints` — which source tiers typically carry this indicator

`pillars/registry.yaml` is the master index — every pillar's id, cluster, profile, status, and gold-set path. The orchestrator iterates the registry; it never hard-codes a pillar list. Filtering "run only `status: built` pillars" is a one-line registry query.

**Day-one state:** pillars 6 and 7 fully populated; pillars 8/9/12 populated enough to demo config-extension; pillars 1–5, 10, 11 present as schema-valid stubs (`status: stub`, indicator skeletons, empty few-shots). The 12-pillar structure is *real* from the first commit — only the depth varies.

---

## 4. Build phases

| Phase | Work | Mode |
|---|---|---|
| **0 — Foundations** | uv workspace; all 16 package skeletons; `docker-compose.yml`; `CLAUDE.md`; ruff/pyright/CI; the `pillars/_schema/` + `_template/`; **write & freeze `rie-contracts`** + contract tests; scaffold all 12 pillar YAML files (6/7 deep, rest stubs) + `registry.yaml`. | Lead session, plan mode |
| **1 — Config & persistence** | `rie-config` (validates pillars/sources/gold against schemas); `rie-profiles` base + registry + the `statutory_legal_text` strategy; `rie-persistence` schema + Alembic migrations. | Lead session |
| **2 — Parallel adapters** | 9 agents build the adapter packages simultaneously, each in its own git worktree, each passing its contract test. | **All-Opus parallel** |
| **3 — Domain & surfaces** | `rie-domain`, `rie-coverage`, `rie-eval`, `rie-api`, `rie-ui` in parallel. | **All-Opus parallel** |
| **4 — Orchestration & integration** | `rie-orchestration` LangGraph wiring; end-to-end on sample laws; gold-set metrics; demo hardening (cached corpus). | Lead session — sequential |

---

## 5. The agent roster — 14 agents, all Opus 4.7 (high reasoning)

With a Max subscription, every agent runs **`opus-4.7` at high reasoning effort**. No tiering — the slowest, most capable model on every package, because correctness on a legal-AI system dominates everything else. The discipline that still matters is **disjoint file ownership** (cost is no longer the constraint, but file collisions and context bloat still are).

Each agent is a `.claude/agents/<name>.md` file: `model: opus-4.7`, high reasoning effort, a tool allow-list, and a first-line scope statement naming the **one package** it owns.

| # | Agent | Owns | Task |
|---|---|---|---|
| 1 | `contracts-agent` | `rie-contracts` | (Phase 0, lead-supervised) ports + models; frozen after review. |
| 2 | `config-agent` | `rie-config` | Pillar/source/gold YAML loading + schema validation. |
| 3 | `profiles-agent` | `rie-profiles` | Profile-strategy base, registry, `statutory_legal_text`; stub the other three. |
| 4 | `ingest-agent` | `rie-ingest` | Sample-law loader, source-registry parser, provenance capture. |
| 5 | `extract-agent` | `rie-extract` | Docling/PyMuPDF/VLM routing, OCR-error handling, structure graph. |
| 6 | `retrieval-agent` | `rie-retrieval` | BGE-M3, Qdrant hybrid + RRF, reranker, parent-document retrieval. |
| 7 | `classify-agent` | `rie-classify` | LLM classification, decomposition, constrained decoding, regime assembly. |
| 8 | `verify-agent` | `rie-verify` | 4 gates, NLI + second-LLM entailment, ID-replacement citations. |
| 9 | `coverage-agent` | `rie-coverage` | 3-state absence reasoning. |
| 10 | `persistence-agent` | `rie-persistence` | Postgres schema, repositories, migrations. |
| 11 | `api-agent` | `rie-api` | FastAPI endpoints. |
| 12 | `ui-agent` | `rie-ui` | Streamlit audit viewer, span highlighting, review queue. |
| 13 | `eval-agent` | `rie-eval` + `gold/` | Gold-set tooling, RAGAS harness, ablation runner. |
| 14 | `test-agent` | `tests/` | Integration tests + fixtures, written against frozen contracts. |

`rie-domain` and `rie-orchestration` are **lead-owned**, not agent-owned — the pure business logic and the final integration must stay coherent in one mind.

### Parallel execution mechanics
- **One git worktree per agent**, one branch per package — never two branches on one checkout.
- `uv sync` once at the shared root; worktrees share the resolved environment.
- Agents open PRs to `main`; because file ownership is disjoint, **PRs do not conflict** — merges are mechanical.
- Even with all-Opus and no cost ceiling: **stop idle agents** (they consume context and rate budget) and **clear context between tasks**.
- **Use `/caveman` on every agent invocation** — all 9 parallel adapter agents should run in caveman mode each time to cut token usage without losing technical accuracy.

### `.claude/commands/` orchestration
- `/scaffold` — Phase 0 skeleton generation.
- `/freeze-contracts` — runs contract tests + the dependency fitness function, then tags.
- `/build-package <name>` — invokes the owning agent end-to-end.
- `/add-pillar <id>` — runs `scripts/new_pillar.py`, then opens the pillar YAML for authoring (§6).
- `/integrate` — Phase 4 end-to-end run + metrics.

---

## 6. Adding a pillar later — the exact walkthrough

This is the payoff of the structure. To incorporate, say, **Pillar 8 (Internet Intermediary Liability)**:

1. **`/add-pillar 08`** → `scripts/new_pillar.py` copies `pillars/_template/pillar_NN_template.yaml` to `pillars/digital_governance/pillar_08_intermediary_liability.yaml` and registers it in `registry.yaml`.
2. **Substantive Lead populates the YAML** — indicator definitions, scoring criteria, keywords, few-shot examples. Pillar 8 is `document_profile: statutory_legal_text` → **no new strategy needed.**
3. **Drop a gold set** into `gold/pillar_08/`.
4. **Flip `status` to `built`** in `registry.yaml`.
5. Run `/integrate` — the orchestrator picks up Pillar 8 automatically.

**Files touched: three** (one YAML, one registry line, one gold folder). **Engine packages touched: zero.**

For a pillar needing a new profile — e.g. **Pillar 1 (Tariffs)**, `document_profile: structured_tabular`:
1–4 as above, plus **5.** implement the `structured_tabular` strategy in `rie-profiles/strategies/` (one file) and register it. Still: classification, retrieval, verification, persistence, UI — **all untouched.**

`docs/guides/adding_a_pillar.md` keeps this walkthrough current as the canonical reference.

---

## 7. CLAUDE.md — rules every Opus agent obeys

- **Model:** every agent is `opus-4.7`, high reasoning effort. No exceptions.
- **uv only** — `uv add`, `uv sync`, `uv run`. Never `pip`. Never hand-edit `uv.lock`.
- **Stay in your package** — edit only files under your assigned `packages/<name>/` (or `pillars/`, `gold/`, `tests/` for the data/test agents).
- **Contracts are frozen** — never edit `rie-contracts` after the freeze tag; if a contract seems wrong, stop and escalate.
- **Dependency direction** — `rie-domain` imports only `rie-contracts`; adapters import only `rie-contracts` (+ `rie-profiles` where relevant). CI fails on violation.
- **No pillar logic in engine code** — pillar behaviour comes only from `pillars/` YAML. A hard-coded indicator id in a `.py` file is a bug.
- **The LLM never authors a fact** — not a citation, not an authority tier, not a score. It selects from enums and emits span IDs; deterministic code does the rest.
- **Every adapter passes its contract test** before its PR opens.
- **Typed throughout** — Pydantic for data, `Protocol` for ports, pyright strict, ruff clean.
- **Python 3.12.**

---

## 8. Decoupling & scale properties this structure buys

- **Add a pillar → touch 3 files.** §6. The 12-pillar structure is real from commit one.
- **Add a document profile → one strategy class.** Tariff and treaty pillars slot in without touching retrieval/classify/verify.
- **Swap any provider → one adapter.** Docling→Unstructured, Ollama→vLLM, all behind ports.
- **Scale stages independently.** Stages talk through contracts + DB, so the heavy ones (extract, classify, verify) can later move behind a task queue with no contract change.
- **Test the core with zero infrastructure.** `rie-domain` runs against in-memory port fakes — no Qdrant, no GPU.
- **Architecture enforced by CI** — the `tests/architecture/` fitness function fails any build that violates dependency direction or hard-codes a pillar.

---

## 9. Non-negotiables

- **Freeze `rie-contracts` before spawning agents.** The basis of safe parallelism.
- **One agent, one package, disjoint files.** No exceptions.
- **Every agent is Opus 4.7, high reasoning.** Capability everywhere.
- **All 12 pillars exist as data from day one** — 6/7 deep, the rest validated stubs.
- **Pillar behaviour lives in `pillars/`, never in `.py`.**
- **Integration (Phase 4) is sequential and lead-owned.**
- **uv workspace, one lockfile, committed.**