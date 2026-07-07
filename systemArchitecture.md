# Regulatory Intelligence Engine — System Architecture (v3)
### UN Global Hackathon on Using AI for Digital Trade Regulatory Analysis 2026

> An **evidence-extraction and verification engine** for RDTII digital-governance indicators. It maps clauses of real legal text to indicators with exact span-level citations and deterministic authority handling, verifies every claim, reasons honestly about coverage, and produces a human-reviewable audit package with a **recommended** score. **Reviewers retain final scoring authority.**

This v3 is a deliberate down-scoping and hardening of v2. It is driven by one fact established in the research: **citation hallucination cannot be eliminated by adding stages — only contained by architecture.** v3 therefore does fewer things, proves them, and is honest about the rest.

---

## 0. What changed from v2, and why (read this first)

Three independent critiques of v2 converged on the same verdict. v3 acts on all of it.

| v2 flaw | What the research says | v3 fix |
|---|---|---|
| Over-scoped: 6 stages, 6 gates, KG, discovery crawler — un-buildable in the window | Winning hackathon entries do fewer things well | **§1 MVP boundary.** A frozen "minimum viable winning system" — Pillars 6+7, sample laws, no crawler, 4 verification gates. Everything else is an explicit, labelled roadmap. |
| The **score** has no verification gate — system verifies grounding, not judgment | Selective-prediction research: confidence must be *screened* before use; abstention is the safety mechanism for high-stakes calls | **§6 two-layer output.** Layer 1 = *extraction* (verifiable, automatable). Layer 2 = *score* (a recommendation, always human-confirmed). The score is never auto-emitted. |
| Clause-centric — misses whole-law reasoning (exceptions in other sections, definitions, amendments) | Legal-RAG research: answering a legal question requires synthesising information scattered across sections; standard chunking severs cross-references | **§5 parent-document retrieval + §6.4 regime assembly.** Classification operates on a clause *plus its structural neighbourhood*, not an isolated chunk. |
| Absence reasoning rests on keyword recall — a "defensible 0" is not defensible | A `0` claim is only as strong as measured retrieval recall | **§7 honest absence.** No bare `0`. Output is "no evidence found in the searched corpus," scoped to a *measured* recall number from the gold set. |
| "Proven systems" table mislabels 3-month-old preprints as battle-tested | Real distinction between adopted infrastructure and recent literature | **§2 split table:** battle-tested infrastructure vs. techniques adapted from recent research. |
| Verifiers (NLI, second LLM) oversold as independent; "6 gates" inflated | Even fine-tuned attribution classifiers reach only ~80% macro-F1; legal entailment is *pragmatic*, not surface-level | **§6.5 honest verification:** 4 gates, each with a stated failure mode and a disagreement-resolution policy. |
| No legal narrative for a half-legal judging panel | — | **§9** the Substantive Lead's role is encoded in the system, not just the gold set. |
| No data model, no graceful degradation, self-arguing coaching notes | — | **§10 data model; §11 degradation; this is a clean architecture doc** — strategy notes are in a separate annex, never shown to judges. |

---

## 1. The MVP boundary — the frozen winning core

This is the single most important section. **Build only this for the prototype round. Resist all additions.**

**In scope (prototype — must be excellent):**
- Pillars **6 and 7 only**.
- Input = the **sample laws provided by the organisers**. No crawler. "Discovery" for the prototype is a folder of files plus a small manual source registry.
- Extraction: Docling + PyMuPDF. VLM-OCR only invoked if a sample is genuinely scanned.
- Parent-document hybrid retrieval + cross-encoder rerank.
- Constrained clause classification → indicator (Layer 1).
- **4-gate verification** (span-exists, verbatim-match, entailment, self-consistency).
- Two-layer output: verified evidence package + recommended score.
- One audit viewer (Streamlit). Side-by-side, Accept/Correct/Reject.
- A 40–60 clause gold set + measured metrics (§8).

**Explicitly roadmap (finale or beyond — labelled "Phase 2", never claimed as built):**
- Autonomous multi-jurisdiction discovery crawler.
- Pillars 8, 9, 12 by config; full 12-pillar config shipped.
- Knowledge-graph verification gate.
- Defensible absence scoring beyond the provided corpus.
- Full multilingual parity.
- React production UI.

A judge rewards a working core plus a credible roadmap over a promised platform delivered in fragments. **The roadmap is a feature, not an apology.**

---

## 2. Grounding — honestly labelled

The research is blunt that "proven" must mean *adopted*, not *published once*. Two tiers:

**Tier 1 — battle-tested infrastructure (years of production adoption):**
| Component | Role |
|---|---|
| **Docling** (IBM, MIT) / **PyMuPDF** | Document parsing, structure recovery, local execution |
| **BGE-M3** (BAAI) | Multilingual embeddings, dense+sparse |
| **Qdrant / pgvector** | Vector store |
| **Cross-encoder rerankers** (bge-reranker family) | Retrieval precision |
| **LangGraph** | Durable orchestration, human-in-the-loop interrupts |
| **vLLM / Ollama**, **Docker** | Serving, reproducibility |
| **RAGAS** | RAG evaluation (faithfulness, context precision/recall) |

**Tier 2 — techniques adapted from recent research (cite as *informing the design*, not as proven products):**
| Technique | Source | What we adapt |
|---|---|---|
| ID-replacement citation (model emits an ID; a non-LLM step swaps in the verified citation) | "ghost references" practitioner work, 2025 | Layer-1 citations are never model-generated text — see §6.6. |
| Claim-triplet / NLI fine-grained checking (RefChecker, ClaimVer) | EMNLP/arXiv 2024–2025 | The entailment gate decomposes a claim before checking it. |
| Span-level verification surfaced to the user (REFIND, SemEval 2025) | 2025 | Per-claim status shown in the audit viewer. |
| Selective prediction / abstention; screen confidence before using it | TACL 2025 abstention survey; SelectLLM; "Screen Before You Interpret" 2026 | §6.7 — confidence is *validated on the gold set* before any threshold is trusted. |
| Citation *existence* ≠ citation *support* | "Cited but Not Verified" 2026 | We check both: span exists AND span supports. |

**The central research finding v3 is built around:** even commercial legal AI with citation constraints hallucinates 17–33% of the time; documented citation-hallucination rates across deployed models run 11–57%; and in retrieval-augmented settings 3–13% of cited URLs are still fabricated. **Conclusion: do not promise "100% citation accuracy."** Promise something stronger and true — *every Layer-1 citation is mechanically verifiable, and every unverifiable claim is withheld from the reviewer's "verified" queue.* That is a claim you can defend on stage; "100%" is not.

---

## 3. Design principles

1. **Two layers, one honest boundary.** *Extraction* (which clause, where, verbatim) is automatable and verifiable. *Scoring* (is this 0.5 or 1?) is legal judgment and is always a recommendation. The architecture never blurs them.
2. **The model proposes; deterministic code disposes.** Every citation, every authority decision, every absence claim is finalised by non-LLM code. The LLM selects from enums and points at offsets; it does not author facts.
3. **Containment, not elimination.** Hallucination is a base rate, not a bug to be patched away. The job is to make every hallucination either *blocked* or *visibly flagged* — never silent.
4. **Absence is bounded by measured recall.** A "no evidence" output carries the gold-set recall number with it. Honesty is a feature the UN is explicitly buying.
5. **Pillar-agnostic engine, expert-authored config.** The engine is generic; the pillar YAML is a legal knowledge artdefact authored by the Substantive Lead. We say so plainly — the config is not a "free lunify," it is where the legal expertise lives.
6. **Reviewer-fast, not reviewer-replacing.** Success metric = expert minutes saved per indicator at a fixed accuracy bar.

---

## 4. End-to-end architecture (MVP)

```
            ┌────────────────────────────────────────────────────┐
            │  ORCHESTRATION  (LangGraph: durable state,          │
            │  HITL interrupts, full trace via Langfuse)          │
            └────────────────────────────────────────────────────┘
                                  │
   ┌─────────────┬────────────────┼───────────────┬──────────────┐
   ▼             ▼                ▼               ▼              ▼
┌────────┐  ┌──────────┐  ┌──────────────────┐ ┌──────────┐ ┌────────┐
│ INGEST │  │ EXTRACT  │  │  MAP + CLASSIFY   │ │  VERIFY  │ │ CONFIG │
│ sample │─▶│ Docling/ │─▶│  parent-doc       │▶│ 4 gates  │ │  KB    │
│ laws + │  │ PyMuPDF  │  │  retrieval →      │ │ + status │ │(pillar │
│ registry│  │ +struct  │  │  regime assembly  │ │          │ │  6,7)  │
└────────┘  └──────────┘  └──────────────────┘ └──────────┘ └────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
          ┌────────────────┐           ┌────────────────────────┐
          │ COVERAGE +      │           │  OUTPUT (two layers)    │
          │ ABSENCE (honest │──────────▶│  L1 evidence package    │
          │ 3-state)        │           │  L2 recommended score   │
          └────────────────┘           │  → AUDIT VIEWER (HITL)  │
                                        └────────────────────────┘
```

Each box is a typed LangGraph node (Pydantic in/out). State persists to disk; a crash resumes without re-work.

---

## 5. Ingest & Extract

### 5.1 Ingest (MVP = no crawler)
Input is the organisers' sample laws plus a small YAML **source registry** that records, per file: `source_url`, `jurisdiction`, `document_type` (statute / amendment / guideline), provisional `authority_tier`, `effective_date`, `sha256_hash`. The registry is a *governed artefact* — versioned, and signed off by the Substantive Lead — not a throwaway file. (Critique point: the registry is a single point of epistemic failure; treating it as governed is the mitigation.)

### 5.2 Extraction
- **Born-digital PDF / HTML → PyMuPDF (fast) then Docling (structure).** Text is taken from the PDF text layer; the tool recovers layout, reading order, tables. Honest scope: this is *lower-hallucination*, not "faithful" — layout artefacts (footnotes, schedules, strike-through amendment marks, side-notes) can still be lost or normalised. The correct claim is "structural preservation sufficient for most statutory text," and the **hallucinated-words metric** (words in output absent from source) is reported to prove it.
- **Scanned / image → VLM-OCR**, confidence-scored, `corrected` flag carried forward, human-gated. A VLM *generates* text and *can* hallucinate — this path is never called "faithful."

### 5.3 Structure graph (the whole-law fix, part 1)
Beyond layout, build a per-document **structure graph**: the article→section→paragraph tree, plus edges for cross-references ("subject to section 12", "notwithstanding subsection (3)"), definitions, provisos, schedules, and amendment markers. This graph is what lets §6.4 assemble a regime instead of judging an isolated clause. Cross-reference resolution is deterministic (regex + grammar over legal numbering), not LLM-guessed.

### 5.4 Output
Per document: an ordered element tree, each element with `text`, `element_type`, `page`, `bounding_box`, `char_offsets`, `extraction_confidence`, `ocr_engine`, `corrected`, plus the structure graph. Offsets + bounding box power the audit highlight.

---

## 6. Map, Classify & Verify

### 6.1 Parent-document retrieval (the whole-law fix, part 2)
Standard small-chunk retrieval severs legal context. v3 uses the proven **small-to-big / parent-document** pattern: embed small child chunks for precise matching, but when a child hits, hand the LLM the **parent element plus its structure-graph neighbourhood** (referenced definitions, linked exceptions, governing provisos). Research shows small-to-big retrieval is a ~65% win over baseline chunking at ~0.2 s extra latency — cheap, and it directly addresses the "exception in another section" failure.

- Embeddings: **BGE-M3** (multilingual). Hybrid dense + BM25, fused with **reciprocal rank fusion (k=60)**.
- **Cross-encoder rerank** over the top 20–40 → keep top 6–10. Reranking is the single highest-leverage precision lever (~59% MRR@5 improvement in the financial-RAG study); never skipped.

### 6.2 Clause decomposition
Each candidate clause is decomposed into logical elements — **subject, condition, constraint, context** (the proven Compliance-to-Code/FinCheck structure). This decomposition is stored, exported, and is what makes regime assembly and entailment checking precise.

### 6.3 Layer 1 — Extraction & classification (verifiable, automatable)
A reasoning LLM receives the clause + decomposition + structure-graph neighbourhood + the pillar's indicator definitions and `few_shot_examples` from config. Output is **constrained-decoded** (Pydantic + grammar decoder); the model selects `indicator_id` from a fixed enum — it **cannot invent a label**.

```json
{
  "clause_id": "doc12_art26_para1",
  "indicator_id": "6.4",
  "clause_pattern": "conditional_regime",
  "decomposition": {"subject": "organisations", "condition": "outbound transfer",
                    "constraint": "comparable-protection requirement"},
  "evidence_span_ids": ["doc12#4120-4215", "doc12#4216-4310"],
  "regime_members": ["doc12_art26_para1", "doc12_art2_def_personal_data"],
  "layer1_status": "pending_verification"
}
```

### 6.4 Regime assembly — and the Q1 PDPA case, correctly framed
v2's Q1 answer was *still* too confident. The corrected position, backed by the legal-RAG literature on fragmented information:

A general prohibition on outbound transfer + a comparable-protection exception is a **conditional flow regime → indicator 6.4** — that classification is solid. But the **restrictiveness score is a regime-level judgment**, not a clause-level read. It depends on the breadth of the adequacy list, how authorisation works in practice, sectoral carve-outs, and implementing regulations — evidence that may sit in other sections or other instruments.

So v3 separates the two cleanly:
- **Layer 1 (automatable):** detect the `general_rule + conditional_exception` structure via the structure graph, assemble all `regime_members`, classify as 6.4. Both spans kept, never discarded.
- **Layer 2 (recommendation only):** propose a score band (e.g. 0.5 vs 1) *with the regime evidence attached and the open questions listed* ("adequacy-list breadth not found in provided corpus → flagged for reviewer"). The system says: *"Structure = 6.4 conditional regime; recommended band 0.5–1; final restrictiveness requires regime-level evidence the reviewer should confirm."* Legally literate, and not overcommitted.

### 6.5 Verification — 4 gates, honestly characterised
Tiered: cheap deterministic gates on every claim; model gates only on survivors.

**Tier A — deterministic (every claim, ~zero cost, ~100% reliable):**
1. **Span-existence.** `char_offsets` must resolve to real text in the stored document. The snippet is re-extracted *by the system, by offset*. Mismatch → reject. *Failure mode: none significant — this is arithmetic.*
2. **Verbatim-match.** Exported snippet must be byte-identical to the re-extracted span. *Failure mode: none — enforces §6.6.*

**Tier B — model-based (survivors only, with stated limits):**
3. **Entailment check.** A separate verifier receives `(decomposed claim, cited span)` and judges support. **Honest limit (from the research): legal entailment is often pragmatic, not surface-level; fine-tuned attribution classifiers top out near 80% macro-F1.** Therefore the gate's question is deliberately narrow — *"is this claim directly supported by the cited span?"* not *"is this legally correct overall?"* It catches citation-shaped hallucination; it does not certify legal truth. Use an NLI model **and** a second LLM; **disagreement policy: any disagreement → `flagged` for human review** (never auto-resolved).
4. **Self-consistency.** Re-run classification N=3 with sampling variation **on a fixed input** (true self-consistency — varying the input conflates retrieval noise with model noise). Unstable `indicator_id` → `flagged`. *Honest limit: a model can be consistently wrong; consistency is an uncertainty signal, not a correctness proof — which is exactly why it only routes to review, never certifies.*

**The KG / entity-grounding gate is roadmap, not MVP.** A reliable legal KG needs legal-tuned NER; stock spaCy will add noise. Listed as Phase 2.

### 6.6 Citation construction — ID-replacement, never generation
Directly from the "ghost references" practitioner pattern. The LLM never writes a citation string. It emits a **span ID** (`doc12#4120-4215`). A deterministic post-step looks the ID up in the document store and **programmatically constructs** the full citation (law title, article, section, page, URL, effective date) from stored metadata. The model is structurally unable to fabricate a citation, because it never produces one. This is the strongest single mechanism in the system and the cheapest.

### 6.7 Per-claim status & screened confidence
Every claim exits as **`verified`**, **`flagged`**, or **`rejected`**. Crucially — per the selective-prediction research — **confidence thresholds are validated on the gold set before being trusted.** "Screen before you interpret": if the model's confidence signal does not actually discriminate correct from incorrect on the gold set, it is not used for routing at all, and everything non-trivial goes to review. We measure this; we do not assume it.

---

## 7. Coverage & Absence — honest by construction

The deepest v2 flaw. v3's rule: **the system never emits a bare `0`.**

Three states per `(jurisdiction, indicator)`:
- **`evidence_found`** — ≥1 `verified` claim; recommended score from evidence.
- **`no_evidence_in_searched_corpus`** — *not* a defensible RDTII `0`. It is exactly what it says: searched the available corpus, found nothing. It is exported **with the measured retrieval recall from the gold set attached** (e.g. "no evidence found; gold-set recall for this indicator = 0.82, so ~18% miss risk"). The reviewer decides whether that supports a real `0`.
- **`insufficient_coverage`** — authoritative sources unreachable or corpus too thin; exported as `null` + reason.

This reframes absence from a silent claim into a *measured, bounded* statement — and the audit viewer ranks `no_evidence` and `insufficient_coverage` items **at the top** of the review queue, because they carry the most hidden risk. Honesty here is the differentiator: a UN data team fears the confident false `0` above all else, and v3 structurally cannot produce one.

---

## 8. Evaluation

Accuracy must be a number, with honestly scoped claims.

**Gold set:** the Substantive Lead annotates 40–60 clauses from the sample laws with ground-truth `(indicator, score, span, authority_tier)`. State it as **"preliminary internal evaluation"** — 40–60 clauses cannot support a "demonstrated multilingual robustness" claim, and the memo must not pretend otherwise.

**Metrics — chosen for true failure costs, not generic RAG vanity:**
- Per-indicator precision / recall / F1 (Layer 1 classification).
- **Citation-support precision** — of claims marked `verified`, fraction a human confirms. The headline trust number.
- **Retrieval recall** (RAGAS context recall) — feeds the §7 absence statement.
- **False-`0` rate** — emitted `no_evidence` where a measure existed. The UN's most-feared error.
- **Authority-tier error rate** and **amendment-resolution error rate**.
- **Reviewer-override rate on `verified` claims** — the real measure of whether "verified" is trustworthy.
- Hallucinated-words rate (extraction); cost / latency per indicator.

**Ablations:** with/without reranker, entailment gate, parent-document retrieval — proves each component earns its place.

---

## 9. The legal narrative (for the half-legal judging panel)

The Substantive/Legal Lead is not a gold-set annotator bolted on — their expertise is **encoded in three system artefacts**:
1. **The pillar config YAML** — indicator definitions, scoring criteria, clause patterns, few-shot examples. This is a legal knowledge base; the Lead authors and owns it.
2. **The authority model** — the tiering rules, and critically the *jurisdiction-specific overrides* (§10) that correct the Western-statute default.
3. **The gold set and the scoring rubric** — the ground truth and the Layer-2 score bands.

The concept video and pitch must show this: the engine is the Technical Lead's; the *correctness* is the Substantive Lead's. That is the story a WTO/World Bank/EUI panellist needs to hear, and it maps directly onto the hackathon's required two-lead team structure.

---

## 10. Data model

For a system pitched as durable UN infrastructure, persistence is explicit (Postgres + pgvector):

```
documents(doc_id, jurisdiction, title, document_type, effective_date,
          authority_tier, source_url, sha256, retrieved_at)
elements(element_id, doc_id, parent_id, element_type, text,
         page, bbox, char_start, char_end, extraction_confidence,
         ocr_engine, corrected)
structure_edges(from_element, to_element, edge_type)   -- cross-ref, defines, proviso
claims(claim_id, indicator_id, clause_pattern, decomposition_json,
       regime_member_ids, layer1_status)
evidence_spans(span_id, claim_id, element_id, char_start, char_end, role)
verifications(claim_id, gate, result, detail, ran_at)
coverage(jurisdiction, indicator_id, state, measured_recall, reason)
reviews(claim_id, reviewer, decision, corrected_score, note, decided_at)
authority_overrides(jurisdiction, source_pattern, authority_tier, rationale)
```

Note `authority_overrides`: the §6 authority tiers are a *default*, and the research warning is real — "ministry guidance = non-binding" is wrong in many civil-law and regulator-driven jurisdictions. The override table, authored per jurisdiction by the Substantive Lead, is how the system avoids encoding a Western statutory bias into a global index.

---

## 11. Operations: cost, latency, degradation

**Cost / latency.** Do the arithmetic v2 hid. Per jurisdiction, Pillars 6+7: after retrieval, a bounded set of candidate clauses (deterministic gates discard most before any model call). Model passes per surviving claim: 1 classification + 3 self-consistency + 1 entailment ≈ 5. Tier the models — a small open-weight model for decomposition/routing, the large reasoning model only for classification + verification. Cache documents by `sha256`, embeddings, and identical clauses. **Publish the measured number** ("≈ X model-minutes per jurisdiction on one GPU") in the memo — and if it is too slow, cut self-consistency to N=1 + entailment only. A real number beats a hand-wave.

**Graceful degradation.** Explicit fallbacks: a source unreachable → `insufficient_coverage`, not a guess. A scan too poor to OCR → flagged, never silently low-confidence-passed. **For the live finale demo: run entirely on a pre-fetched, cached corpus** — never depend on a live government site during the pitch.

---

## 12. Tool stack (MVP)

| Stage | Tool | Licence |
|---|---|---|
| Ingest registry | YAML, governed | — |
| Extraction | PyMuPDF + Docling | open ✔ |
| Scanned fallback | a current open-weight VLM-OCR | open weights ✔ |
| Embeddings | BGE-M3 | open ✔ |
| Retrieval | hybrid + RRF(k=60) + bge-reranker | open ✔ |
| Classification LLM | a current open-weight reasoning model; hosted frontier model for dev only | verify licence per model ✔ |
| Verifier | NLI model + a second LLM (different family) | open ✔ |
| Constrained decoding | Pydantic + Outlines/XGrammar | open ✔ |
| Orchestration | LangGraph | open ✔ |
| Storage | Postgres + pgvector | open ✔ |
| Evaluation | RAGAS + Langfuse | open ✔ |
| Audit UI | Streamlit (one UI, carried to finale) | open ✔ |
| Serving | vLLM / Ollama + Docker Compose | open ✔ |

**Licensing must be verified per model, individually** — open-weight model licences vary by release and size; do not assume Apache 2.0. One non-permissive dependency caught by a judge is real damage on a hackathon that grades on open-source compliance.

---

## 13. Build sequence

**Prototype round (on sample laws):** ingest → extract + structure graph → parent-doc retrieval → Layer-1 classification (Pillars 6, 7) → 4-gate verification → two-layer output → Streamlit audit viewer → gold set + metrics. Deliverable: a working backend, machine-readable output with verified citations, the 10-minute demo.

**Finale round:** harden into a thin real discovery agent; add Pillars 8/9/12 by config (ship all 12 configs); add the KG gate if time allows; carry the same Streamlit UI forward (do not rebuild in React unless time is abundant); full Apache-2.0 repo, README, Docker Compose, documented open-weight swap path.

---

## 14. The core claim (use this exact framing everywhere)

> *"We built a pillar-agnostic evidence-extraction and verification engine for digital-governance regulatory indicators. It maps real legal text to RDTII indicators with exact span-level citations, deterministic authority handling, and whole-law regime assembly, and produces a human-reviewable audit package with a recommended score. Every citation is mechanically verified; every uncertain or unsupported claim is withheld and flagged. Reviewers retain final scoring authority. The engine is configuration-driven and extends across the digital-governance pillar cluster."*

Narrower than v2. Fully defensible. Demonstrable on what you can actually build. On this hackathon, **belief is most of the win** — and this claim is one no judge can puncture.

---

## 15. How v3 answers each application question

| Q | Answered by |
|---|---|
| Q1 — PDPA conflict / precedence / programming the AI | §6.4 — 6.4 conditional-regime classification (Layer 1, automatable) + regime-level score *recommendation* (Layer 2, human-confirmed); structure graph assembles the rule + exception even when separated. |
| Q2 — end-to-end workflow | §4 + §5–§7. |
| Q3 — data sources & scope for demo | §5.1 governed source registry; §5.2 routing covers HTML / clean PDF / scanned. |
| Q4 — evidence & citation method (anti-hallucination) | §6.6 ID-replacement citations; §6.5 4-gate verification; §6.7 per-claim status. |
| Q5 — three sources, authority, OCR errors, conflicts | §10 authority tiers + jurisdiction overrides; §5.2 OCR handling; §5.3 amendment/cross-ref resolution. |
| Q6 — implementable anti-hallucination design, allowed/not-allowed, evidence linkage, one failure case | §3 + §6: model selects from enums and emits span IDs only (allowed) / never authors citations or scores (not allowed); §6.6 linkage; the ministry-guideline case — guideline tagged non-binding, entailment gate finds the statute is a conditional regime not a ban → `flagged`, score withheld. |

---

## 16. Why v3 wins

- **It is buildable.** A frozen MVP a real team finishes, demos, and measures — the thing this hackathon actually grades.
- **It is honest where v2 overclaimed.** No "100% citations," no silent `0`, no "faithful" VLM path, no preprint mislabelled as proven. On an anti-hallucination project, calibrated honesty *is* the credibility.
- **It fixes the score gap.** Two layers: extraction is verified and automated; scoring is a recommendation a human confirms. The system never emits an unverified number into a UN dataset.
- **It reasons about the whole law.** Parent-document retrieval + structure graph + regime assembly — not isolated-clause guessing.
- **Anti-hallucination is architectural, not aspirational.** ID-replacement citations make fabrication structurally impossible; deterministic gates are ~100% reliable; model gates have stated limits and route to humans.
- **It has a legal story** for the half-legal panel, and a real data model for the durable-infrastructure pitch.

---

# v4 Update Plan — Agent-native, autonomous, citation-exact

> v3 is the *frozen winning core*: a verified, two-layer, pillar-agnostic engine. v4 does **not** discard it — it promotes the v3 ports/adapters into **tools** wired behind an autonomous **multi-agent** control plane, and tightens the two hard promises the rubric actually pays for: **knowledge discovery that is exhaustive-by-construction**, and **citations that cannot be wrong**. Everything below is additive; the 690 existing tests remain the regression floor.

This plan is written against the real submission spec we reverse-engineered from the brief: the **40 / 30 / 30** rubric (Substantive Accuracy / Technical Resilience / Architecture), the **13-column CSV + JSON** output contract that judges validate *programmatically*, the **Discovery-Tag NEW/KNOWN** differentiator, and the **real RDTII 2.1 indicator taxonomy** (Pillar 6: ban&local-processing, local-storage, infrastructure, conditional-flow; Pillar 7: lack-of-DP-framework, lack-of-cybersecurity, retention, DPIA/DPO, government-access — note the "Lack of…" inversions).

## 17. Why v4 (what v3 leaves on the table)

| Rubric / guideline pressure | v3 today | v4 closes it |
|---|---|---|
| **Discovery of NEW evidence beyond the sample kit** (the single biggest Substantive differentiator) | Source registry is hand-seeded; no autonomous breadth | Autonomous multi-strategy **Discovery swarm** + loop-until-dry + NEW/KNOWN diff against the Round-1 gold DB |
| **Output validated programmatically** | JSON shape is ours, not the judges' | **Conformance agent** emits the exact 13-col CSV + JSON, schema-checked in CI |
| **Framework alignment (40%)** | Pillar 6/7 YAML used invented indicator defs | Realign to the **authoritative RDTII 2.1** taxonomy + weights from the 130-page guide |
| **Live portal crawl, HTML harder than PDF** | Crawler exists but thin; HTML extraction weak | **Portal-Navigator agent** with HTML-DOM + PDF + scanned routing; freshness/amendment resolver |
| **"Better than Perplexity" citations** | Strong (verbatim gate) but single-pass | **Cite-then-verify two-pass + adversarial refutation panel**; nothing emitted unless it survives |
| **Autonomy / "no manual steps"** | LangGraph linear pipeline | **Planner-supervised agent graph** that decomposes, schedules, budgets, and self-heals |
| **Cost-efficiency (measured)** | Tracing exists | **Telemetry agent** writes a measured per-document cost report judges can verify |

## 18. North-star: three guarantees v4 must make true

1. **Discovery is exhaustive-by-construction.** For every `(economy × pillar × indicator)` cell, the system either returns evidence, or returns a *measured* absence state — never a silent gap. Completeness is enforced by a critic loop, not hoped for.
2. **Every emitted citation is byte-true and claim-supporting.** The "answer" is never generated prose; it is the **statutory span itself**, re-extracted byte-identically and entailment-checked against the indicator's legal question. A wrong citation is structurally impossible, not merely unlikely.
3. **The machine proposes, the human disposes.** Layer-1 extraction is automated and verified; Layer-2 scoring is a recommendation with `human_confirmation_required = True`. No unverified number ever enters a UN dataset.

> Honesty boundary (kept from v3): "100% correct" is a guarantee about **what is emitted**, not a claim of omniscient recall. Citation fidelity is deterministically ~100% because the gate is byte-equality. Discovery recall is driven toward 100% by the critic loop and **reported with a measured gold-set recall number** — we make the absence *defensible*, we never fake the presence.

## 19. Agent-native architecture (ground-up)

A **supervisor/planner** agent owns a typed **blackboard** (Postgres + Qdrant + a light structure/knowledge graph) and dispatches specialist agents over a LangGraph graph. Every agent is a *tool-calling* worker with a constrained-output contract; every adapter from v3 (ingest, extract, retrieval, classify, verify, coverage, persistence) is exposed as a **tool** in a shared registry. Agents never free-write facts — they call tools and select from enums, exactly as in v3 §3.

```
                         ┌──────────────────────────────────────────────┐
            user/cron →  │  PLANNER · supervisor                        │
                         │  decompose (economy×pillar×indicator) →       │
                         │  budget · schedule · self-heal · stop-rules   │
                         └──────────────┬───────────────────────────────┘
                                        │   (writes tasks to blackboard)
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                ▼               ▼                ▼               ▼
  DISCOVERY swarm   SOURCE-VALIDATOR  EXTRACTION      RETRIEVAL       CLASSIFIER
  (4 blind          authority tier·   profile-routed  hybrid parent- constrained
   strategies)      freshness·dedup·  Docling/PyMuPDF doc + cross-   decode → real
   ↓ candidates     NEW/KNOWN tag     /VLM-OCR/HTML   lingual        RDTII indicator
        └───────────────┬───────────────┴───────────────┬───────────────┘
                        ▼                                 ▼
                 VERIFIER PANEL  (adversarial, N skeptics)        COVERAGE / ABSENCE
                 4 gates + refutation vote → verified/flagged     3-state + defensible-0
                        │                                          + COMPLETENESS CRITIC
                        ▼                                                  │ "what's missing?"
                 OUTPUT / CONFORMANCE agent                               └─► re-queues Planner
                 exact 13-col CSV + JSON, schema-validated
                        │
                        ▼  (FLAGGED → human review interrupt; TELEMETRY agent logs cost/trace throughout)
                 audit package + measured cost report
```

**Agent roster (each = constrained tool-caller, not a free-text oracle):**

| Agent | Job | Tools it calls | Output contract |
|---|---|---|---|
| **Planner / Supervisor** | Decompose the run into indicator-cells; budget tokens; schedule; apply stop-rules; route FLAGGED to humans | blackboard, registry | task DAG |
| **Discovery swarm** (×4, blind) | Find candidate official sources four independent ways (see §20) | portal-nav, search-API, seed-registry, treaty-list | candidate URLs + provenance |
| **Source-Validator** | Authority tier; official-source gate; **freshness/amendment** resolution to the in-force version; dedup; **NEW/KNOWN** tag vs Round-1 DB | http, registry, KG | validated source records |
| **Extraction** | Route by profile → elements + structure graph | Docling, PyMuPDF, VLM-OCR, HTML-DOM parser | `Element[]` + edges |
| **Retrieval** | Parent-document hybrid (BGE-M3 + BM25 + RRF + rerank), cross-lingual query expansion | Qdrant, embedder, reranker | `RetrievalHit[]` |
| **Classifier** | Constrained-decode the clause → **real** RDTII indicator + decomposition + regime assembly | LLM (guided JSON), structure graph | `Claim` (span IDs only) |
| **Verifier panel** | 4 gates + **adversarial refutation** (§21) | NLI, 2nd-LLM, deterministic span tools | `VerificationReport` |
| **Coverage / Critic** | 3-state absence + defensible-0; **completeness critique** re-queues discovery until dry | gold-recall, blackboard | `CoverageRecord` + missing-list |
| **Scorer** (Zone 3, optional) | Layer-2 recommendation only | rubric config | `Layer2Recommendation` |
| **Conformance** | Materialise the exact 13-col CSV + JSON; validate schema | citation builder, schema validator | submission files |
| **Telemetry** | Measured per-doc cost + latency + trace (Langfuse) | tracer | cost report |

## 20. 100%-correct knowledge discovery (exhaustive-by-construction)

Discovery is the Substantive differentiator, so v4 treats it as a *coverage problem with a proof obligation*, not a single search.

1. **Multi-modal sweep — four blind strategies.** Run concurrently, each unaware of the others (so each surfaces what the others miss):
   - **Portal-Navigator** — crawl the official legislation portal hierarchy (e.g. `sso.agc.gov.sg`, `legislation.gov.au`, `federalgazette.agc.gov.my`), follow consolidation/amendment links, render JS where needed.
   - **Search agent** — query search APIs scoped to official domains + known legal aggregators; promote `.gov` / official-gazette hits.
   - **Seed-registry** — the governed `sources/` registry (KNOWN baseline).
   - **Treaty/membership agent** — the non-regulatory set (6.5, 12.10–12.13, etc.) resolved against official instrument lists (these are *lookups*, never extracted as clauses).
2. **Corroboration before trust.** A candidate law is admitted only when the **Source-Validator** confirms it on an official source and resolves it to the **in-force version**. The amendment chain is followed explicitly — this is exactly where ESCAP's own trial failed (Singapore Patents Act shown as the 2021 revision when Act 5 of 2025 governed; Malaysia PDPA 2010 superseded by Act A1693/2024 adding DPO/breach). v4 captures `last_amended` and flags staleness.
3. **Dedup + NEW/KNOWN.** Every admitted provision is diffed against the Round-1 gold DB by `(law, indicator, span)`. In-kit ⇒ `KNOWN`; independently found valid provision ⇒ `NEW` (the points multiplier).
4. **Loop-until-dry.** The **Completeness Critic** asks, per indicator cell, *"what official source type have we not yet searched, what amendment have we not followed, what sector-specific law could also carry this obligation?"* Discovery is **not done** until *K* consecutive critic rounds surface nothing new. Silent truncation is logged, never hidden.
5. **Coverage ledger.** Every `(economy, indicator)` cell ends in exactly one explicit state — `evidence_found` / `no_evidence_in_searched_corpus` (with measured gold-set recall) / `insufficient_coverage` — so a missing answer is always an *accounted-for* state, never a gap. Defensible-zero only when recall ≥ floor **and** authoritative-source reachability ≥ floor.

## 21. Citation correctness that beats Perplexity

Perplexity-class tools **generate an answer, then attach citations post-hoc** — so the citation can fail to support the sentence. v4 **inverts** the pipeline so that failure mode cannot exist:

- **The claim is a span, not prose.** What we emit *is* the statutory text (`verbatim_snippet`), addressed by a span ID `doc#start-end`. There is no generated sentence for a citation to mismatch.
- **Cite-then-verify, two-pass.** Pass 1 (Classifier) emits only **span IDs + an enum indicator choice** — never a citation string, never a score (v3 §6.6). Pass 2 (Verifier) materialises and re-checks before anything is emitted:
  1. **Span-existence** (deterministic) — offsets resolve to real stored text.
  2. **Verbatim-match** (deterministic) — exported snippet is **byte-identical** to a fresh re-extraction at those offsets. This is the gate that makes a wrong citation *impossible*.
  3. **Entailment** (NLI + a second, different-family LLM) — the span actually answers the indicator's legal question; any disagreement → `flagged`.
  4. **Self-consistency** (N-sample on fixed input) — unstable label → `flagged`.
- **Adversarial refutation panel (new in v4).** Before a claim is marked `verified`, *N* skeptic agents are each prompted to **refute** the mapping ("show why this span does NOT evidence this indicator"), with diverse lenses (wrong-indicator, confidentiality-≠-localisation, government-data-scope-exception, draft-not-in-force). Survive a majority → `verified`; otherwise `flagged` and withheld. This directly kills the misinterpretation class from Assignment 1 (Banking Act confidentiality mis-mapped as a transfer ban; MAS notice mis-read as "lack of framework").
- **Source re-fetch at verify time** catches dead/changed URLs (a named failure mode), and `source_url` is checked to resolve.
- **Calibrated confidence + withholding.** Below threshold → `flagged`, never shipped as `verified`. The audit row carries `raw_context_before/after` so a human closes the loop in seconds.

> Net claim, defensible in front of any judge: *every Layer-1 citation we emit is mechanically re-verified to be byte-identical to its official source and to entail the indicator's legal question; anything that cannot pass is withheld and flagged, never paraphrased into a confident answer.* That is a stronger guarantee than any generate-then-cite system can make.

## 22. Output-contract conformance (judge-validated)

The **Conformance agent** is the only writer of submission artefacts and the contract is enforced in CI:

- **CSV — exact 13 columns, exact order**: `economy, law_name, law_number_ref, last_amended, indicator_id, article, discovery_tag, location_reference, verbatim_snippet, mapping_rationale, source_url, confidence, notes`. One row per provision; one article→two indicators ⇒ two rows.
- **JSON** mirrors CSV + technical metadata: `source_pdf_path, ocr_quality_cer, processing_time_seconds, model_version, pdf_is_scanned, retrieval_method, raw_context_before/after, provisions[]`.
- **CLI**: `python main.py --economy Singapore --pillar 6` → `outputs/Singapore_P6_<ts>.csv` + `.json` + `logs/`. `--pdf` bypasses the crawler; `batch_run.py` covers many economies.
- **Indicator IDs** emitted in the judges' `P{pillar}-I{n}` display form, derived from the authoritative decimal taxonomy, with the decimal retained in JSON. (The format ambiguity between template and gold DB is resolved in favour of the gold DB and surfaced in `notes`.)
- A schema test fails the build if any column is renamed, reordered, paraphrased, or missing a required field.

## 23. Evaluation & self-improvement

- **Gold harness** runs the engine against the Round-1 DB (Singapore / Australia / Malaysia, Pillars 6 & 7) and reports **per-(indicator, law) precision / recall / F1**, field accuracy, and inter-annotator agreement — scored per `(indicator, law)` because one indicator legitimately maps to many provisions.
- **Citation-fidelity rate** (verbatim-gate pass %), **hallucinated-words rate**, **reviewer-override rate**, and **false-zero rate** are tracked as first-class metrics.
- **Self-improvement loop**: failing gold cells feed the Completeness Critic, which re-queues discovery/extraction with adjusted strategy — a measurable convergence curve to show judges, not a static number.

## 24. Modern stack additions (on top of v3 §12)

LangGraph multi-agent supervisor + Postgres checkpointer (durable, resumable runs, HITL interrupts) · tool registry with typed schemas · structured/guided decoding for every agent · optional KG (entity-grounding advisory gate) · Langfuse tracing · pluggable LLM (Ollama/vLLM/OpenAI/Mistral via one config) and OCR (Tesseract/PaddleOCR/VLM/Azure) — **no vendor lock-in**, the Architecture-block requirement. Apache-2.0.

## 25. Migration path (additive, low-risk)

1. **Wrap, don't rewrite.** Expose each v3 adapter as a registry tool; the 690 tests stay green as the regression floor.
2. **Conformance + taxonomy first** (highest rubric ROI, lowest risk): emit the exact 13-col output and realign Pillar 6/7 YAML to the authoritative RDTII taxonomy + weights.
3. **Discovery swarm + NEW/KNOWN**: the Substantive differentiator.
4. **Adversarial verifier panel + source re-fetch**: hardens the citation guarantee.
5. **Planner supervisor + Completeness Critic**: turns the linear pipeline into an autonomous, self-healing graph.
6. **Real economies** (Singapore/Australia/Malaysia) with measured gold F1 + cost report; then scale to the 8 final-round economies.

## 26. Why v4 wins (beyond v3)

- **It is autonomous, not scripted** — a planner decomposes, budgets, and self-heals; "no manual steps" is literally true.
- **Discovery is a proof, not a hope** — multi-strategy sweep + loop-until-dry + a coverage ledger that can *defend* every absence.
- **Citations cannot be wrong** — the claim *is* the verified span; an adversarial panel must fail to refute it before it ships. Strictly stronger than generate-then-cite.
- **It conforms to exactly what the judges grade** — the 13-column contract, the NEW/KNOWN tag, the real indicator taxonomy, the measured cost.
- **It is still honest** — calibrated "100%": deterministic citation fidelity, *measured* discovery recall, human-confirmed scores. The credibility that wins an anti-hallucination hackathon.
- **It is a public good ESCAP can run** — open-weights, permissive, Dockerised, local — and it survives past the hackathon as UN infrastructure, which is the reason this competition exists.