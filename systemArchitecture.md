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
- **It is a public good ESCAP can run** — open-weights, permissive, Dockerised, local — and it survives past the hackathon as UN infrastructure, which is the reason this competition exists.