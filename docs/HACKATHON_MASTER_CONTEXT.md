# HACKATHON MASTER CONTEXT — ESCAP RDTII 2.1 Regulatory Intelligence Engine

> Persistent memory file. Synthesizes the **entire** `Hackthon Knowledge Products/` corpus
> (workshops June 1/4/5/11/12/15, assignments + answer key, Round 1 & 2 databases, framework PDFs,
> sample legislation, submission templates, reference CSVs) plus current repo state. Source of truth
> for building the RIE. Do not delete.
>
> Last refreshed from a 12-agent deep re-read (Opus 4.8, "very thorough") of all 66 corpus files.
> Score convention everywhere: **0 = open/simplified, 1 = restrictive/heavily-regulated** (higher = more compliance cost).

---

## 0. TL;DR — what we are building

Automate **Zone 1 (source identification)** + **Zone 2 (clause extraction & mapping)** of the ESCAP RDTII 2.1
data-collection workflow for **Pillar 6 (Cross-border Data Policies)** and **Pillar 7 (Domestic Data
Protection & Privacy)** across **Singapore, Australia, Malaysia**.

- **Round 1 deadline:** 20 July 2026, **Bangkok time**. **Zone 3 (scoring) = optional bonus** — emit only as a *recommendation*, `human_confirmation_required = True`.
- **5 deliverables** (4 uploaded by 20 Jul via one online form + live demo on 3 Aug for shortlisted teams).
- **Submission scoring:** 40% substantive accuracy / 30% technical resilience / 30% architecture.
- **Biggest point levers:** **NEW discovery beyond the sample kit (20 of 40 substantive pts)**, live portal
  crawling (not only pre-downloaded PDFs), OCR error <5%, modular LLM/OCR swap (config-only), verbatim
  audit trail per row, cost-efficiency (measured), handling both PDF **and** HTML, anti-bot resilience.
- **Goal stated by ESCAP:** refresh the RDTII database every 6–12 months by automating the tedious desk
  research — **NOT to replace researchers**; "human-in-the-loop remains very, very essential."

### 0.1 ⚠️ Hard environment constraints (June 15 Q&A — these override earlier assumptions)
- **Runs on a standard CPU box.** Do **NOT** assume GPU. Do **NOT** assume unrestricted internet. **No private API key is provided.** No secret credentials in the repo. External services allowed but must be documented + cost-justified in the README.
- **CPU-friendly stack matters:** BGE-M3 (~2 GB) runs on CPU; prefer local/open-weight LLMs (Llama/Qwen/Gemma) as the swappable default; commercial APIs only as an optional, config-gated path with an open-source fallback.
- **Judges run it.** Repo must clone + set up + run **end-to-end in <10 minutes** from the README with no team help. Judges do **document-by-document** verification (URL alive? document exists? interpretation correct against indicator criteria?) → **design a document-by-document / HITL mode**, batch is fine too but must prove end-to-end automation.
- **Speed is scored:** faster = higher efficiency score (no fixed time budget given).

---

## 1. RDTII 2.1 Framework (authoritative — from the 3 framework PDFs + Methodology sheet)

### 1.0 Structure & scoring model
- **12 pillars → ~70 policy indicators → 3 clusters.** Composite and every indicator run **0 → 1**.
- **Indicator → pillar aggregation uses differential indicator weights** (from the 2018 DTRI study); **pillar → composite uses equal weighting** (simple average of 12 pillars).
- A score **> 0** signals one of: (1) complex requirements raising compliance burden (may hit foreign *and* domestic firms); (2) differential treatment (domestic vs foreign, or online vs offline); (3) failure to adopt a recognized international norm/agreement.
- The index measures the **extent** of regulation (compliance cost / friction) — **explicitly not** a value judgment on policy quality.

### 1.1 Regulatory vs non-regulatory split
- **14 non-regulatory indicators** are sourced from external databases / treaty-status lookups and are **removed from the sample database** — the automation tool is **NOT** required to extract them:
  - **WITS** (tariffs): `1.1`, `1.2` · **V-Dem** (`v2smgovshut`, internet shutdowns): `9.2`
  - **Treaty / signatory lookups:** `1.3` (WTO ITA), `2.4` (WTO GPA), `4.4` (WIPO PCT), `4.7` (WCT), `4.8` (WPPT), `5.6` (WTO Telecom Ref Paper), `6.5` (binding data-transfer agreement / TAPED), `12.10` (UN e-Comms Convention), `12.11`–`12.13` (UNCITRAL model laws).
- **Enforcement/practice indicators** (regulatory, but secondary/de-facto sources ARE recorded and scored for these three only): **`3.4`** (investment screening actually used), **`5.3`** (government telecom shareholding), **`9.1`** (commercial-content blocking/filtering).
- **All other indicators are regulatory** → extract clauses from official **primary** sources; secondary sources only lead you to the primary (park them in the Note column).

### 1.2 The 12 pillars + scope (sample-DB regulatory subset = 61 indicators)

Full framework ≈70 indicators; **sample database scope = 61 regulatory indicators.** **Indicator-ID gaps are intentional** (they are the carved-out non-regulatory IDs) — never chase missing IDs (`1.1–1.3`, `2.4`, `4.4/4.7/4.8`, `5.6`, `6.5`, `9.2`, `12.10–12.13`).

| Pillar | Name | # reg. ind. | Sample-DB indicator IDs |
|--------|------|-------------|--------------------------|
| 1 | Tariffs and Trade Defence | 1 | `1.4` |
| 2 | Public Procurement | 3 | `2.1`–`2.3` |
| 3 | Foreign Direct Investment | 5 | `3.1`–`3.5` |
| 4 | Intellectual Property Rights | 7 | `4.01`, `4.1`, `4.2`, `4.3`, `4.5`, `4.6`, `4.9` |
| 5 | Telecom Regulations & Competition | 6 | `5.1`–`5.5`, `5.7` |
| **6** | **Cross-border Data Policies** | **4** | **`6.1`–`6.4`** |
| **7** | **Domestic Data Protection & Privacy** | **5** | **`7.1`–`7.5`** |
| 8 | Internet Intermediary Liability | 4 | `8.1`–`8.4` |
| 9 | Content Access | 3 | `9.1`, `9.3`, `9.4` |
| 10 | Non-technical NTMs | 4 | `10.1`–`10.4` |
| 11 | Standards and Procedures | 4 | `11.1`–`11.4` |
| 12 | Online Sales and Transactions | 15 | `12.01`, `12.2`–`12.9`, `12.4.1`–`12.4.7` |

- **3 clusters:** (1) Traditional trade measures, (2) Other domestic policies, (3) **Digital governance measures** (contains Pillars 6 & 7 — most relevant to the hackathon).
- **IP numbering caveat:** the framework Guide lists `4.1` (patent application) … `4.10` (trade secrets); the sample DB stores `4.01` + `4.1` (with `4.1` sorting after `4.9`). Treat the DB IDs as the operational ones; reconcile to Guide titles when in doubt.

### 1.3 Pillar 6 — Cross-border Data Policies (MANDATORY) — weights + rubrics

Pillar-wide carve-out: **measures applied to government data are not scored.**

| ID | Policy issue | Weight | Scores | Rubric / classification |
|----|-------------|--------|--------|-------------------------|
| 6.1 | Ban / local-processing requirement | 38% | 1 / 0.5 / 0 | personal data OR horizontal → 1; specific sector/non-personal/one-country → 0.5; free transfer → 0 |
| 6.2 | Local storage requirement | 12% | 1 / 0.5 / 0 | a copy must stay in-country; does NOT itself ban transfer |
| 6.3 | Infrastructure requirement | 31% | 1 / 0 | must use/establish local server/data-centre as condition of service |
| 6.4 | Conditional flow regime | 12% | 1 / 0.5 / 0 | transfer allowed only if conditions met (consent/adequacy/contract/approval) |

- **CRITICAL traps:** 6.1 (ban / forced local processing) ≠ 6.4 (conditional flow — transfer still legally possible). 6.2 (a copy stays) ≠ 6.3 (must build local infrastructure). Classify by **semantic function, not keywords**.
- Canonical examples: 6.4 = Singapore PDPA **s.26** / Armenia DPL Art.27 (consent+adequacy); 6.2 = Kazakhstan Art.12(2) (store in-country); 6.3 = China ride-hailing Art.5 (servers in Mainland China); 6.1 = Korea financial cloud local-processing.

### 1.4 Pillar 7 — Domestic Data Protection & Privacy (MANDATORY, "Lack of …" framing)

Pillars 7.1/7.2 are **inverted**: a framework *existing* scores **toward 0**.

| ID | Policy issue | Weight | Scores | Rubric / trap |
|----|-------------|--------|--------|---------------|
| 7.1 | Lack of comprehensive data-protection framework | 31% | 1 / 0.5 / 0 | no framework → 1; sectoral-only → 0.5; comprehensive horizontal → 0 |
| 7.2 | Lack of dedicated cybersecurity framework | 31% | 1 / 0.5 / 0 | none → 1; non-dedicated/sectoral → 0.5; dedicated horizontal → 0 |
| 7.3 | Minimum data-retention requirement | 16% | 1 / 0 | minimum specified → 1; none / "as long as necessary" → 0 |
| 7.4 | DPIA / DPO requirement | 6% | 1 / 0.5 / 0.25 / 0 | DPO (±DPIA) horizontal → 1; sectoral → 0.5; DPIA-only → 0.25; none → 0 |
| 7.5 | Government access to personal data w/o independent judicial authorization | 16% | 1 / 0 | any such power → 1; none → 0 |

- **7.3 trap:** "minimum retention" (keep *at least* N years → 7.3=1) ≠ "do not keep longer than necessary" (a *maximum*/limitation → not scored, 7.3=0). Government-data retention excluded.
- **7.5:** look beyond privacy law — criminal procedure, surveillance, lawful-access, national-security, telecom statutes (e.g. Singapore CPC 2010 ss.39–40).
- **7.5 is MANDATORY and scored** (June 15 Q&A). **6.5 is OUT OF SCOPE** (non-regulatory, removed from the rubric — do not include it).

### 1.5 Scoring scales across the 61 sample-DB indicators
- 31 indicators: `1 / 0.5 / 0` · 23 indicators: `1 / 0` binary.
- `1 / 0.8 / 0.5 / 0`: `3.1`, `5.2` (foreign equity) · `1 / 0.5 / 0.25 / 0`: `3.4`, `5.4` · `1 / 0.75 / 0.5 / 0.25 / 0`: **`1.4` only** (+0.25 per trade-defence measure, cap 1) · `7.4` adds a `0.25` step.
- Observed raw values in data: **{0, 0.25, 0.5, 0.75, 0.8, 1}**. Score must come from the indicator's own rubric set — not arbitrary floats.
- **Explicit Exception/carve-out clauses are pervasive** (6.1–6.4 & 7.3 govt-data; 9.1 political/criminal/IP content; 9.3 misleading-ad & product-specific; 3.1 telecom→5 / e-comm→12; 12.2 consumer-protection goods; 12.8 corporate-registration; 4.3/4.9 TRIPS exceptions; 10.x raw materials; 11.4 mere licensing). These are **hard exclusion rules → must live in `pillars/*.yaml`, never Python `if`.**

### 1.6 Absence / official-silence handling (framework rule)
- "Lack of regulatory measures" → **note the absence AND cite the relevant general law** as reference, state the reason in the **Impact** column, score `0`. A `0` is "evidence found that no restriction exists," **never a silent/blank zero**.
- De jure vs de facto: walk down the hierarchy (Act → Decree → Rule) before concluding absence.

---

## 2. The Databases (ESCAP gold corpus)

Two workbooks; **Methodology sheet byte-identical in both** (75 rows).

| File | Sheets | Economies |
|------|--------|-----------|
| Round 1 | Methodology, **Consolidated**, Australia, Malaysia, Singapore | 3 |
| Round 2 | Methodology, China, India, Indonesia, Lao PDR, Mongolia, Russian Federation, Thailand | 7 |

### 2.1 Jurisdiction sheet schema (an evidence store, NOT one-row-per-indicator)

| Col | Header | Role |
|-----|--------|------|
| A | `Pillar_ID` | 1–12 (only on pillar-header rows) |
| B | `Indicator_ID` | `X.Y` code **or** pillar name on header rows |
| C | `Raw Score` | per-evidence score (a recommendation) |
| D | `Act and/or practice` | law/regulation/practice (often several, newline-separated) |
| E | `Coverage` | sector: `Horizontal` dominant, else `Telecommunications`, `E-commerce`, `Financial`, HS codes… |
| F | `Impact or comments…` | **primary evidence text** — legal analysis with section cites |
| G | `Timeframe` | enacted / last-amended dates ("Since [Mon Yr], last amended [Mon Yr]") |
| H | `References` | **first** URL |
| I–M | merged under "References" (unlabeled) | **up to ~5–6 URLs per row** (AU & India hit 6 cols) — parsers must treat H→(col before Note) as ONE multi-valued field, never skip |
| last | `Note` | analyst caveats, cross-refs, ITA exclusions |

- **Two row types:** 12 pillar-header rows per sheet (no score) + indicator evidence rows (filter `Indicator_ID` starting with a digit).
- **One indicator → many evidence rows** is normal. Expect **70–185 rows per economy**, not 61 (India `7.3` peaks at 19 rows; Indonesia has 172 evidence rows).
- **Score 0 still carries a written absence narrative** in Impact — never a blank/silent zero. Score `0` is the modal value in 9 of 10 sheets.
- **Round 2 government-QA columns** (Thailand richest): `Data verification feedback` (`Correct`/`Not correct`), `…(full)`, `Actions` (e.g. "Maintain score & update text", "Change the score to 0.00", "Add … as new row", "TO CHECK"); Indonesia adds "Policy change or Correction"; Lao/Russia add verification-question columns. Maps to RIE `human_confirmation_required` + flagged states. Treat Round 2 as a curated/reviewed eval set, not raw extraction targets.
- **Round 1 `Consolidated`** = vertical stack of AU+MY+SG with a leading `Country` column (not a pivot).
- **⚠️ Data-quality artifacts for the ingester:** some scores stored as the string `'1.00\n'` (China, Russia) → normalize; Singapore uses non-breaking spaces `\xa0` in blank cells → treat as empty; Mongolia has ~27 blank spacer rows → skip; continuation rows sometimes blank the repeated `Indicator_ID` (why a sheet may report 62 "distinct" IDs).

### 2.2 Companion CSVs (KNOWN coverage baseline — NOT validated citations)
- **Legal Inventory** (`Singapore, Malaysia, Australia, Legal Inventory.csv`): **384 logical rows**, 10 cols, 3 clusters / 12 policy areas (MY 146, SG 131, AU 107). **91 rows are directly P6/P7** (53 domestic data protection + 38 cross-border). ~70–75 rows (~18%) have non-authoritative/malformed/empty URLs; a further ~24 are mismatched (e.g. **MY Copyright rows linking to Patent PDFs**, FTA title/URL cross-links). Need canonical re-sourcing before any citation use.
- **Government portals CSV** (`Sample governemnt portals_Pillar 6_7.csv`): **93 rows**, same schema (col 9 = `pillar.name`), P6/P7 only (55 domestic + 38 cross-border; MY 39, SG 34, AU 20). ~26–28% non-authoritative/mirror/mismatched, **concentrated almost entirely on Malaysia**.
- **Authoritative anchors:** `sso.agc.gov.sg` (SG, clean), `legislation.gov.au` + `oaic.gov.au` (AU, clean). **Malaysia is fragmented** across `pdp.gov.my`, `mcmc.gov.my`, `myipo.gov.my` + many third-party mirrors (`mohre.um.edu.my`, `f.datasrvr.com`, `cyrilla.org`, `commonlii.org`, law-firm blogs) → bulk of URL-repair effort.

---

## 3. Sample Legislation Corpus (8 PDFs — extraction edge cases)

Path: `Database/Sample legislations/`. 5 born-digital, 3 image-only scans. Folder names encode the edge case being tested.

| # | File | Pages | Origin | Lang | Challenge |
|---|------|-------|--------|------|-----------|
| 1 | Niue-Legislation Volume 1 | 683 | digital | EN (+Niuean) | **multi-Act consolidation in one PDF** (~33 Acts, A–C, Vol 1 of set) → act-boundary segmentation; "1 Title" recurs ~33× |
| 2 | Lao PDR e-Transactions No.31 | 29 | **scan** | Lao only | **0 text chars, 1 img/page; path lacks `scanned`** → mis-routes to born-digital path → **silently extracts nothing (BUG)** |
| 3 | C2026C00098VOL01 (AU Criminal Code) | 666 | digital | EN | deep dotted-decimal hierarchy, endnotes vs operative text, **3-volume split**, per-page "Authorised Version" boilerplate |
| 4 | PERSONAL DATA PROTECTION ACT 2010 (MY PDPA Act 709) | 52 | digital | EN | **cleanest / P6 keystone**; **s.129** = cross-border transfer (adequacy whitelist + derogations) |
| 5 | Telecommunications Act 1999 (SG) | 130 | digital | EN | SSO boilerplate per page; alphanumeric Part inserts (`5A/5B/5C`); Part 7 = International Obligations |
| 6 | India Public Procurement Order 2017 | 7 | **scan** | Hindi+EN | OCR (ilovepdf re-wrap artifacts); subordinate order |
| 7 | Pakistan PECA | 4 | **scan** | EN(±Urdu) | hardest OCR (low DPI) → flag everything |
| 8 | INDIA-~1.PDF (FEMA Gazette 2024) | 4 | digital | Hindi+EN parallel | **Devanagari extracts as mojibake** (Mangal font) — validate script integrity, not just char count; bilingual de-interleave + FDI cap table |

- **Routing rule (keystone):** decide scan-vs-digital from the **file content** (`has_embedded_fonts AND nonspace_chars_per_page > threshold`), **never from the folder path**. Also validate **script integrity** (FEMA Hindi passes a length check but is garbage).
- **Recommended MVP ingestion order:** MY PDPA → SG Telecoms Act → AU Criminal Code → Niue → India FEMA gazette → OCR track (Lao, India procurement, Pakistan) as flagged-only Phase 1b.

---

## 4. Submission Template & Output Contract

### 4.1 The 13-column provision-level output (`OUTPUT_TEMPLATE_31MAY.xlsx`, exact order)
Provision-level rows (one per extracted clause), distinct from the indicator-evidence DB schema.

`Economy`(req) · `Law Name`(req) · `Law Number / Ref`(opt*) · `Last Amended`(req) · `Indicator ID`(req) · `Article / Section`(req) · `Discovery Tag`(req, dropdown NEW/KNOWN) · `Location Reference`(**optional** — confirmed June 15) · `Verbatim Snippet`(req, no paraphrase) · `Mapping Rationale`(opt, ≤300 chars, **not directly scored**) · `Source URL`(req, official portal only) · `Confidence`(opt, 0.00–1.00) · `Notes`(opt, flexible, no char limit).

- **CSV/JSON on-disk schema is snake_case** (`economy, law_name, law_number_ref, last_amended, indicator_id, article, discovery_tag, location_reference, verbatim_snippet, mapping_rationale, source_url, confidence, notes`). JSON adds `ocr_quality_cer, processing_time_seconds, raw_context_before/after, pdf_is_scanned, retrieval_method`. "Judges validate programmatically — do not rename/reorder."
- **NEW vs KNOWN** = **NEW worth 20 of the 40 substantive points** (single largest differentiator). A newly-found *provision* counts NEW even if the parent law was already in the kit; same law+article+context already cited = KNOWN.
- *opt\* conflict:* XLSX marks `Law Number/Ref` OPTIONAL but README lists it Required → safest to fill it.

### 4.2 ⚠️ Indicator-ID conflict to resolve (template vs official)
- **June 15 ruling:** `P6-I1 = Pillar 6, Indicator 1 = 6.1`; following **official RDTII decimal codes is preferred**, but `P6-I#` is accepted **if mapped to the correct indicator description**.
- **BUT** the template's `Indicator Reference` sheet defines `P6-I1…P6-I5` with meanings (general prohibition / adequacy / contractual safeguards / consent / other exceptions) and `P7-I1…P7-I5` (legal basis / purpose limitation / data-subject rights / breach notification / enforcement) that **do NOT match** the official 6.1–6.4 / 7.1–7.5 definitions. **Use official RDTII `6.x`/`7.x` codes + descriptions as the source of truth**; do not adopt the template sheet's divergent taxonomy.

### 4.3 Non-English & absence output rules (June 15)
- **Verbatim = exact quotation.** If no official English version: quote the original language and **append an extra English-translation column after the 13** (record both original + English).
- **Never leave an indicator blank** when no law is found → write **"no evidence" / "no provision found"** (documented absence > empty cell).
- **Only record active/in-force law.** A repealed law may be kept **only if clearly flagged repealed** (in Notes); recording a repealed law as if active = wrong evidence → penalized. No status column exists → use Notes for broken-URL/repealed/OCR flags.
- **Unit of observation:** keep output clean — **one row per provision**; one provision → multiple indicators = multiple rows; cross-reference act+sub-regulation hierarchy via adjacent rows + Notes.
- **Final submission = ONE consolidated file** (per-pillar files only useful during dev); design for the end user (policymaker/reviewer).

### 4.4 Pitch deck (12 slides) + README
- Deck slides: Title/Team · Executive Summary (tie to P6&7) · Problem · Objective · System Architecture · Tech Solution/Innovation · Backend Logic (≤2 slides: chunking, hybrid retrieval, mapping logic, **how citation matches source 100% / anti-hallucination**) · Evaluation Metrics · Demo · Innovation/Advantage · Scalability · References.
- README mandatory now = **Quick Start (<10-min run)**; mandatory at finale = full set (Architecture, **Swapping the LLM** / **Swapping the OCR** = "No Vendor Lock-in", Supported Economies, Output Format, **Actual Cost Per Document — measured, not estimated**, Known Limitations, Test Suite, Reproducing Sample Kit, License = **Apache-2.0**, Key Dates).

---

## 5. Workshop Knowledge

### 5.1 June 1 — Orientation
Three automation **zones**: Zone 1 (discovery/crawl/OCR) + Zone 2 (extract/map/cite) = **mandatory**; **Zone 3 (scoring) = optional bonus**. Mandatory P6+P7 for SG/AU/MY (Malaysia is harder — includes error-correction of a seeded dataset). Sub-point rubric: Live Portal Crawling 10 (zero if only pre-downloaded PDFs) · OCR <5% error 10 · Modular backend (config-swap LLM) 15 · Audit trail (verbatim per row) 15. **HTML is harder than PDF** (differentiator). Manual baseline: 10+ researchers, 1–4 weeks/country, 2,600+ regulations.

### 5.2 June 4 — Knowledge workshop
- **Henry Gao:** 7-part statute skeleton (preamble → definitions → substantive obligations → institutional → enforcement/penalty → judicial review → schedules) + implementing regs/cross-refs; modal verbs (`must/shall` = binding, `must not … unless` = conditional prohibition, `may` = discretion); **scope = domestic regulations only** (FTA/treaty conflict out of scope). Singapore PDPA 2012 mapped: **s.26 → 6.4=1**, s.25 retention = ceiling (7.3=0), s.11(3) DPO → 7.4=1, Part 6A breach notification, CPC ss.39–40 → 7.5.
- **WTO / end-user (Solomonyan):** want **clear, reliable, actionable** structured output, not law compilations; one-sentence research aim; map `shall` vs `should/may`; top tool failures = broken/wrong-page URLs, **right paragraph under wrong heading**, hallucination, missed cross-law exclusions. "Accuracy and traceability of all cited sources."
- **DTI / EUI (González & Rogaler):** **secondary-then-primary** workflow; output = **structured findings** ("a documented record linking a regulatory measure to its legal source, provision, text, translation, timeline, methodological relevance"), field-separated (CSV/Excel/JSON/API), **citation to exact provision + URL + access date + archived copy**, distinguish primary vs secondary, **multilingual by design** (original + translation), flag uncertainty/contradiction/amendment. "Acceleration, not replacement."
- **TH2OECD (Thipsena):** build a **workflow/state-gate app, NOT a chatbot** (chatbots don't scale to 270×65); **GraphRAG** (Neo4j/Cypher, multi-layer legal hierarchy) for amendments-in-notifications; **heuristic LLM vs deterministic Python boundary** (LLM for ambiguity/reading/comparison; **code for validation, required fields, scoring, version control, output packaging**); 5 principles (iteration; observability early; domain-grounded reasoning; right AI/code boundary; balance generic vs specific); **"fighting the No, not the Yes"** = absence search needs whole-universe coverage.

### 5.3 June 5 — Hands-on + assignments (deepest extraction methodology)
- **Two zones** + exact DB columns **D Raw score · E Act/practice · F Coverage (horizontal/sectoral) · G Impact (exact provision + interpretation) · H Timeframe · I–J References · M Note** ("never record only the law title").
- **Primary = record + score; secondary = lead only (Note column)** — except **3.4 / 5.3 / 9.1** where de-facto/secondary evidence IS recorded and scored.
- **Hierarchy ≠ coverage:** horizontal ≠ higher rank; a sectoral Act and a horizontal Act can share the same hierarchy and score. **Controlling evidence vs recorded evidence:** record sectoral notices even when a horizontal law controls the score.
- Worked classifications: 6.4 (Armenia consent+adequacy), 6.2 (Kazakhstan store-in-country), 6.3 (China local servers), 7.5 (SG CPC §39), 7.2 (Bhutan cyber code), 7.4 (India DPDP §10). **Higher RDTII score = more regulatory complexity, not a value judgment.**
- **Answer-key gold (Assignment 1):** MAS Cyber Hygiene 2019 output wrong for **two** reasons — dead URL **and** superseded (cancelled 1 Jul 2022) → use **Notice FSM-N16**; Cybersecurity Act 2018 is controlling for 7.2 but the sectoral notice is still recorded. Other graded errors: hallucinated section numbers (IT Act §70B sub-sections that don't exist), country-context misses (SG gambling already illegal → not 9.1), CPTPP isn't domestic law, Malaysia Financial Procedure Act 1957 should score 1 (international tender = last resort).
- **Practice-dataset methodology sheet uses official decimal IDs** (`5.3`, `6.1`, `6.4`, `7.3`) with verbatim score criteria → confirms official numbering is the source of truth.

### 5.4 June 11 — Responsible AI (guest best-practice/coaching, NOT formal ESCAP rules)
Trust → adoption → impact; "should we build it?"; **accountability stays with humans/orgs — "AI should never remove accountability"**; **explainability = regulatory necessity** in regulated industries (link output↔input, "trace back"); continuous monitoring; compliance designed-in, not bolt-on; bias scales catastrophically without HITL (Tay, Amazon-hiring cases); 5 cross-border "borders" (language/culture/regulation/infrastructure/organization); judges look for Impact/Feasibility/Scalability/Innovation/Adoption, problem-first. **Does NOT mandate span/verbatim/NLI gates — those are RIE extensions.**

### 5.5 June 12 — AI toolchain & architecture
- **RAG pipeline:** parse → chunk → embed (**same model for index & query**) → similarity top-k → augment → grounded generation ("use ONLY provided context").
- **BGE-M3** = headline embedder: dense+sparse+multi-vector (native **hybrid**), 100+ languages, ≤**8192 tokens**, **1024-dim**, ~2 GB, fine-tunable, CPU-runnable. Cosine distance.
- **Re-ranking is mandatory**; hybrid (semantic + deterministic/keyword + cache) beats pure top-k; evidence filtered out early can't be recovered.
- **GraphRAG (Neo4j + Cypher / Microsoft GraphRAG)** for cross-referenced/amended legal text; plain RAG struggles with amendments-in-attachments.
- **Deterministic-vs-heuristic keystone:** Claude Code is **~98.4% deterministic infra / 1.6% AI decision logic** → wrap the small LLM core in a large deterministic shell. **"Do not collapse evidence into scores too early; keep fragments, interpretations, scoring separate; model uncertainty."**
- Provenance tiering for citations: `[settled]` / `[verify]` / `[verify-pinpoint]` (pinpoint cites = highest fabrication risk). LLM sizing: >10B for good answers, >30B to fine-tune; "good-enough" model filters noise. *(Note: the workshop gave **no** numeric chunk-size/overlap; any specific chunk numbers are repo/engineering defaults, not an ESCAP spec.)*

### 5.6 June 15 — Q&A (rule clarifications — see §0.1 and §4.3; full timeline §6)
Research process; native-language preferred as source-of-truth (English lags); blocked portals → ministry sites → secondary (Note column); P6-I1=6.1; 7.5 mandatory / 6.5 out of scope; CPU-only/no-GPU/no-API; CLI not required (GUI/web/API/notebook all fine, polished UI only required at finale); RAG ingestion allowed but corpus must be **updatable** (laws change); any framework/model allowed if **modular/swappable**; evaluation tests against the **FULL** known evidence **and beyond** ("find MORE, not less"); document-by-document HITL verification.

### 5.7 Transcripts available
All sessions (June 1/4/5/11/12/15) have **`.txt`/`.vtt`/`.srt` transcripts that were ingested**. Only the raw `.mp4`/`.mov` video/audio files are not separately processed.

---

## 6. Timeline (exact)
- **20 Jul 2026 (Bangkok time)** — Round 1 submission deadline; 4 deliverables via one online form (link emailed ~28 Jun to team head).
- **31 Jul** — 20 teams shortlisted. **3 Aug** — live pitch/demo (interview, real working prototype). **5 Aug** — 5 finalists announced. **Oct 2026** — Grand Finale, Bangkok (in-person pitch). No resubmission after deadline.

---

## 7. Repo Architecture (RIE — hexagonal)

- Packages under `packages/rie-*`. **Frozen:** `rie-contracts`. **Lead-owned (never edit):** `rie-domain`, `rie-orchestration`.
- Pipeline: **ingest → extract → retrieval → classify → verify (4 gates) → coverage (3-state) → output**.
- CLI: `python main.py --economy Malaysia --pillar 6,7`.
- **Pillar logic in YAML only:** `pillars/`, `sources/jurisdictions/`, `gold/pillar_06|07/`. Hard-coded pillar/indicator/jurisdiction id in `.py` = bug CI catches.

### 7.1 Two-layer output (anti-hallucination keystone)
- **LLM never authors a fact** — no citation, no authority tier, no score. It selects from enums (constrained decoding) + emits **span IDs** (`doc12#4120-4215`). Deterministic code does ID-replacement to materialize citations from stored metadata.
- Layer 1 (extraction): verifiable, never emitted without all 4 gates. Layer 2 (score): recommendation, `human_confirmation_required = True`. (Directly endorsed by June 4 TH2OECD heuristic/deterministic boundary + June 12 "don't collapse evidence early.")

### 7.2 The 4 verification gates
1. **Span-existence** — char offsets resolve to real text (deterministic).
2. **Verbatim-match** — exported snippet byte-identical to re-extracted span (deterministic).
3. **Entailment** — NLI + 2nd LLM; any disagreement → `flagged`.
4. **Self-consistency** — N=3 on fixed input; unstable label → `flagged`.
(KG/entity-grounding = Phase 2, not MVP.)

### 7.3 Absence reasoning (3-state, never bare 0)
`evidence_found` · `no_evidence_in_searched_corpus` (carries measured gold-set recall) · `insufficient_coverage`. RIE extension beyond ESCAP PDFs (which note absence + score 0 with reasoning). Aligns with June 4 "fighting the No."

### 7.4 Gold YAML pattern (derived from Round 1 sheets)
`gold_id` (pillar_jur_indicator_doc), `pillar_id`, `indicator_id`, `jurisdiction`, `doc_id` (== source registry id), `span_text`, `expected_clause_pattern` (obligation|prohibition|conditional_regime), `expected_score_band`, `expected_authority_tier` (tier_1_statute|tier_3_guideline|…), `notes` (law_name, source_url, coverage, rationale). One indicator → many gold items.

---

## 8. Gaps to close before submission

| Gap | Action |
|-----|--------|
| **CPU-only / no-GPU / no-API-key** target | ensure default path is local BGE-M3 + open-weight LLM on CPU; gate any commercial API behind config + fallback |
| No `batch_run.py` / no `--pdf` on `main.py` | add batch + single-PDF entrypoints |
| README says `pip` but repo uses **uv** | fix Quick Start to `uv sync` / `uv run` (still <10-min) |
| Most sources lack `local_path` | cache demo-economy corpus locally for offline <10-min demo |
| Malaysia run produced **0 provisions** | debug retrieval/extract on MY PDPA s.129 (known-good keystone) |
| **Lao PDF mis-routes** (image-only, path lacks `scanned`) | content-based scan detection + script-integrity check, not path hints |
| DB ingester | normalize `'1.00\n'`, `\xa0`, blank rows; treat H→pre-Note as one multi-URL field |
| Template `P6-I#` vs official `6.x` (different *meanings*) | emit official `6.x`/`7.x`; map `P6-I#` only as accepted alias at display layer |
| `location_reference` optional; verbatim non-EN → append EN column | align output schema with June 15 rulings |
| Single consolidated final file + "no evidence" rows + repealed-flagging | implement output-assembly rules |
| P8–12 gold draft; Pillar 12 YAML missing 12.5–12.9 | fill once P6/P7 solid (out of Round-1 scope) |

---

## 9. Hard rules (repo `CLAUDE.md`)

- Python 3.12 only; **uv only** (`uv add/sync/run`), never pip, never hand-edit `uv.lock`.
- Stay in your package. `rie-contracts` frozen. Dependency direction CI-enforced.
- pydantic ≥2, Protocol ports, pyright strict, ruff clean. No `Any` in public sigs.
- No pillar logic in `.py`. No LLM-authored citations/scores. No silent `0`. One package per PR.

---

## 10. Provenance

Refreshed via a 12-agent parallel deep re-read (Opus 4.8) of all 66 files in `Hackthon Knowledge Products/`:
orientation/knowledge/hands-on/responsible-AI/toolchain/Q&A workshops (June 1/4/5/11/12/15) + transcripts;
Takehome assignments 1 & 2 + answer key + practice dataset; Round 1 & 2 Excel databases + Methodology;
3 RDTII framework PDFs (Guide / Internal Guide / Non-regulatory list); 8 sample legislation PDFs;
portals & legal-inventory CSVs; OUTPUT_TEMPLATE + pitch deck + README — cross-referenced against repo
`pillars/`, `sources/`, `gold/`, `packages/rie-*`. The 3-state absence / 4-gate / two-layer designs are
RIE-specific reinforcements of the ESCAP methodology.
