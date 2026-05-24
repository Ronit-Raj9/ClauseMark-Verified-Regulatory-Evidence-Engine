I'll research every tool in the v3 stack so you have current, accurate build information. Let me search.Good. Now the retrieval stack — Qdrant with Python, and BGE-M3.Now LangGraph, the verification/NLI tooling, and constrained decoding.One more — constrained decoding (Outlines/XGrammar), the NLI verifier, and reranker libraries.I have enough to give you a complete, accurate build reference. Here it is — every tool in the v3 stack, current as of 2026, with Python + Qdrant as your base.

## Document extraction

**Docling** (IBM, MIT licence). `pip install docling`. Install as a Python library, run the CLI directly, or code conversion into a Python app with `DocumentConverter`. It converts any PDF to JSON or Markdown, understands page layout, reading order and recovers table structure, extracts metadata like title and language, and optionally applies OCR for scanned PDFs. Two things to know: it works on macOS and Linux but Windows is not officially tested — use Python 3.10/3.11/3.12 (run on Linux, or WSL if you're on Windows). And for your span-citation requirement, the companion package matters: `docling-parse` extracts text with char-, word- and line-level coordinates from programmatic PDFs — this is the package Docling uses internally. Those coordinates are what power your audit-viewer highlight and your offset-based verification.

**PyMuPDF** for the fast born-digital path — `pip install pymupdf`, mature, stable.

For scanned legal PDFs, Docling needs an OCR engine installed separately (Tesseract from the official GitHub releases); for degraded scans route to a VLM-OCR as v3 specifies.

## Qdrant — your vector store, with native hybrid search

This is the key one since you've committed to it. `pip install qdrant-client`. Run the server via Docker (`qdrant/qdrant` image). Qdrant supports hybrid search natively — and importantly, **it does the fusion for you**. You prefetch dense and sparse results, then merge them with Reciprocal Rank Fusion (RRF) — running the two sub-queries in parallel and fusing the ranked lists into a single result. That's your RRF k=60 step, built in, no custom code.

The structure: create a collection with both a dense vector config and a sparse vector config; use `models.FusionQuery(fusion=models.Fusion.RRF)` to combine them. Qdrant also offers DBSF (distribution-based score fusion) as an alternative — Qdrant's own analysis shows a naive linear combination of BM25 and dense scores fails because relevant and non-relevant objects aren't linearly separable, which is why RRF or DBSF is needed.

One **specific gotcha** for your stack: BGE-M3 can generate sparse vectors natively, but some hybrid-search wrappers always instantiate a separate BM25 sparse encoder anyway — which means hybrid retrieval can combine dense vectors from one model with sparse vectors from a different representation space. Decide deliberately: either use BGE-M3's *own* sparse output for both vectors (one consistent space), or use BGE-M3 dense + Qdrant's BM25 sparse — but know which you're doing. For legal text where exact terms (section numbers, defined terms) matter, BM25 sparse is genuinely valuable, so the BGE-M3-dense + BM25-sparse combination is the sensible default; just wire it explicitly rather than letting a wrapper guess.

## Embeddings

**BGE-M3** (BAAI, open). Two ways to run it: `FlagEmbedding` (`BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)` — gives you dense, sparse, and ColBERT vectors), or **FastEmbed**, which Qdrant maintains and integrates cleanly — `from fastembed import TextEmbedding, SparseTextEmbedding`, with `SparseTextEmbedding("Qdrant/bm25")` for the BM25 sparse side. FastEmbed is the path of least resistance with Qdrant.

## Reranking

After hybrid retrieval, rerank with a cross-encoder. Qdrant's documented pattern: hybrid search prefetches and RRF-merges, then results are reranked using late-interaction (ColBERT-style) embeddings for maximum precision. You have two viable choices — a `bge-reranker` cross-encoder (run via `sentence-transformers` or FastEmbed), or a ColBERT late-interaction model stored as multi-vectors in Qdrant. For a hackathon, the `bge-reranker` cross-encoder is simpler: retrieve top 20–40, rerank, keep top 6–10.

## Orchestration — LangGraph

`pip install langgraph`. LangGraph 1.2.0 released May 2026; use Python 3.11 or 3.12 — 3.9 was dropped in LangGraph 1.1, and avoid 3.13 until dependencies confirm support. The human-in-the-loop mechanism you need: calling `interrupt()` pauses graph execution, marks the thread as interrupted, and puts your payload into the persistence layer; persistence is first-class — every step reads and writes a checkpoint, so you can pause, let a human edit the checkpoint, and resume. Resume with `Command(resume=...)`.

**Two production caveats the docs gloss over** — relevant because your reviewer workflow depends on this:
- When a thread is interrupted, nobody gets notified — no email, no Slack, no ping; you build that layer yourself. And LangGraph's persistence records graph state but is not an audit log — it doesn't produce a structured record of who was asked, when, what they decided; you must instrument that yourself. Your §10 `reviews` table *is* that audit log — good, you already planned it.
- A node containing `interrupt()` re-executes from the start on resume — so put `interrupt()` in a dedicated node that does nothing else, or re-execution causes side effects.

For the checkpointer: `MemorySaver` for dev, but use the **Postgres checkpointer** in your real build since you're already running Postgres for pgvector/the data model.

## Constrained decoding — the anti-hallucination keystone

This is what makes the model physically unable to invent an indicator label. XGrammar is the default constrained-decoding backend for vLLM, SGLang, and TensorRT-LLM; Ollama also uses XGrammar under the hood — all give 100% JSON-schema compliance via constrained decoding. XGrammar works *during* token generation, masking invalid tokens so the model cannot produce malformed output; Outlines is the Python-native alternative — an FSM-based library with a Python API you call from application code, working with transformers, vLLM and llama.cpp.

Practical guidance for 2026: default to XGrammar; switch to Outlines if your schemas are very complex and reused across thousands of requests so the FSM compile cost amortises; use lm-format-enforcer only if XGrammar is unavailable. If you self-host on vLLM, you pass `guided_json` / `guided_choice` and XGrammar handles it — structured output is available in vLLM 0.8.5+, supporting choice lists, JSON schemas and regex. For your use case `guided_choice` is the one that matters most — it forces the `indicator_id` to be exactly one of your enum values.

There's even direct evidence this helps your goal: a 2025 study examined guided decoding's role in RAG specifically, comparing Outlines, XGrammar and LM Format Enforcer on success rates and hallucination rates — guided decoding is studied as a hallucination-reduction mechanism, not just a formatting tool.

## Serving, verification, evaluation

- **vLLM** (throughput, self-hosted reasoning model) or **Ollama** (quick local dev). Both use XGrammar for structured output.
- **NLI verifier:** an off-the-shelf NLI model via `transformers` (e.g. a DeBERTa-MNLI) for the entailment gate, plus a second LLM call. Keep the gate's question narrow per v3 §6.5.
- **RAGAS** for evaluation — `pip install ragas` — gives you context precision, context recall, faithfulness.
- **Langfuse** (open-source) for tracing — `pip install langfuse`.
- **Postgres + pgvector** for your data model (§10). Note you could store vectors in either pgvector *or* Qdrant — since you've chosen Qdrant for retrieval, use Qdrant for vectors and plain Postgres for the relational data model. Don't run both vector stores.
- **Streamlit** for the audit viewer — `pip install streamlit`.

## One honest flag on a reference, two stack notes

Reproducible 2026 build pattern, end to end: a documented "modern RAG pipeline 2026" stack is exactly Docling → Qdrant hybrid (BM25 sparse + dense) → AI agent via LangGraph — Docling for document understanding, Qdrant for native hybrid search, LangGraph for reasoning and tool use. That is your stack; it's a trodden path, which de-risks the build.

Two notes. First — **licences must be verified per model individually** at build time; the infrastructure tools above are cleanly open (Docling MIT, Qdrant Apache 2.0, LangGraph MIT, vLLM Apache 2.0), but the reasoning LLM and the VLM-OCR you pick each need their own licence checked, since open-weight model licences vary by release and size. Second — **pin every version in `requirements.txt`**; this ecosystem moves monthly (LangGraph 1.2, vLLM structured-output API changes), and an unpinned build will break between your prototype and finale rounds.

Want me to turn this into an actual annotated `requirements.txt` plus a `docker-compose.yml` (Qdrant + Postgres + vLLM/Ollama) so you have a running skeleton to build on?