"""``python -m rie_eval`` — Typer CLI for gold-set inspection and metrics.

Subcommands:

- ``rie-eval gold list``       — list gold items per pillar.
- ``rie-eval gold validate``   — re-validate every gold set against the schema.
- ``rie-eval metrics``         — load claims from the database and compute metrics.
- ``rie-eval recall``          — measure retrieval recall for a pillar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rie_config import ConfigError, ConfigRepository
from rie_contracts import GoldItem

from rie_eval.gold_harness import score_against_gold
from rie_eval.hallucinated_words import compute_hallucinated_words_rate
from rie_eval.service import Evaluator

app = typer.Typer(
    name="rie-eval",
    add_completion=False,
    help="Gold-set runner + RIE-specific evaluation metrics.",
    no_args_is_help=True,
)
gold_app = typer.Typer(
    name="gold",
    add_completion=False,
    help="Gold-set inspection commands.",
    no_args_is_help=True,
)
app.add_typer(gold_app, name="gold")


def _repo_root_option() -> Path:
    """Default to the repo root (4 parents up from this file)."""
    return Path(__file__).resolve().parents[4]


def _all_pillar_dirs(repo_root: Path) -> list[Path]:
    gold_dir = repo_root / "gold"
    if not gold_dir.exists():
        return []
    return sorted(p for p in gold_dir.glob("pillar_*") if p.is_dir())


def _pillar_id_from_dirname(dirname: str) -> str:
    # "pillar_06" → "6", "pillar_12" → "12"
    return str(int(dirname.split("_", 1)[1]))


# ─── gold list ───────────────────────────────────────────────────────────────


@gold_app.command("list")
def gold_list(
    pillar: Annotated[
        str | None,
        typer.Option("--pillar", "-p", help="Pillar id (e.g. '6'). Omit for all."),
    ] = None,
    repo_root: Annotated[
        Path,
        typer.Option("--repo-root", help="Override repo root."),
    ] = _repo_root_option(),
) -> None:
    """List gold items, grouped by pillar."""
    config = ConfigRepository(repo_root=repo_root)
    pillar_ids: list[str]
    if pillar:
        pillar_ids = [pillar]
    else:
        pillar_ids = [_pillar_id_from_dirname(p.name) for p in _all_pillar_dirs(repo_root)]

    total = 0
    for pid in pillar_ids:
        items = list(config.load_gold(pid))
        total += len(items)
        typer.echo(f"pillar {pid}: {len(items)} item(s)")
        for item in items:
            typer.echo(
                f"  {item.gold_id}  ind={item.indicator_id}  "
                f"juris={item.jurisdiction}  band={item.expected_score_band}  "
                f"tier={item.expected_authority_tier}"
            )
    typer.echo(f"total: {total}")


# ─── gold validate ───────────────────────────────────────────────────────────


@gold_app.command("validate")
def gold_validate(
    repo_root: Annotated[
        Path,
        typer.Option("--repo-root", help="Override repo root."),
    ] = _repo_root_option(),
) -> None:
    """Re-validate every gold set against the JSON schema."""
    config = ConfigRepository(repo_root=repo_root)
    pillar_dirs = _all_pillar_dirs(repo_root)
    if not pillar_dirs:
        typer.echo("no gold pillars found", err=True)
        raise typer.Exit(code=1)

    failures: list[str] = []
    total = 0
    for pdir in pillar_dirs:
        pid = _pillar_id_from_dirname(pdir.name)
        try:
            items: list[GoldItem] = list(config.load_gold(pid))
            total += len(items)
            typer.echo(f"OK   pillar {pid}: {len(items)} item(s)")
        except ConfigError as exc:
            failures.append(f"pillar {pid}: {exc}")
            typer.echo(f"FAIL pillar {pid}: {exc}", err=True)

    typer.echo(f"validated {total} item(s) across {len(pillar_dirs)} pillar(s)")
    if failures:
        raise typer.Exit(code=2)


# ─── metrics ─────────────────────────────────────────────────────────────────


@app.command("metrics")
def metrics(
    pillar: Annotated[str, typer.Option("--pillar", "-p", help="Pillar id.")],
    repo_root: Annotated[
        Path,
        typer.Option("--repo-root", help="Override repo root."),
    ] = _repo_root_option(),
    no_db: Annotated[
        bool,
        typer.Option("--no-db", help="Skip DB load; print zero-claim baseline."),
    ] = False,
) -> None:
    """Load claims from the database, compute metrics, print a flat table."""
    config = ConfigRepository(repo_root=repo_root)

    claims = []
    repo = None
    if not no_db:
        try:
            from rie_persistence import DocumentRepository, create_engine
            from rie_persistence.db import make_session_factory

            engine = create_engine()
            factory = make_session_factory(engine)
            repo = DocumentRepository(session_factory=factory)
            # Pull all flagged + unflagged claims for the pillar.
            claims = list(repo.list_claims_for_review(pillar_id=pillar))
        except Exception as exc:
            typer.echo(f"DB unavailable ({exc}); proceeding with empty claim set", err=True)
            claims = []
            repo = None

    evaluator = Evaluator(config=config, repo=repo)
    results = evaluator.evaluate_pillar(pillar, claims)
    # Pretty-print a stable two-column table.
    width = max((len(k) for k in results), default=0)
    for key in sorted(results):
        value = results[key]
        typer.echo(f"{key:<{width}}  {value:.4f}")


# ─── recall ──────────────────────────────────────────────────────────────────


@app.command("recall")
def recall(
    pillar: Annotated[str, typer.Option("--pillar", "-p", help="Pillar id.")],
    retrieved: Annotated[
        Path,
        typer.Option(
            "--retrieved",
            help="JSON file: a list-of-lists of retrieved ids, one per gold item.",
        ),
    ],
    repo_root: Annotated[
        Path,
        typer.Option("--repo-root", help="Override repo root."),
    ] = _repo_root_option(),
) -> None:
    """Compute retrieval recall over the pillar's gold set."""
    if not retrieved.exists():
        typer.echo(f"missing file: {retrieved}", err=True)
        raise typer.Exit(code=1)
    payload = json.loads(retrieved.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(x, list) for x in payload):
        typer.echo("--retrieved must point at a JSON list-of-lists of strings", err=True)
        raise typer.Exit(code=1)
    config = ConfigRepository(repo_root=repo_root)
    evaluator = Evaluator(config=config, repo=None)
    score = evaluator.measure_retrieval_recall(pillar, payload)
    typer.echo(f"retrieval_recall[pillar {pillar}] = {score:.4f}")


# ─── gold-score ────────────────────────────────────────────────────────────


@app.command("gold-score")
def gold_score(
    pillar: Annotated[str, typer.Option("--pillar", "-p", help="Pillar id (e.g. '6').")],
    predicted: Annotated[
        Path,
        typer.Option(
            "--predicted",
            help="JSON file: a list of predicted-provision records (see gold_harness docs).",
        ),
    ],
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Override repo root."),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the full score dict as JSON instead of the table."),
    ] = False,
) -> None:
    """Score predicted provisions against the gold set; print an F1 table.

    Substantive-Accuracy (40%) benchmark. ``--predicted`` points at a JSON
    array of records (keys: jurisdiction, indicator_id, doc_id, span_text,
    score_band, authority_tier, discovery_tag). Gold is loaded via rie_config
    for ``--pillar``. Scoring is per (indicator, law) provision pair.
    """
    repo_root = repo_root if repo_root is not None else _repo_root_option()
    if not predicted.exists():
        typer.echo(f"missing file: {predicted}", err=True)
        raise typer.Exit(code=1)
    payload = json.loads(predicted.read_text(encoding="utf-8"))
    # Accept either a bare list, or {"items": [...]} / {"provisions": [...]}.
    if isinstance(payload, dict):
        records = payload.get("items") or payload.get("provisions") or []
    else:
        records = payload
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        typer.echo("--predicted must be a JSON list of record objects", err=True)
        raise typer.Exit(code=1)

    config = ConfigRepository(repo_root=repo_root)
    gold = list(config.load_gold(pillar))
    result = score_against_gold(records, gold)

    if as_json:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
        return

    counts = result["counts"]
    assert isinstance(counts, dict)
    typer.echo(f"== gold-score: pillar {pillar} ==")
    typer.echo(
        f"gold={counts['gold_total']}  predicted={counts['predicted_total']}  "
        f"tp={counts['tp']}  fn={counts['fn']}  fp={counts['fp']}"
    )
    typer.echo(
        f"precision={result['precision']:.4f}  recall={result['recall']:.4f}  "
        f"f1={result['f1']:.4f}  (macro, per-provision)"
    )
    typer.echo(
        f"field_accuracy={result['field_accuracy']:.4f}  "
        f"citation_fidelity={result['citation_fidelity']:.4f}  "
        f"false_zero_rate={result['false_zero_rate']:.4f}  "
        f"discovery_new={result['discovery_new_count']}"
    )
    typer.echo("")
    per_indicator = result["per_indicator"]
    assert isinstance(per_indicator, dict)
    typer.echo(f"{'indicator':<10}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}{'field':>8}{'cite':>8}")
    typer.echo("-" * 59)
    for ind in sorted(per_indicator):
        row = per_indicator[ind]
        typer.echo(
            f"{ind:<10}{row['precision']:>8.3f}{row['recall']:>8.3f}{row['f1']:>8.3f}"
            f"{int(row['support']):>9}{row['field_accuracy']:>8.3f}{row['citation_fidelity']:>8.3f}"
        )


# ─── hallucinated ────────────────────────────────────────────────────────────


@app.command("hallucinated")
def hallucinated(
    doc: Annotated[str, typer.Option("--doc", help="doc_id to evaluate.")],
    repo_root: Annotated[
        Path,
        typer.Option("--repo-root", help="Override repo root."),
    ] = _repo_root_option(),
    raw_dir: Annotated[
        Path | None,
        typer.Option(
            "--raw-dir",
            help="Override raw-cache dir (default: <repo_root>/data/cache/raw).",
        ),
    ] = None,
) -> None:
    """Compute the hallucinated-words rate for one document.

    Loads the document's element texts from the persistence layer and the
    raw bytes from ``data/cache/raw/<sha256>.bin``. If either is missing the
    command degrades gracefully (skip + non-zero exit only on a hard error).
    """
    try:
        from rie_persistence import DocumentRepository, create_engine
        from rie_persistence.db import make_session_factory
    except Exception as exc:  # pragma: no cover - import failure is environmental
        typer.echo(f"persistence unavailable ({exc})", err=True)
        raise typer.Exit(code=1) from exc

    try:
        engine = create_engine()
        factory = make_session_factory(engine)
        repo = DocumentRepository(session_factory=factory)
    except Exception as exc:
        typer.echo(f"DB unavailable ({exc})", err=True)
        raise typer.Exit(code=1) from exc

    try:
        meta = repo.get_document(doc)
    except Exception as exc:
        typer.echo(f"doc {doc!r} not found ({exc})", err=True)
        raise typer.Exit(code=1) from exc

    raw_root = raw_dir if raw_dir is not None else (repo_root / "data" / "cache" / "raw")
    raw_path = raw_root / f"{meta.sha256}.bin"
    if not raw_path.exists():
        typer.echo(
            f"raw bytes missing for {doc} (expected {raw_path}); "
            "skipping hallucinated-words computation",
            err=True,
        )
        raise typer.Exit(code=0)

    try:
        source_text = raw_path.read_bytes().decode("utf-8", errors="ignore")
    except Exception as exc:
        typer.echo(f"failed to read {raw_path} ({exc})", err=True)
        raise typer.Exit(code=1) from exc

    # Pull all element texts for the doc. The repo doesn't expose a
    # by-document iterator, so use structure edges → element_ids (its
    # closest available accessor) and fall back gracefully.
    try:
        edges = repo.get_structure_edges(doc)
    except Exception:
        edges = []
    element_ids: list[str] = []
    seen: set[str] = set()
    for e in edges:
        for eid in (e.from_element, e.to_element):
            if eid not in seen:
                seen.add(eid)
                element_ids.append(eid)
    extracted_chunks: list[str] = []
    for eid in element_ids:
        try:
            extracted_chunks.append(repo.get_element_text(eid))
        except Exception:
            continue
    extracted_text = "\n".join(extracted_chunks)
    if not extracted_text:
        typer.echo(f"no element text recovered for {doc}", err=True)
        raise typer.Exit(code=0)

    rate = compute_hallucinated_words_rate(extracted_text, source_text)
    typer.echo(f"hallucinated_words_rate[{doc}] = {rate:.4f}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
