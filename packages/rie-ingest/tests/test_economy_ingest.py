"""Tests for per-economy real ingest (`economy_ingest.py`).

Exercises:
  * a tmp source registry + tmp local file → `(DocumentMeta, bytes, ExtraMeta)`
    with sha256 + `law_number_ref` / `last_amended` from the `extra:` block,
  * missing-registry → clear `EconomyIngestError`,
  * the real Malaysia PDPA sample PDF when present (skip-if-missing guard),
  * `available_economies` discovery.

No network. No LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from rie_contracts import DocumentMeta
from rie_ingest import compute_sha256
from rie_ingest.economy_ingest import (
    EconomyIngestError,
    EconomyIngestor,
    available_economies,
    load_economy_documents,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MALAYSIA_PDPA_PDF = (
    REPO_ROOT
    / "details"
    / "Sample Kit"
    / "Sample legislations"
    / "General"
    / "PERSONAL DATA PROTECTION ACT 2010.pdf"
)


def _write_registry(repo_root: Path, jurisdiction: str, sources: list[dict[str, object]]) -> Path:
    jur_dir = repo_root / "sources" / "jurisdictions"
    jur_dir.mkdir(parents=True, exist_ok=True)
    path = jur_dir / f"{jurisdiction.lower()}.yaml"
    path.write_text(
        yaml.safe_dump(
            {"jurisdiction": jurisdiction, "language": "en", "sources": sources},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _seed_schemas(repo_root: Path) -> None:
    """Copy the source-registry + gold + pillar schemas a `ConfigRepository`
    needs so a tmp repo_root can validate the source registry."""
    for rel in (
        Path("sources") / "_schema" / "source_registry.schema.json",
        Path("gold") / "_schema" / "gold_item.schema.json",
        Path("pillars") / "_schema" / "pillar.schema.json",
        Path("pillars") / "_schema" / "indicator.schema.json",
    ):
        src = REPO_ROOT / rel
        if not src.exists():
            continue
        dst = repo_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


# ─── tmp registry + tmp file ─────────────────────────────────────────────────


def test_load_economy_documents_local_file_with_extra(tmp_path: Path) -> None:
    _seed_schemas(tmp_path)
    body = b"PERSONAL DATA PROTECTION ACT 2010 -- section 129 cross-border transfer."
    law_file = tmp_path / "data" / "samples" / "my_pdpa.txt"
    law_file.parent.mkdir(parents=True, exist_ok=True)
    law_file.write_bytes(body)

    _write_registry(
        tmp_path,
        "Malaysia",
        [
            {
                "source_id": "my_pdpa_act709_2010",
                "title": "Personal Data Protection Act 2010",
                "source_url": None,
                "local_path": "data/samples/my_pdpa.txt",
                "document_type": "statute",
                "authority_tier": "tier_1_statute",
                "effective_date": "2010-06-10",
                "language": "en",
                # The frozen contract cannot hold these; they live in `extra:`.
                "extra": {
                    "law_number_ref": "Act 709",
                    "last_amended": "2024",
                },
            }
        ],
    )

    docs = load_economy_documents(tmp_path, "Malaysia")
    assert len(docs) == 1
    meta, raw, extra = docs[0]

    assert isinstance(meta, DocumentMeta)
    assert meta.doc_id == "my_pdpa_act709_2010"
    assert meta.jurisdiction == "Malaysia"
    assert raw == body
    assert meta.sha256 == compute_sha256(body)

    # Extra dict carries the CSV-facing fields.
    assert extra["source_id"] == "my_pdpa_act709_2010"
    assert extra["law_number_ref"] == "Act 709"
    assert extra["last_amended"] == "2024"
    assert extra["local_path"] == "data/samples/my_pdpa.txt"


def test_extra_fields_default_none_without_extra_block(tmp_path: Path) -> None:
    _seed_schemas(tmp_path)
    law_file = tmp_path / "data" / "samples" / "plain.txt"
    law_file.parent.mkdir(parents=True, exist_ok=True)
    law_file.write_bytes(b"a plain statute body with enough bytes to be real")

    _write_registry(
        tmp_path,
        "Singapore",
        [
            {
                "source_id": "sg_pdpa_2012",
                "title": "Personal Data Protection Act 2012",
                "local_path": "data/samples/plain.txt",
                "document_type": "statute",
                "authority_tier": "tier_1_statute",
            }
        ],
    )

    docs = load_economy_documents(tmp_path, "Singapore")
    _, _, extra = docs[0]
    assert extra["law_number_ref"] is None
    assert extra["last_amended"] is None


# ─── missing registry → clear error ──────────────────────────────────────────


def test_missing_registry_raises_clear_error(tmp_path: Path) -> None:
    _seed_schemas(tmp_path)
    with pytest.raises(EconomyIngestError) as exc:
        load_economy_documents(tmp_path, "Atlantis")
    msg = str(exc.value)
    assert "Atlantis" in msg
    assert "atlantis.yaml" in msg


# ─── available_economies ─────────────────────────────────────────────────────


def test_available_economies_lists_registries(tmp_path: Path) -> None:
    _seed_schemas(tmp_path)
    (tmp_path / "data" / "samples").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "samples" / "x.txt").write_bytes(b"some statute bytes here")
    for jur in ("Singapore", "Malaysia"):
        _write_registry(
            tmp_path,
            jur,
            [
                {
                    "source_id": f"{jur.lower()}_doc",
                    "title": f"{jur} law",
                    "local_path": "data/samples/x.txt",
                    "document_type": "statute",
                    "authority_tier": "tier_1_statute",
                }
            ],
        )
    # `_template.yaml` style files (leading underscore) are excluded.
    (tmp_path / "sources" / "jurisdictions" / "_template.yaml").write_text(
        "jurisdiction: TEMPLATE\nsources: []\n", encoding="utf-8"
    )

    econ = list(available_economies(tmp_path))
    assert econ == ["malaysia", "singapore"]


# ─── real Malaysia PDPA sample PDF (skip-if-missing) ─────────────────────────


@pytest.mark.skipif(
    not MALAYSIA_PDPA_PDF.exists(),
    reason="Malaysia PDPA sample PDF not present in this checkout",
)
def test_load_real_malaysia_pdpa_pdf(tmp_path: Path) -> None:
    _seed_schemas(tmp_path)
    # Point the registry at the real sample PDF via a path under tmp_path so the
    # IngestService resolves `repo_root / local_path`. Symlink it in.
    samples = tmp_path / "data" / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    linked = samples / "my_pdpa_2010.pdf"
    linked.write_bytes(MALAYSIA_PDPA_PDF.read_bytes())

    _write_registry(
        tmp_path,
        "Malaysia",
        [
            {
                "source_id": "my_pdpa_act709_2010",
                "title": "Personal Data Protection Act 2010",
                "local_path": "data/samples/my_pdpa_2010.pdf",
                "document_type": "statute",
                "authority_tier": "tier_1_statute",
                "extra": {"law_number_ref": "Act 709", "last_amended": "2024"},
            }
        ],
    )

    ingestor = EconomyIngestor.for_repo(tmp_path)
    docs = ingestor.load_economy_documents("Malaysia")
    assert len(docs) == 1
    meta, raw, extra = docs[0]
    # PDFs start with the %PDF magic header.
    assert raw[:5] == b"%PDF-"
    assert len(raw) > 1000
    assert meta.sha256 == compute_sha256(raw)
    assert extra["law_number_ref"] == "Act 709"

    # sha256-keyed cache landed under data/cache/raw/.
    cache_file = tmp_path / "data" / "cache" / "raw" / f"{meta.sha256}.bin"
    assert cache_file.exists()
