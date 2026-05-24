"""Tests for `rie_ingest.discovery` — the Phase 2 thin crawler.

Covers:
  * `discovery:` block parsing (optional, backward compatible).
  * Robots.txt respected — a Disallow blocks the URL.
  * Diff cache short-circuits a second run on unchanged content.
  * Local `file://` seed via a custom httpx transport.
  * Proposal YAML output.

No real network is touched — `httpx.MockTransport` provides every byte.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import yaml

from rie_ingest.discovery import (
    USER_AGENT,
    DiscoveryConfig,
    DiscoveryConfigError,
    DiscoveryCrawler,
    load_discovery_config,
    write_proposals,
)

# ─── Helpers ────────────────────────────────────────────────────────────────


def _write_jurisdiction_yaml(repo_root: Path, jurisdiction: str, body: str) -> Path:
    path = repo_root / "sources" / "jurisdictions" / f"{jurisdiction.lower()}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _make_transport(routes: dict[str, httpx.Response]) -> httpx.MockTransport:
    """Build an httpx MockTransport whose handler returns the response keyed by
    the request URL string (full URL including scheme/host)."""

    def handler(request: httpx.Request) -> httpx.Response:
        # Strip fragments to match how the crawler normalises URLs.
        url = str(request.url).split("#", 1)[0]
        if url in routes:
            return routes[url]
        return httpx.Response(404, content=b"not found")

    return httpx.MockTransport(handler)


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    (tmp_path / "sources" / "jurisdictions").mkdir(parents=True)
    (tmp_path / "data" / "cache" / "discovery").mkdir(parents=True)
    (tmp_path / "data" / "outputs" / "discovery").mkdir(parents=True)
    return tmp_path


# ─── Config parsing ─────────────────────────────────────────────────────────


def test_load_discovery_config_parses_full_block(repo_root: Path) -> None:
    _write_jurisdiction_yaml(
        repo_root,
        "test",
        """
jurisdiction: "TEST"
language: "en"
sources: []
discovery:
  seed_urls:
    - "https://laws.example.gov/index.html"
  crawl_patterns:
    link_regexes:
      - "/acts/"
      - "\\\\.pdf$"
    mime_allow:
      - "application/pdf"
      - "text/html"
  depth_limit: 1
  crawl_delay_seconds: 0.0
  max_pages: 10
""",
    )
    cfg = load_discovery_config(repo_root, "TEST")
    assert cfg.jurisdiction == "TEST"
    assert cfg.seed_urls == ("https://laws.example.gov/index.html",)
    assert cfg.depth_limit == 1
    assert cfg.crawl_delay_seconds == 0.0
    assert cfg.max_pages == 10
    assert "application/pdf" in cfg.patterns.mime_allow
    # Both regexes must compile and match the expected URLs.
    assert any(p.search("https://x/acts/foo") for p in cfg.patterns.link_regexes)
    assert any(p.search("https://x/a.pdf") for p in cfg.patterns.link_regexes)


def test_load_discovery_config_missing_block_raises(repo_root: Path) -> None:
    _write_jurisdiction_yaml(
        repo_root,
        "nodisc",
        'jurisdiction: "NODISC"\nlanguage: "en"\nsources: []\n',
    )
    with pytest.raises(DiscoveryConfigError, match="no `discovery:` block"):
        load_discovery_config(repo_root, "NODISC")


def test_load_discovery_config_rejects_bad_depth(repo_root: Path) -> None:
    _write_jurisdiction_yaml(
        repo_root,
        "bad",
        """
jurisdiction: "BAD"
sources: []
discovery:
  seed_urls: ["https://x.example.gov/"]
  depth_limit: -1
""",
    )
    with pytest.raises(DiscoveryConfigError, match="depth_limit"):
        load_discovery_config(repo_root, "BAD")


# ─── Crawl: robots + diff cache + proposals ─────────────────────────────────


def _seed_yaml_for(jurisdiction: str, seed: str) -> str:
    return f"""
jurisdiction: "{jurisdiction}"
language: "en"
sources: []
discovery:
  seed_urls:
    - "{seed}"
  crawl_patterns:
    link_regexes: []
    mime_allow:
      - "application/pdf"
      - "text/html"
  depth_limit: 2
  crawl_delay_seconds: 0.0
  max_pages: 50
"""


def test_crawl_respects_robots_and_diff_cache(repo_root: Path) -> None:
    """Two-stage test:

    1. First crawl: index page links to ``allowed.pdf`` and ``blocked.pdf``;
       robots.txt disallows ``/private/``. Expect one proposal for the PDF
       under the allowed prefix, none for the blocked one.
    2. Second crawl: same bytes. Expect ZERO proposals (diff cache hit).
    """
    seed = "https://laws.example.gov/index.html"
    _write_jurisdiction_yaml(repo_root, "TEST", _seed_yaml_for("TEST", seed))

    index_html = (
        b"<html><head><title>Index</title></head><body>"
        b'<a href="/public/allowed.pdf">allowed</a>'
        b'<a href="/private/blocked.pdf">blocked</a>'
        b"</body></html>"
    )
    pdf_bytes = b"%PDF-1.4 allowed body"
    blocked_pdf = b"%PDF-1.4 blocked body"

    robots_body = (
        "User-agent: *\n"
        "Disallow: /private/\n"
    )

    routes: dict[str, httpx.Response] = {
        "https://laws.example.gov/robots.txt": httpx.Response(
            200, content=robots_body.encode(), headers={"content-type": "text/plain"}
        ),
        "https://laws.example.gov/index.html": httpx.Response(
            200, content=index_html, headers={"content-type": "text/html; charset=utf-8"}
        ),
        "https://laws.example.gov/public/allowed.pdf": httpx.Response(
            200, content=pdf_bytes, headers={"content-type": "application/pdf"}
        ),
        "https://laws.example.gov/private/blocked.pdf": httpx.Response(
            200, content=blocked_pdf, headers={"content-type": "application/pdf"}
        ),
    }

    cfg = load_discovery_config(repo_root, "TEST")
    # Stable clock for deterministic `retrieved_at`.
    fixed_now = datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC)
    crawler = DiscoveryCrawler(
        repo_root=repo_root,
        config=cfg,
        http_client=httpx.Client(
            transport=_make_transport(routes),
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ),
        now_fn=lambda: fixed_now,
    )
    proposals_1 = crawler.crawl()
    # Exactly one accepted (the PDF and the HTML index).
    urls = sorted(p.url for p in proposals_1)
    assert urls == [
        "https://laws.example.gov/index.html",
        "https://laws.example.gov/public/allowed.pdf",
    ]
    # Blocked URL must NOT appear anywhere in the proposals.
    assert not any("private" in p.url for p in proposals_1)
    # PDF proposal must carry the correct mime + document_type guess.
    pdf_prop = next(p for p in proposals_1 if p.url.endswith("allowed.pdf"))
    assert pdf_prop.content_type == "application/pdf"
    assert pdf_prop.document_type == "statute"
    assert pdf_prop.status == "pending_review"
    assert pdf_prop.jurisdiction == "TEST"
    assert pdf_prop.retrieved_at == fixed_now.isoformat()

    # Diff cache must have been persisted.
    cache_path = repo_root / "data" / "cache" / "discovery" / "test.json"
    assert cache_path.exists()

    # ─── Second crawl: identical responses → zero proposals ────────────────
    crawler2 = DiscoveryCrawler(
        repo_root=repo_root,
        config=cfg,
        http_client=httpx.Client(
            transport=_make_transport(routes),
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ),
        now_fn=lambda: fixed_now,
    )
    proposals_2 = crawler2.crawl()
    assert proposals_2 == [], "unchanged content must produce zero proposals on rerun"


def test_crawl_local_file_seed(repo_root: Path, tmp_path: Path) -> None:
    """A `file://` seed is supported via a custom transport that resolves
    the URL against a temp directory. Demonstrates that the crawler does
    not assume HTTPS-only — useful for offline / hermetic testing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "index.html").write_bytes(
        b'<html><body><a href="/laws/a.pdf">a</a></body></html>'
    )
    (docs_dir / "laws").mkdir()
    (docs_dir / "laws" / "a.pdf").write_bytes(b"%PDF-1.4 local body")

    seed = "file://localhost/index.html"
    _write_jurisdiction_yaml(repo_root, "LOCAL", _seed_yaml_for("LOCAL", seed))

    def file_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/robots.txt"):
            # `file://` has no robots; mimic a 404 so the parser treats it as
            # "no rules" (allow).
            return httpx.Response(404, content=b"")
        rel = request.url.path.lstrip("/")
        target = docs_dir / rel
        if not target.exists() or not target.is_file():
            return httpx.Response(404, content=b"")
        if target.suffix == ".pdf":
            ct = "application/pdf"
        elif target.suffix in {".html", ".htm"}:
            ct = "text/html; charset=utf-8"
        else:
            ct = "application/octet-stream"
        return httpx.Response(200, content=target.read_bytes(), headers={"content-type": ct})

    cfg = load_discovery_config(repo_root, "LOCAL")
    crawler = DiscoveryCrawler(
        repo_root=repo_root,
        config=cfg,
        http_client=httpx.Client(
            transport=httpx.MockTransport(file_handler),
            headers={"User-Agent": USER_AGENT},
        ),
    )
    proposals = crawler.crawl()
    urls = sorted(p.url for p in proposals)
    assert urls == [
        "file://localhost/index.html",
        "file://localhost/laws/a.pdf",
    ]


# ─── Output ─────────────────────────────────────────────────────────────────


def test_write_proposals_emits_yaml(repo_root: Path) -> None:
    cfg = DiscoveryConfig(
        jurisdiction="OUT",
        seed_urls=("https://x.example.gov/",),
        patterns=__import__("rie_ingest.discovery", fromlist=["CrawlPatterns"]).CrawlPatterns(),
    )
    _ = cfg  # silence unused (kept for documentation parity)
    from rie_ingest.discovery import ProposedSource

    props = [
        ProposedSource(
            url="https://x.example.gov/a.pdf",
            jurisdiction="OUT",
            document_type="statute",
            retrieved_at="2026-05-24T12:00:00+00:00",
            sha256="0" * 64,
            content_type="application/pdf",
            title=None,
            discovered_from="https://x.example.gov/",
        )
    ]
    fixed_now = datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC)
    out_path = write_proposals(repo_root, "OUT", props, now=fixed_now)
    assert out_path.exists()
    assert out_path.parent == repo_root / "data" / "outputs" / "discovery" / "out"
    payload = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert payload["jurisdiction"] == "OUT"
    assert payload["generator"] == USER_AGENT
    assert len(payload["proposals"]) == 1
    assert payload["proposals"][0]["status"] == "pending_review"
