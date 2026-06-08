"""Tests for `rie_ingest.crawler.SourceDiscoveryCrawler` (Phase 2 thin crawler).

Network is mocked with respx. Covers: sitemap discovery, BFS fallback, legal
filter + dedup, max_pages bound, robots disallow respected, and the offline
seed-dir hermetic path for the cached-corpus demo.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import respx
from rie_config import ConfigRepository
from rie_contracts import AuthorityTier, DocumentType, IngestPort, SourceRegistryEntry
from rie_ingest import IngestService, SourceDiscoveryCrawler

BASE = "https://laws.example.gov"


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": "rie-bot"}, follow_redirects=True)


# ─── Sitemap discovery ────────────────────────────────────────────────────────


@respx.mock
def test_discover_via_sitemap() -> None:
    respx.get(f"{BASE}/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow:\n")
    )
    sitemap = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{BASE}/acts/dpa.pdf</loc></url>"
        f"<url><loc>{BASE}/about/contact.html</loc></url>"
        f"<url><loc>{BASE}/regulation/2020.pdf</loc></url>"
        "</urlset>"
    )
    respx.get(f"{BASE}/sitemap.xml").mock(
        return_value=httpx.Response(200, text=sitemap, headers={"content-type": "application/xml"})
    )

    crawler = SourceDiscoveryCrawler(http_client=_client())
    entries = crawler.discover(f"{BASE}/index.html", "SAMPLE")

    urls = sorted(e.source_url or "" for e in entries)
    # contact.html is filtered out (no legal hint); the two PDFs survive.
    assert urls == [f"{BASE}/acts/dpa.pdf", f"{BASE}/regulation/2020.pdf"]
    assert all(isinstance(e, SourceRegistryEntry) for e in entries)
    assert all(e.authority_tier == AuthorityTier.TIER_3_GUIDELINE for e in entries)
    by_url = {e.source_url: e for e in entries}
    assert by_url[f"{BASE}/acts/dpa.pdf"].document_type == DocumentType.STATUTE
    assert by_url[f"{BASE}/regulation/2020.pdf"].document_type == DocumentType.STATUTE


# ─── BFS fallback + robots disallow ───────────────────────────────────────────


@respx.mock
def test_discover_bfs_respects_robots_and_dedups() -> None:
    respx.get(f"{BASE}/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
    )
    # No sitemap → BFS.
    respx.get(f"{BASE}/sitemap.xml").mock(return_value=httpx.Response(404))

    index = (
        "<html><body>"
        f'<a href="{BASE}/acts/dpa.pdf">Data Protection Act</a>'
        f'<a href="{BASE}/acts/dpa.pdf">duplicate link</a>'
        f'<a href="{BASE}/private/secret.pdf">hidden statute</a>'
        f'<a href="{BASE}/news/index.html">News</a>'
        "</body></html>"
    )
    respx.get(f"{BASE}/index.html").mock(
        return_value=httpx.Response(200, text=index, headers={"content-type": "text/html"})
    )
    respx.get(f"{BASE}/news/index.html").mock(
        return_value=httpx.Response(
            200,
            text=f'<html><body><a href="{BASE}/law/gazette.html">Gazette</a></body></html>',
            headers={"content-type": "text/html"},
        )
    )
    respx.get(f"{BASE}/law/gazette.html").mock(
        return_value=httpx.Response(
            200, text="<html></html>", headers={"content-type": "text/html"}
        )
    )

    crawler = SourceDiscoveryCrawler(http_client=_client())
    entries = crawler.discover(f"{BASE}/index.html", "SAMPLE")
    urls = sorted(e.source_url or "" for e in entries)

    # dpa.pdf once (dedup), gazette.html (legal anchor); private blocked by robots.
    assert urls == [f"{BASE}/acts/dpa.pdf", f"{BASE}/law/gazette.html"]
    assert not any("private" in (e.source_url or "") for e in entries)


@respx.mock
def test_robots_disallow_not_respected_when_flag_false() -> None:
    respx.get(f"{BASE}/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /\n")
    )
    respx.get(f"{BASE}/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/index.html").mock(
        return_value=httpx.Response(
            200,
            text=f'<html><body><a href="{BASE}/acts/x.pdf">act</a></body></html>',
            headers={"content-type": "text/html"},
        )
    )

    crawler = SourceDiscoveryCrawler(http_client=_client(), respect_robots=False)
    entries = crawler.discover(f"{BASE}/index.html", "SAMPLE")
    assert [e.source_url for e in entries] == [f"{BASE}/acts/x.pdf"]


# ─── max_pages bound ──────────────────────────────────────────────────────────


@respx.mock
def test_max_pages_bound_halts_crawl() -> None:
    respx.get(f"{BASE}/robots.txt").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/sitemap.xml").mock(return_value=httpx.Response(404))

    # Chain: page0 -> page1 -> page2 ... each links the next + one PDF.
    for i in range(6):
        nxt = f"{BASE}/p{i + 1}.html"
        body = (
            f'<html><body><a href="{BASE}/act{i}.pdf">act {i}</a>'
            f'<a href="{nxt}">next</a></body></html>'
        )
        respx.get(f"{BASE}/p{i}.html").mock(
            return_value=httpx.Response(200, text=body, headers={"content-type": "text/html"})
        )

    crawler = SourceDiscoveryCrawler(http_client=_client(), max_pages=2)
    entries = crawler.discover(f"{BASE}/p0.html", "SAMPLE")
    # Only 2 HTML pages fetched → at most act0 + act1 discovered.
    pdf_urls = {e.source_url for e in entries}
    assert pdf_urls <= {f"{BASE}/act0.pdf", f"{BASE}/act1.pdf"}
    assert len(pdf_urls) <= 2


# ─── Offline seed dir (cached-corpus demo) ────────────────────────────────────


def test_discover_offline_seed_dir(tmp_path: Path) -> None:
    seed = tmp_path / "portal"
    seed.mkdir()
    (seed / "index.html").write_text(
        "<html><body>"
        '<a href="acts/dpa.pdf">Data Protection Act</a>'
        '<a href="about.html">About</a>'
        "</body></html>",
        encoding="utf-8",
    )
    (seed / "regulation_2020.html").write_text("<html></html>", encoding="utf-8")

    crawler = SourceDiscoveryCrawler(offline_seed_dir=seed)
    entries = crawler.discover("ignored://seed", "SAMPLE")

    urls = [e.source_url or "" for e in entries]
    # The pdf anchor and the legal-looking seed file surface; about.html dropped.
    assert any(u.endswith("acts/dpa.pdf") for u in urls)
    assert any(u.endswith("regulation_2020.html") for u in urls)
    assert not any("about.html" in u for u in urls)
    assert all(e.authority_tier == AuthorityTier.TIER_3_GUIDELINE for e in entries)


# ─── fetch_and_register reuses IngestService + sha256 dedup ───────────────────


def test_fetch_and_register_dedups_by_sha256(tmp_path: Path) -> None:
    # Two local files with identical bytes → one registered entry.
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "a.txt").write_bytes(b"IDENTICAL LEGAL TEXT")
    (samples / "b.txt").write_bytes(b"IDENTICAL LEGAL TEXT")
    (samples / "c.txt").write_bytes(b"DIFFERENT TEXT")

    service = IngestService(repo_root=tmp_path, config=ConfigRepository(repo_root=tmp_path))
    entries = [
        SourceRegistryEntry(
            source_id=f"s_{name}",
            jurisdiction="SAMPLE",
            title=name,
            document_type=DocumentType.OTHER,
            authority_tier=AuthorityTier.TIER_3_GUIDELINE,
            local_path=f"samples/{name}",
        )
        for name in ("a.txt", "b.txt", "c.txt")
    ]

    crawler = SourceDiscoveryCrawler()
    out = crawler.fetch_and_register(entries, service)
    shas = {m.sha256 for m, _ in out}
    assert len(out) == 2  # a/b collapse, c distinct
    assert len(shas) == 2


# ─── Contract-shape sanity: discovered entries are valid registry rows ────────


@respx.mock
def test_discovered_entries_are_ingest_compatible(tmp_path: Path) -> None:
    respx.get(f"{BASE}/robots.txt").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/sitemap.xml").mock(
        return_value=httpx.Response(
            200,
            text=f"<urlset><url><loc>{BASE}/acts/dpa.pdf</loc></url></urlset>",
            headers={"content-type": "application/xml"},
        )
    )
    crawler = SourceDiscoveryCrawler(http_client=_client())
    entries = crawler.discover(f"{BASE}/index.html", "SAMPLE")
    assert entries
    # IngestService implements IngestPort; the entries it consumes are exactly
    # SourceRegistryEntry — confirm the crawler yields that contract type.
    service = IngestService(repo_root=tmp_path, config=ConfigRepository(repo_root=tmp_path))
    assert isinstance(service, IngestPort)
    assert all(isinstance(e, SourceRegistryEntry) for e in entries)
