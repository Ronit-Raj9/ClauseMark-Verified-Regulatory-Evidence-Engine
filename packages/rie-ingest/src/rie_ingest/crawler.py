"""`SourceDiscoveryCrawler` — thin Phase 2 discovery crawler.

Discovers candidate legal documents on a jurisdiction portal and emits
provisional `SourceRegistryEntry` rows (NOT auto-promoted: `authority_tier` is
always the lowest provisional tier, for a human reviewer to confirm).

Strict scope guarantees:
  * NO LLM calls anywhere — discovery is pure heuristics + deterministic regex.
  * Robots.txt is honoured when `respect_robots` (RFC 9309: absent → allow-all).
  * Same-domain BFS bounded by `max_pages`; sitemap.xml preferred when present.
  * `offline_seed_dir` makes the crawler hermetic for the cached-corpus demo
    (systemArchitecture §11): seed pages are read from disk, never the network.
  * `fetch_and_register` reuses `IngestService` byte-fetch + sha256 cache and
    dedups by content hash, so the same bytes never enter the corpus twice.
"""

from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import httpx
from rie_contracts import (
    AuthorityTier,
    DocumentMeta,
    DocumentType,
    SourceRegistryEntry,
)

from rie_ingest.robots import DEFAULT_USER_AGENT, RobotsPolicy
from rie_ingest.service import IngestService

_DEFAULT_TIMEOUT_SECONDS: float = 30.0

# Heuristic: a URL or anchor text that looks like primary legal material.
# Pure regex — the system never lets an LLM author this judgment.
_LEGAL_HINT_RE: re.Pattern[str] = re.compile(
    # Keyword must start at a non-letter boundary and end at a non-letter
    # boundary so "contact" never matches "act"; treats `_`, `-`, `/`, `.` as
    # separators — unlike `\b`, which counts `_` as a word char and would miss
    # "regulation_2020". The trailing `.pdf` is a direct extension hint.
    r"(?<![a-z])(?:act|acts|statute|statutes|regulation|regulations|law|laws|gazette)(?![a-z])"
    r"|\.pdf(?![a-z])",
    re.IGNORECASE,
)

# Anchor scrape — deliberately tolerant; gov portals link via plain `<a href>`.
_ANCHOR_RE: re.Pattern[str] = re.compile(
    r"<a\b[^>]*\bhref\s*=\s*['\"]([^'\"#]+)['\"][^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)

# Sitemap `<loc>` extraction without committing to a namespace.
_SITEMAP_LOC_RE: re.Pattern[str] = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)

# Extension → provisional document_type sniff. Reviewer overrides.
_EXT_TO_DOCUMENT_TYPE: dict[str, DocumentType] = {
    ".pdf": DocumentType.STATUTE,
    ".html": DocumentType.OTHER,
    ".htm": DocumentType.OTHER,
    ".txt": DocumentType.OTHER,
}


def _normalise(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    cleaned = parsed._replace(
        scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower(), fragment=""
    )
    return urllib.parse.urlunparse(cleaned)


_TERMINAL_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".doc", ".docx", ".txt", ".xml", ".zip", ".rtf"}
)


def _is_terminal_doc(url: str) -> bool:
    """A link to a downloadable document, not a crawlable HTML page."""
    return Path(urllib.parse.urlparse(url).path).suffix.lower() in _TERMINAL_EXTENSIONS


def _same_domain(a: str, b: str) -> bool:
    return urllib.parse.urlparse(a).netloc.lower() == urllib.parse.urlparse(b).netloc.lower()


def _looks_legal(url: str, anchor_text: str = "") -> bool:
    # Match against the path + query only — the host (e.g. "laws.gov") would
    # otherwise false-positive every URL on a legal portal.
    parsed = urllib.parse.urlparse(url)
    path_target = parsed.path
    if parsed.query:
        path_target = f"{path_target}?{parsed.query}"
    if not parsed.scheme and not parsed.netloc:
        path_target = url
    return bool(_LEGAL_HINT_RE.search(path_target) or _LEGAL_HINT_RE.search(anchor_text))


def _sniff_document_type(url: str) -> DocumentType:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix in _EXT_TO_DOCUMENT_TYPE:
        return _EXT_TO_DOCUMENT_TYPE[suffix]
    # Path-keyword fall-through for extension-less legal portals.
    lowered = url.lower()
    if "regulation" in lowered:
        return DocumentType.REGULATION
    if "gazette" in lowered or "notice" in lowered:
        return DocumentType.NOTICE
    if "guideline" in lowered:
        return DocumentType.GUIDELINE
    return DocumentType.OTHER


def _source_id_for(url: str, jurisdiction: str) -> str:
    """Deterministic, human-legible id from the URL path. Stable across runs."""
    parsed = urllib.parse.urlparse(url)
    slug = re.sub(r"[^a-z0-9]+", "_", parsed.path.lower()).strip("_") or "root"
    host = re.sub(r"[^a-z0-9]+", "_", parsed.netloc.lower()).strip("_")
    return f"{jurisdiction.lower()}_{host}_{slug}"


def _extract_anchors(html: str, base_url: str) -> list[tuple[str, str]]:
    """Return `(absolute_url, anchor_text)` pairs from an HTML body."""
    out: list[tuple[str, str]] = []
    for match in _ANCHOR_RE.finditer(html):
        href = match.group(1).strip()
        if not href:
            continue
        scheme = urllib.parse.urlparse(href).scheme.lower()
        if scheme in {"javascript", "mailto", "tel", "data"}:
            continue
        absolute = urllib.parse.urljoin(base_url, href)
        absolute, _ = urllib.parse.urldefrag(absolute)
        text = re.sub(r"<[^>]+>", " ", match.group(2))
        text = re.sub(r"\s+", " ", text).strip()
        out.append((_normalise(absolute), text))
    return out


def _extract_sitemap_urls(body: bytes) -> list[str]:
    """Parse sitemap.xml `<loc>` entries. Falls back to regex if XML is dirty."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        text = body.decode("utf-8", errors="replace")
        return [m.group(1).strip() for m in _SITEMAP_LOC_RE.finditer(text)]
    locs: list[str] = []
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1].lower()
        if tag == "loc" and elem.text:
            locs.append(elem.text.strip())
    return locs


class SourceDiscoveryCrawler:
    """Thin same-domain discovery crawler emitting provisional registry rows."""

    def __init__(
        self,
        http_client: httpx.Client | None = None,
        respect_robots: bool = True,
        max_pages: int = 50,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        offline_seed_dir: Path | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.http_client = http_client
        self.respect_robots = respect_robots
        self.max_pages = max_pages
        self.timeout = timeout
        self.offline_seed_dir = offline_seed_dir
        self.user_agent = user_agent

    # ─── Public API ───────────────────────────────────────────────────────────

    def discover(self, seed_url: str, jurisdiction: str) -> list[SourceRegistryEntry]:
        """Discover likely legal documents reachable from `seed_url`.

        Offline mode (`offline_seed_dir`) short-circuits all network access and
        scans pre-fetched HTML files from disk instead — the demo never hits a
        live government site.
        """
        if self.offline_seed_dir is not None:
            return self._discover_offline(jurisdiction)

        client = self.http_client
        owned = False
        if client is None:
            client = httpx.Client(
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
            owned = True
        try:
            return self._discover_online(client, seed_url, jurisdiction)
        finally:
            if owned:
                client.close()

    def fetch_and_register(
        self,
        entries: Sequence[SourceRegistryEntry],
        service: IngestService,
    ) -> list[tuple[DocumentMeta, bytes]]:
        """Fetch bytes for each entry via `IngestService` (sha256-keyed cache),
        deduping by content sha256 so identical bytes register only once."""
        out: list[tuple[DocumentMeta, bytes]] = []
        seen_sha: set[str] = set()
        for entry in entries:
            meta, raw = service.load_document_bytes(entry)
            if meta.sha256 in seen_sha:
                continue
            seen_sha.add(meta.sha256)
            out.append((meta, raw))
        return out

    # ─── Offline (cached-corpus demo) ─────────────────────────────────────────

    def _discover_offline(self, jurisdiction: str) -> list[SourceRegistryEntry]:
        seed_dir = self.offline_seed_dir
        assert seed_dir is not None  # guarded by caller
        anchors: list[tuple[str, str]] = []
        for path in sorted(seed_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".html", ".htm"}:
                continue
            html = path.read_text(encoding="utf-8", errors="replace")
            base = path.as_uri()
            anchors.extend(_extract_anchors(html, base))
        # Also surface the seed files themselves if they look legal.
        seeds: list[tuple[str, str]] = [
            (p.as_uri(), p.name)
            for p in sorted(seed_dir.rglob("*"))
            if p.is_file() and _looks_legal(p.name)
        ]
        return self._entries_from_hits(seeds + anchors, jurisdiction)

    # ─── Online crawl ─────────────────────────────────────────────────────────

    def _discover_online(
        self, client: httpx.Client, seed_url: str, jurisdiction: str
    ) -> list[SourceRegistryEntry]:
        seed = _normalise(seed_url)
        policy = self._load_robots(client, seed)

        sitemap_urls = self._try_sitemap(client, seed, policy)
        if sitemap_urls:
            hits = [
                (u, "") for u in sitemap_urls if _same_domain(u, seed) and self._allowed(policy, u)
            ]
            return self._entries_from_hits(hits, jurisdiction)

        return self._entries_from_hits(self._bfs(client, seed, policy), jurisdiction)

    def _load_robots(self, client: httpx.Client, seed: str) -> RobotsPolicy | None:
        if not self.respect_robots:
            return None
        parsed = urllib.parse.urlparse(seed)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            resp = client.get(robots_url, headers={"User-Agent": self.user_agent})
        except httpx.HTTPError:
            return RobotsPolicy.from_text("")
        if resp.status_code == 200:
            return RobotsPolicy.from_text(resp.text)
        # Absent robots.txt → allow-all (RFC 9309).
        return RobotsPolicy.from_text("")

    def _allowed(self, policy: RobotsPolicy | None, url: str) -> bool:
        if policy is None:
            return True
        return policy.allowed(url, user_agent=self.user_agent)

    def _try_sitemap(
        self, client: httpx.Client, seed: str, policy: RobotsPolicy | None
    ) -> list[str]:
        parsed = urllib.parse.urlparse(seed)
        sitemap_url = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
        if not self._allowed(policy, sitemap_url):
            return []
        try:
            resp = client.get(sitemap_url, headers={"User-Agent": self.user_agent})
        except httpx.HTTPError:
            return []
        if resp.status_code != 200:
            return []
        return [_normalise(u) for u in _extract_sitemap_urls(resp.content)]

    def _bfs(
        self, client: httpx.Client, seed: str, policy: RobotsPolicy | None
    ) -> list[tuple[str, str]]:
        seen: set[str] = set()
        hits: list[tuple[str, str]] = []
        queue: deque[str] = deque([seed])
        pages = 0
        while queue and pages < self.max_pages:
            url = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            if not self._allowed(policy, url):
                continue
            try:
                resp = client.get(url, headers={"User-Agent": self.user_agent})
            except httpx.HTTPError:
                continue
            pages += 1
            if resp.status_code != 200:
                continue
            content_type = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            is_html = content_type in {"text/html", "application/xhtml+xml"} or url.endswith(
                (".html", ".htm")
            )
            if not is_html:
                continue
            html = resp.content.decode("utf-8", errors="replace")
            for link, text in _extract_anchors(html, base_url=url):
                if not _same_domain(link, seed):
                    continue
                # Robots applies to a discovered document too: a disallowed
                # path must never enter the corpus, fetched or not.
                if not self._allowed(policy, link):
                    continue
                if _looks_legal(link, text):
                    hits.append((link, text))
                # Only recurse into pages that are plausibly HTML. A link whose
                # extension marks a terminal document (e.g. `.pdf`) is recorded
                # as a hit above but never fetched — it is a leaf, not a page.
                if _is_terminal_doc(link):
                    continue
                if link not in seen and link not in queue:
                    queue.append(link)
        return hits

    # ─── Hit → entry ──────────────────────────────────────────────────────────

    def _entries_from_hits(
        self, hits: Sequence[tuple[str, str]], jurisdiction: str
    ) -> list[SourceRegistryEntry]:
        entries: list[SourceRegistryEntry] = []
        seen_urls: set[str] = set()
        for raw_url, anchor in hits:
            url = _normalise(raw_url)
            if url in seen_urls:
                continue
            if not _looks_legal(url, anchor):
                continue
            seen_urls.add(url)
            entries.append(
                SourceRegistryEntry(
                    source_id=_source_id_for(url, jurisdiction),
                    jurisdiction=jurisdiction,
                    title=anchor or Path(urllib.parse.urlparse(url).path).name or url,
                    source_url=url,
                    document_type=_sniff_document_type(url),
                    # Provisional — discovery never confirms authority.
                    authority_tier=AuthorityTier.TIER_3_GUIDELINE,
                )
            )
        return entries
