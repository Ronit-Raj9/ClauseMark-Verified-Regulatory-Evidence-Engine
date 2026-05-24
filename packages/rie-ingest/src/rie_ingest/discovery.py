"""Polite, thin discovery crawler for jurisdiction portals (Phase 2).

This module is *additive* — `IngestService` and its `IngestPort` implementation
are unchanged. Discovery proposes new `ProposedSource` rows for human review;
nothing it produces is auto-promoted into the source registry.

Design notes:
  * Reads an *optional* `discovery:` block from a jurisdiction YAML
    (`sources/jurisdictions/<j>.yaml`). Schema is backward compatible.
  * Polite by construction: honours `robots.txt`, per-host concurrency 1,
    configurable crawl-delay (default 1.0s), explicit `User-Agent`.
  * Diff cache keyed by sha256(content) per URL, stored as JSON at
    `data/cache/discovery/<jurisdiction>.json`. Unchanged URLs are skipped.
  * Proposals serialised to YAML at
    `data/outputs/discovery/<jurisdiction>/<utc-timestamp>.yaml`.
  * NO LLM calls. NO mutation of contracts.

Feature flag: only ever runs via `python -m rie_ingest.discovery crawl <j>` or
the console script `rie-discovery`. Importing the module is side-effect free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.robotparser
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

# ─── Constants ──────────────────────────────────────────────────────────────

USER_AGENT: str = "RIE-DiscoveryBot/0.1 (hackathon; contact via repo)"
"""Stable UA string — used both for robots.txt resolution and for fetches.
Government portals occasionally block unknown UAs; we identify ourselves
honestly so admins can see who we are and reach us."""

DEFAULT_CRAWL_DELAY_SECONDS: float = 1.0
DEFAULT_DEPTH_LIMIT: int = 2
DEFAULT_MAX_PAGES: int = 200
DEFAULT_HTTP_TIMEOUT_SECONDS: float = 30.0

_MIME_TO_DOCUMENT_TYPE: dict[str, str] = {
    # Conservative mapping. The LLM never authors a fact; this is heuristic
    # metadata for the human reviewer, NOT a verified document_type.
    "application/pdf": "statute",
    "text/html": "other",
    "application/xhtml+xml": "other",
    "text/plain": "other",
    "application/msword": "other",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "other",
}

_DEFAULT_MIME_ALLOW: frozenset[str] = frozenset(
    {
        "application/pdf",
        "text/html",
        "application/xhtml+xml",
        "text/plain",
    }
)

# ─── Public dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class CrawlPatterns:
    """Per-jurisdiction crawl policy.

    `link_regexes` are matched against absolute, normalised URLs *before*
    fetching — a link that does not match any regex is silently skipped.
    `mime_allow` filters by `Content-Type` *after* fetching (we cannot know
    the MIME without a HEAD/GET; we GET because servers vary in HEAD support).
    """

    link_regexes: tuple[re.Pattern[str], ...] = field(default_factory=tuple)
    mime_allow: frozenset[str] = field(default_factory=lambda: _DEFAULT_MIME_ALLOW)


@dataclass(frozen=True)
class DiscoveryConfig:
    """Parsed `discovery:` block from a jurisdiction YAML."""

    jurisdiction: str
    seed_urls: tuple[str, ...]
    patterns: CrawlPatterns
    depth_limit: int = DEFAULT_DEPTH_LIMIT
    crawl_delay_seconds: float = DEFAULT_CRAWL_DELAY_SECONDS
    max_pages: int = DEFAULT_MAX_PAGES


@dataclass(frozen=True)
class ProposedSource:
    """A candidate `SourceRegistryEntry` for human confirmation.

    NOT a contract type — discovery is upstream of the registry. The
    Substantive Lead reviews these YAMLs and promotes accepted rows into
    `sources/jurisdictions/<j>.yaml`.
    """

    url: str
    jurisdiction: str
    document_type: str  # heuristic guess from MIME; reviewer overrides
    retrieved_at: str  # ISO-8601 UTC
    sha256: str
    status: str = "pending_review"
    content_type: str | None = None
    title: str | None = None
    discovered_from: str | None = None  # parent URL in the crawl tree


# ─── Internals: config loading ──────────────────────────────────────────────


def load_discovery_config(repo_root: Path, jurisdiction: str) -> DiscoveryConfig:
    """Read `sources/jurisdictions/<j>.yaml` and parse its `discovery:` block.

    Raises `DiscoveryConfigError` if the block is missing or malformed.
    """
    path = repo_root / "sources" / "jurisdictions" / f"{jurisdiction.lower()}.yaml"
    if not path.exists():
        raise DiscoveryConfigError(f"missing jurisdiction file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DiscoveryConfigError(f"{path}: top-level must be a mapping")
    block = raw.get("discovery")
    if block is None:
        raise DiscoveryConfigError(
            f"{path}: no `discovery:` block — discovery is opt-in per jurisdiction"
        )
    if not isinstance(block, dict):
        raise DiscoveryConfigError(f"{path}: `discovery:` must be a mapping")

    seeds_raw = block.get("seed_urls", [])
    if not isinstance(seeds_raw, list) or not all(isinstance(s, str) for s in seeds_raw):
        raise DiscoveryConfigError(f"{path}: `discovery.seed_urls` must be a list[str]")
    seed_urls: tuple[str, ...] = tuple(seeds_raw)

    patterns_raw = block.get("crawl_patterns", {}) or {}
    if not isinstance(patterns_raw, dict):
        raise DiscoveryConfigError(f"{path}: `discovery.crawl_patterns` must be a mapping")

    link_regexes_raw = patterns_raw.get("link_regexes", []) or []
    if not isinstance(link_regexes_raw, list) or not all(
        isinstance(s, str) for s in link_regexes_raw
    ):
        raise DiscoveryConfigError(
            f"{path}: `discovery.crawl_patterns.link_regexes` must be a list[str]"
        )
    compiled: tuple[re.Pattern[str], ...] = tuple(re.compile(s) for s in link_regexes_raw)

    mime_allow_raw = patterns_raw.get("mime_allow")
    if mime_allow_raw is None:
        mime_allow: frozenset[str] = _DEFAULT_MIME_ALLOW
    else:
        if not isinstance(mime_allow_raw, list) or not all(
            isinstance(s, str) for s in mime_allow_raw
        ):
            raise DiscoveryConfigError(
                f"{path}: `discovery.crawl_patterns.mime_allow` must be a list[str]"
            )
        mime_allow = frozenset(mime_allow_raw)

    depth_limit_raw: Any = block.get("depth_limit", DEFAULT_DEPTH_LIMIT)
    if not isinstance(depth_limit_raw, int) or depth_limit_raw < 0:
        raise DiscoveryConfigError(f"{path}: `discovery.depth_limit` must be int >= 0")
    crawl_delay_raw: Any = block.get("crawl_delay_seconds", DEFAULT_CRAWL_DELAY_SECONDS)
    if not isinstance(crawl_delay_raw, (int, float)) or crawl_delay_raw < 0:
        raise DiscoveryConfigError(
            f"{path}: `discovery.crawl_delay_seconds` must be a non-negative number"
        )
    max_pages_raw: Any = block.get("max_pages", DEFAULT_MAX_PAGES)
    if not isinstance(max_pages_raw, int) or max_pages_raw <= 0:
        raise DiscoveryConfigError(f"{path}: `discovery.max_pages` must be int > 0")

    jur = raw.get("jurisdiction", jurisdiction)
    if not isinstance(jur, str) or not jur:
        raise DiscoveryConfigError(f"{path}: `jurisdiction` must be a non-empty string")

    return DiscoveryConfig(
        jurisdiction=jur,
        seed_urls=seed_urls,
        patterns=CrawlPatterns(link_regexes=compiled, mime_allow=mime_allow),
        depth_limit=int(depth_limit_raw),
        crawl_delay_seconds=float(crawl_delay_raw),
        max_pages=int(max_pages_raw),
    )


class DiscoveryConfigError(RuntimeError):
    """Raised when the `discovery:` block is missing or malformed."""


# ─── Internals: diff cache ──────────────────────────────────────────────────


@dataclass
class _DiffCache:
    """Persisted `{url: sha256}` map. The cache is best-effort; a missing or
    corrupt file degrades gracefully to "everything is new"."""

    path: Path
    entries: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> _DiffCache:
        if not path.exists():
            return cls(path=path, entries={})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(path=path, entries={})
        if not isinstance(data, dict):
            return cls(path=path, entries={})
        entries: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, str):
                entries[k] = v
        return cls(path=path, entries=entries)

    def is_unchanged(self, url: str, sha: str) -> bool:
        return self.entries.get(url) == sha

    def update(self, url: str, sha: str) -> None:
        self.entries[url] = sha

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.entries, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


# ─── Internals: link extraction ─────────────────────────────────────────────

_HREF_RE: re.Pattern[str] = re.compile(
    r"""href\s*=\s*['"]([^'"#]+)""", re.IGNORECASE
)


def _extract_links(html: str, base_url: str) -> list[str]:
    """Naive `href` scrape. We deliberately avoid a full HTML parser dep — the
    crawler is thin and gov portals predominantly link via plain `<a href>`.
    Anything we miss is fine: the human reviewer also browses the portal."""
    out: list[str] = []
    for match in _HREF_RE.finditer(html):
        raw = match.group(1).strip()
        if not raw:
            continue
        # Drop client-side schemes — not crawlable.
        scheme = urllib.parse.urlparse(raw).scheme.lower()
        if scheme in {"javascript", "mailto", "tel", "data"}:
            continue
        absolute = urllib.parse.urljoin(base_url, raw)
        # Strip fragment; we don't crawl by anchor.
        absolute, _ = urllib.parse.urldefrag(absolute)
        out.append(absolute)
    return out


def _same_host(a: str, b: str) -> bool:
    return urllib.parse.urlparse(a).netloc.lower() == urllib.parse.urlparse(b).netloc.lower()


def _normalise(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    # Lowercase scheme + host; preserve path/query as-is to avoid mangling.
    return urllib.parse.urlunparse(
        parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower(), fragment="")
    )


def _guess_document_type(content_type: str | None) -> str:
    if not content_type:
        return "other"
    primary = content_type.split(";", 1)[0].strip().lower()
    return _MIME_TO_DOCUMENT_TYPE.get(primary, "other")


def _matches_any(url: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    if not patterns:
        # Empty allow-list means "no constraint" — accept everything on host.
        return True
    return any(p.search(url) for p in patterns)


# ─── Robots ─────────────────────────────────────────────────────────────────


class _RobotsCache:
    """Per-host `robots.txt` cache. Fetches lazily, caches forever for the
    duration of a single crawl invocation. A failed fetch is treated as
    "allow" — robots.txt is advisory and many gov sites omit it — but a
    PRESENT robots.txt that disallows is honoured strictly."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client
        self._cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    def can_fetch(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        host_key = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._cache.get(host_key)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = urllib.parse.urljoin(host_key, "/robots.txt")
            rp.set_url(robots_url)
            try:
                resp = self._client.get(robots_url, headers={"User-Agent": USER_AGENT})
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    # No robots.txt — RFC 9309 treats this as "no rules".
                    rp.parse([])
            except (httpx.HTTPError, httpx.InvalidURL):
                rp.parse([])
            self._cache[host_key] = rp
        return rp.can_fetch(USER_AGENT, url)


# ─── Crawler ────────────────────────────────────────────────────────────────


@dataclass
class DiscoveryCrawler:
    """Polite, BFS, per-host-serial discovery crawler.

    Hexagonal seam: the crawler accepts an optional `httpx.Client` so tests
    can inject a transport-mocked client (respx) without monkey-patching.
    """

    repo_root: Path
    config: DiscoveryConfig
    http_client: httpx.Client | None = None
    http_timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS
    # Allow tests to inject a deterministic clock for the `retrieved_at`
    # field; default uses real wall time.
    now_fn: Any = field(default=lambda: datetime.now(UTC))

    def crawl(self) -> list[ProposedSource]:
        """Run the crawl. Returns proposals and persists the diff cache.

        Idempotent: re-running with the same on-disk cache returns *only*
        proposals for URLs whose content sha has changed (or new URLs).
        """
        cache_path = (
            self.repo_root
            / "data"
            / "cache"
            / "discovery"
            / f"{self.config.jurisdiction.lower()}.json"
        )
        cache = _DiffCache.load(cache_path)

        owned_client = False
        client = self.http_client
        if client is None:
            client = httpx.Client(
                timeout=self.http_timeout,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            )
            owned_client = True
        try:
            proposals = self._crawl_with(client, cache)
        finally:
            if owned_client:
                client.close()
        cache.persist()
        return proposals

    # ─── BFS core ───────────────────────────────────────────────────────────

    def _crawl_with(
        self, client: httpx.Client, cache: _DiffCache
    ) -> list[ProposedSource]:
        robots = _RobotsCache(client)
        seen: set[str] = set()
        # Per-host last-fetch timestamp for the crawl-delay budget.
        last_fetched: dict[str, float] = {}
        proposals: list[ProposedSource] = []
        queue: deque[tuple[str, int, str | None]] = deque()
        for seed in self.config.seed_urls:
            normalised = _normalise(seed)
            queue.append((normalised, 0, None))

        pages_fetched = 0
        while queue and pages_fetched < self.config.max_pages:
            url, depth, parent = queue.popleft()
            if url in seen:
                continue
            seen.add(url)

            if not robots.can_fetch(url):
                # Robots said no. Skip silently — no proposals, no recursion.
                continue

            self._respect_crawl_delay(url, last_fetched)
            try:
                resp = client.get(url, headers={"User-Agent": USER_AGENT})
            except httpx.HTTPError:
                continue
            last_fetched[_host(url)] = time.monotonic()
            pages_fetched += 1
            if resp.status_code != 200:
                continue

            content_type = resp.headers.get("content-type", "").lower() or None
            primary_mime = (
                content_type.split(";", 1)[0].strip() if content_type else None
            )
            body = resp.content
            sha = hashlib.sha256(body).hexdigest()

            mime_allowed = primary_mime is not None and primary_mime in self.config.patterns.mime_allow

            if mime_allowed and not cache.is_unchanged(url, sha):
                # New or changed: propose. The reviewer decides authority.
                proposals.append(
                    ProposedSource(
                        url=url,
                        jurisdiction=self.config.jurisdiction,
                        document_type=_guess_document_type(content_type),
                        retrieved_at=self.now_fn().isoformat(),
                        sha256=sha,
                        content_type=primary_mime,
                        title=_extract_title(body, primary_mime),
                        discovered_from=parent,
                    )
                )
                cache.update(url, sha)
            elif mime_allowed:
                # Same bytes as last run — refresh the cache anyway so the
                # next run still sees this URL as "known".
                cache.update(url, sha)

            # Only recurse from HTML pages. PDFs etc. are terminal.
            is_html = primary_mime in {"text/html", "application/xhtml+xml"}
            if not is_html or depth >= self.config.depth_limit:
                continue

            try:
                html = body.decode("utf-8", errors="replace")
            except (UnicodeDecodeError, LookupError):
                continue
            for link in _extract_links(html, base_url=url):
                norm = _normalise(link)
                if norm in seen:
                    continue
                # Stay on the seed's own host. Cross-host crawls are out of
                # scope for the polite discovery thin-mode.
                if not any(_same_host(norm, s) for s in self.config.seed_urls):
                    continue
                if not _matches_any(norm, self.config.patterns.link_regexes):
                    continue
                queue.append((norm, depth + 1, url))

        return proposals

    def _respect_crawl_delay(self, url: str, last_fetched: dict[str, float]) -> None:
        host = _host(url)
        prev = last_fetched.get(host)
        if prev is None:
            return
        elapsed = time.monotonic() - prev
        remaining = self.config.crawl_delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


_TITLE_RE: re.Pattern[str] = re.compile(
    r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL
)


def _extract_title(body: bytes, mime: str | None) -> str | None:
    if mime not in {"text/html", "application/xhtml+xml"}:
        return None
    try:
        text = body.decode("utf-8", errors="replace")
    except (UnicodeDecodeError, LookupError):
        return None
    m = _TITLE_RE.search(text)
    if not m:
        return None
    title = re.sub(r"\s+", " ", m.group(1)).strip()
    return title or None


# ─── Output ─────────────────────────────────────────────────────────────────


def write_proposals(
    repo_root: Path, jurisdiction: str, proposals: Iterable[ProposedSource], *, now: datetime | None = None
) -> Path:
    """Persist proposals to `data/outputs/discovery/<j>/<ts>.yaml`.

    Returns the path written. The timestamp is UTC, ISO-8601 compact form.
    """
    ts = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    out_dir = repo_root / "data" / "outputs" / "discovery" / jurisdiction.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ts}.yaml"
    payload: dict[str, Any] = {
        "jurisdiction": jurisdiction,
        "generated_at": (now or datetime.now(UTC)).isoformat(),
        "generator": USER_AGENT,
        "proposals": [asdict(p) for p in proposals],
    }
    out_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return out_path


# ─── CLI ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rie_ingest.discovery",
        description="Polite discovery crawler for jurisdiction portals (Phase 2).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    crawl = sub.add_parser("crawl", help="Crawl a single jurisdiction's discovery seeds.")
    crawl.add_argument("jurisdiction", help="Jurisdiction code, e.g. SAMPLE.")
    crawl.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (defaults to cwd).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command != "crawl":  # pragma: no cover — argparse enforces required=True
        parser.print_help()
        return 2
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    try:
        config = load_discovery_config(repo_root, args.jurisdiction)
    except DiscoveryConfigError as exc:
        print(f"discovery config error: {exc}", file=sys.stderr)
        return 1
    crawler = DiscoveryCrawler(repo_root=repo_root, config=config)
    proposals = crawler.crawl()
    out_path = write_proposals(repo_root, config.jurisdiction, proposals)
    print(f"wrote {len(proposals)} proposal(s) to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
