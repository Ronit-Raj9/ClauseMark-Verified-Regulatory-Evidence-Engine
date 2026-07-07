"""Cross-lingual retrieval helpers (Phase 2 — full multilingual parity).

BGE-M3 (and the fastembed-catalogued ``intfloat/multilingual-e5-large`` we
fall back to) embed many languages into one shared vector space, so a query in
language A can in principle match a clause in language B. Two cheap, pure
helpers lift that latent capability into the pipeline without changing default
behaviour:

  * :func:`detect_language` — a dependency-free script-range heuristic
    (Latin / CJK / Cyrillic / Arabic) returning an ISO-639-1 code, with an
    optional lazy ``langdetect`` fallback for finer Latin-script
    discrimination (en / fr / es). Never raises; returns ``"unknown"`` when it
    genuinely cannot tell.
  * :class:`LanguageAwareQueryExpander` — appends cross-lingual keyword
    variants drawn from a pillar indicator's multilingual ``positive_keywords``
    (``dict[str, list[str]]`` keyed by ISO-639-1). Appending the other-language
    surface forms to the query text lets the shared-space embedder — and the
    BM25 sparse leg — surface clauses written in a language the query was not.

Both are pure functions / pure data: trivially unit-testable, no model load,
no I/O.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

__all__ = [
    "LanguageAwareQueryExpander",
    "detect_language",
]


# ── language detection ─────────────────────────────────────────────────────
# We classify by dominant unicode script. This is intentionally coarse: it is
# enough to route the sparse tokenizer (CJK bigram vs. Latin word split) and to
# pick which *other* languages to expand into. Distinguishing en/fr/es — all
# Latin — needs lexical signal, so we optionally defer to ``langdetect``.

_LATIN = "latin"
_CJK = "cjk"
_CYRILLIC = "cyrillic"
_ARABIC = "arabic"

# ISO-639-1 returned for each non-Latin script (1:1 — these scripts map cleanly
# to a representative language for keyword-expansion routing).
_SCRIPT_TO_LANG: dict[str, str] = {
    _CJK: "zh",
    _CYRILLIC: "ru",
    _ARABIC: "ar",
}


def _script_of(ch: str) -> str | None:
    """Map a single character to a coarse script bucket, or ``None`` if it is
    punctuation / digit / whitespace (carries no language signal)."""
    code = ord(ch)
    if (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Ext A
        or 0x3040 <= code <= 0x309F  # Hiragana
        or 0x30A0 <= code <= 0x30FF  # Katakana
        or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
    ):
        return _CJK
    if 0x0400 <= code <= 0x04FF or 0x0500 <= code <= 0x052F:  # Cyrillic (+ supp.)
        return _CYRILLIC
    if (
        0x0600 <= code <= 0x06FF  # Arabic
        or 0x0750 <= code <= 0x077F  # Arabic Supplement
        or 0x08A0 <= code <= 0x08FF  # Arabic Extended-A
    ):
        return _ARABIC
    if unicodedata.category(ch).startswith("L"):
        # Any other letter — treat as Latin for our purposes (the Latin-script
        # legal corpora here are en/fr/es). Non-Latin letters outside the
        # handled scripts are rare in this corpus and degrade gracefully.
        return _LATIN
    return None


def detect_language(text: str) -> str:
    """Best-effort ISO-639-1 language guess for ``text``.

    Returns one of ``"en"`` / ``"fr"`` / ``"es"`` / ``"zh"`` / ``"ru"`` /
    ``"ar"`` / ``"unknown"``. Pure heuristic by default; for Latin-script text
    it lazily tries ``langdetect`` (if installed) to separate en/fr/es and
    otherwise defaults to ``"en"`` (the corpus's majority language). Never
    raises.
    """
    if not text or not text.strip():
        return "unknown"

    counts: dict[str, int] = {}
    for ch in text:
        bucket = _script_of(ch)
        if bucket is not None:
            counts[bucket] = counts.get(bucket, 0) + 1

    if not counts:
        return "unknown"

    dominant = max(counts, key=lambda k: counts[k])

    if dominant != _LATIN:
        return _SCRIPT_TO_LANG[dominant]

    # Latin script: try langdetect for en/fr/es discrimination, else default en.
    return _detect_latin_language(text)


def _detect_latin_language(text: str) -> str:
    """Resolve a Latin-script string to en/fr/es via optional ``langdetect``.

    The dependency is optional and lazily imported. If it is absent, raises,
    or returns a code outside our handled Latin set, we fall back to ``"en"``.
    """
    try:
        from langdetect import DetectorFactory  # noqa: PLC0415
        from langdetect import detect as _ld_detect  # noqa: PLC0415 — lazy/optional

        DetectorFactory.seed = 0  # deterministic output across runs
        code = _ld_detect(text)
    except Exception:
        return "en"

    code = (code or "").lower()[:2]
    if code in {"en", "fr", "es"}:
        return code
    return "en"


# ── cross-lingual query expansion ──────────────────────────────────────────


class LanguageAwareQueryExpander:
    """Append cross-lingual keyword variants to a query.

    Given a query and a pillar indicator's multilingual ``positive_keywords``
    (``{"en": [...], "fr": [...], ...}``), :meth:`expand` appends the keyword
    surface forms from *every* registered language so the shared-space embedder
    and the BM25 sparse leg can both surface clauses written in a language the
    query was not. The original query text is preserved verbatim at the front;
    expansion only ever *adds* tokens, never rewrites — keeping behaviour
    monotone and the result deterministic.
    """

    def __init__(self, max_keywords_per_lang: int | None = None) -> None:
        # Optional cap so an indicator with a long keyword list does not swamp
        # the original query signal. ``None`` => append all.
        self._max_per_lang = max_keywords_per_lang

    def expand(
        self,
        query: str,
        keywords_by_lang: dict[str, list[str]],
    ) -> str:
        """Return ``query`` with deduplicated cross-lingual keywords appended.

        Pure function. Languages are processed in sorted order and keywords in
        listed order so the output is stable. Keywords already present
        (case-insensitively) in the query — or duplicated across languages —
        are emitted once.
        """
        base = query.strip()
        if not keywords_by_lang:
            return base

        seen: set[str] = set()
        if base:
            seen.update(_normalise(t) for t in base.split())

        additions: list[str] = []
        for lang in sorted(keywords_by_lang):
            kws = keywords_by_lang[lang]
            picked = kws if self._max_per_lang is None else kws[: self._max_per_lang]
            for kw in _dedupe(picked, seen):
                additions.append(kw)

        if not additions:
            return base
        if not base:
            return " ".join(additions)
        return f"{base} {' '.join(additions)}"


def _normalise(token: str) -> str:
    return unicodedata.normalize("NFKC", token).strip().lower()


def _dedupe(keywords: Iterable[str], seen: set[str]) -> list[str]:
    """Yield keywords not already in ``seen`` (by normalised key), updating it."""
    out: list[str] = []
    for kw in keywords:
        stripped = kw.strip()
        if not stripped:
            continue
        key = _normalise(stripped)
        if key in seen:
            continue
        seen.add(key)
        out.append(stripped)
    return out
