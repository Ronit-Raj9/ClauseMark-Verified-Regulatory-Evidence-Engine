"""Multilingual unicode-aware tokenizer + per-language stopword lists.

Used by the sparse (BM25-style) leg of hybrid retrieval. The dense leg
(BGE-M3) is already multilingual; the sparse leg historically assumed English
whitespace/punct splitting, which destroys recall on French elisions
("l'État"), Spanish accents ("protección"), German compounds, CJK scripts,
and RTL Arabic.

Design choices kept small on purpose (no `nltk` / `spacy` dep — those are
multi-MB downloads and not portable to CPU-only test runners):

  * `unicodedata.normalize("NFKC", ...)` to fold compatibility variants
    (full-width ASCII in CJK corpora, ligatures, etc.) into canonical form.
  * Whitespace + unicode punctuation split (using the unicode category, not
    a hand-rolled regex of ASCII punct only).
  * Elision split for FR-style ``l'`` / ``d'`` / ``qu'`` / ``n'`` / ``j'`` /
    ``m'`` / ``s'`` / ``t'`` / ``c'`` — the apostrophe is treated as a word
    boundary AFTER lower-casing, so "l'État" -> ["l", "etat"-ish] (accents
    preserved, only the elision split is forced).
  * CJK fallback: if a "token" (whitespace-delimited run) contains no Latin
    characters and is longer than 1 char of CJK script, fall back to
    character bigrams. This is the standard cheap trick when you don't
    have a real CJK segmenter — it keeps BM25 recall sane without pulling
    in jieba / sudachi.
  * Stopword lists are tiny inline sets per language. Removed AFTER tokenization,
    so they apply to the post-elision-split form (e.g. ``l`` from ``l'État``
    is dropped as a FR stopword).

Public API is intentionally narrow:

    tokenize(text, lang=None) -> list[str]
    is_supported_language(lang) -> bool
    SUPPORTED_LANGUAGES -> frozenset[str]
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

__all__ = [
    "SUPPORTED_LANGUAGES",
    "is_supported_language",
    "tokenize",
]


# ── stopwords ────────────────────────────────────────────────────────────────
# Tiny, hand-picked, lowercase, NFKC-folded. Source: standard high-frequency
# closed-class function words. Kept small (≤ ~50 per language) — BM25 already
# down-weights frequent terms; the goal here is to drop the worst offenders.

_STOPWORDS_EN: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "he", "in", "is", "it", "its", "of", "on", "or",
        "that", "the", "to", "was", "were", "will", "with", "this", "these",
        "those", "but", "not", "any", "all", "shall", "may", "if", "than",
    }
)  # fmt: skip

_STOPWORDS_FR: frozenset[str] = frozenset(
    {
        "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou",
        "que", "qui", "dans", "pour", "par", "sur", "avec", "sans", "au",
        "aux", "ce", "ces", "cet", "cette", "il", "elle", "ils", "elles",
        "est", "sont", "ne", "pas", "se", "son", "sa", "ses",
        # post-elision single letters that remain after l'/d'/qu'/n'/j'/m'/s'/t'/c'
        "l", "d", "qu", "n", "j", "m", "s", "t", "c",
    }
)  # fmt: skip

_STOPWORDS_ES: frozenset[str] = frozenset(
    {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
        "y", "o", "que", "en", "para", "por", "con", "sin", "al", "lo",
        "es", "son", "se", "su", "sus", "no", "si", "este", "esta", "estos",
        "estas", "ese", "esa", "esos", "esas",
    }
)  # fmt: skip

_STOPWORDS_DE: frozenset[str] = frozenset(
    {
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
        "eines", "einem", "einen", "und", "oder", "in", "im", "an", "am",
        "auf", "mit", "ohne", "von", "vom", "zu", "zur", "zum", "für",
        "ist", "sind", "war", "waren", "nicht", "auch", "wie", "als", "so",
    }
)  # fmt: skip

# Chinese — characters function as both content + function words; keep only
# obvious particles / connectives. CJK tokenization fallback is bigram, so
# stopwords here apply to character-pair tokens that we never want indexed.
_STOPWORDS_ZH: frozenset[str] = frozenset(
    {
        "的", "了", "和", "是", "在", "也", "都", "及", "或", "与",
        "为", "对", "从", "向", "由", "把", "被",
    }
)  # fmt: skip

# Arabic — common conjunctions / prepositions / pronouns / particles.
_STOPWORDS_AR: frozenset[str] = frozenset(
    {
        "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
        "هو", "هي", "هم", "أن", "إن", "كان", "كانت", "لا", "ما", "و",
        "أو", "ثم", "كل", "بعض", "غير",
    }
)  # fmt: skip

_STOPWORDS: dict[str, frozenset[str]] = {
    "en": _STOPWORDS_EN,
    "fr": _STOPWORDS_FR,
    "es": _STOPWORDS_ES,
    "de": _STOPWORDS_DE,
    "zh": _STOPWORDS_ZH,
    "ar": _STOPWORDS_AR,
}

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(_STOPWORDS.keys())


# ── elision (FR / IT-style apostrophe contractions) ───────────────────────────
# Split on both straight ' (U+0027) and typographic right-single-quote ’ (U+2019).
# After NFKC + lower-casing, ``l'État`` -> ``l'etat`` (NFKC keeps the apostrophe
# but folds nothing visible to ASCII). The regex below then splits it into
# ``l`` and ``etat``.
_APOSTROPHE_CHARS = "'’ʼ"
_ELISION_SPLIT_RE = re.compile(rf"[{re.escape(_APOSTROPHE_CHARS)}]")


# Match a maximal run of "word" characters: anything that's NOT whitespace,
# punctuation, or a symbol per the unicode category system. Implemented as a
# function-driven splitter (not a single regex) because Python's ``re`` does
# not expose unicode general categories.
def _is_word_char(ch: str) -> bool:
    cat = unicodedata.category(ch)
    # Letters (L*), numbers (N*), and connector punctuation (Pc, e.g. underscore).
    return cat.startswith(("L", "N")) or cat == "Pc"


def _split_unicode_words(text: str) -> list[str]:
    """Split on any non-word unicode character. Empty pieces are dropped."""
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        if _is_word_char(ch):
            buf.append(ch)
        elif buf:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


# ── CJK detection + bigram fallback ──────────────────────────────────────────
# Conservative CJK unicode ranges. Sufficient for "is there CJK in this token"
# triage; not a full Unihan check.
def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF   # CJK Unified Ideographs Extension A
        or 0x3040 <= code <= 0x309F   # Hiragana
        or 0x30A0 <= code <= 0x30FF   # Katakana
        or 0xAC00 <= code <= 0xD7AF   # Hangul Syllables
    )


def _has_cjk(s: str) -> bool:
    return any(_is_cjk_char(ch) for ch in s)


def _bigrams(s: str) -> list[str]:
    if len(s) <= 1:
        return [s] if s else []
    return [s[i : i + 2] for i in range(len(s) - 1)]


def _expand_cjk_runs(tokens: Iterable[str]) -> list[str]:
    """For each token: if it's a CJK run (no Latin letters, has CJK), emit its bigrams.

    A token mixing CJK + Latin (rare; usually proper nouns) is kept whole AND
    its CJK portion is also bigrammed — gives BM25 a shot from either side.
    """
    out: list[str] = []
    for tok in tokens:
        if not _has_cjk(tok):
            out.append(tok)
            continue
        latin = "".join(ch for ch in tok if "a" <= ch.lower() <= "z")
        cjk_only = "".join(ch for ch in tok if _is_cjk_char(ch))
        if latin:
            out.append(tok)  # preserve mixed-script form
        if cjk_only:
            out.extend(_bigrams(cjk_only))
    return out


# ── public entry point ───────────────────────────────────────────────────────


def is_supported_language(lang: str | None) -> bool:
    """True iff `lang` has an explicit stopword list registered."""
    return lang is not None and lang.lower() in _STOPWORDS


def tokenize(text: str, lang: str | None = None) -> list[str]:
    """Tokenize `text` for the BM25 sparse leg.

    Pipeline:
      1. NFKC-normalize (fold compatibility variants).
      2. Lower-case (cheap unicode lowering; preserves accents).
      3. Split on apostrophes (FR-style elision: ``l'État`` -> ``l état``).
      4. Whitespace + unicode-punct split.
      5. CJK runs -> character bigrams (no-space-language fallback).
      6. Drop tokens that are pure-punct or empty.
      7. Drop language-specific stopwords (if `lang` is recognised).

    `lang` may be ``None`` (fall back to English stopwords as the safe default —
    legal corpora in this repo are >70% English even in non-English jurisdictions
    because international treaties are bilingual).
    """
    if not text:
        return []

    normalized = unicodedata.normalize("NFKC", text).lower()
    de_elided = _ELISION_SPLIT_RE.sub(" ", normalized)
    raw_tokens = _split_unicode_words(de_elided)
    expanded = _expand_cjk_runs(raw_tokens)

    lang_key = (lang or "").lower()
    stopwords = _STOPWORDS.get(lang_key, _STOPWORDS_EN)

    cleaned: list[str] = []
    for tok in expanded:
        if not tok:
            continue
        if tok in stopwords:
            continue
        cleaned.append(tok)
    return cleaned
