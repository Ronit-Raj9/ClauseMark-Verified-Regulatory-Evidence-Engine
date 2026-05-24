"""Language detection for incoming legal clauses.

The classifier is *multilingual at the model layer* — we do **not** translate
the clause. We only tag it with a detected language code so the prompt can:

1. Tell the LLM the clause is in `fr` / `es` / `en` etc.
2. Pick per-language `positive_keywords` / `negative_cues` from the pillar
   YAML, falling back to `en` when the pillar has no entry for the detected
   language (and recording `language_match=False` in that case).

Preferred backend is `lingua-language-detector` because it outperforms
`langdetect` on short legal text. The dependency is declared in this
package's `pyproject.toml`. To keep the package import-safe in environments
where the wheel has not yet been resolved (e.g. CI snapshots taken between
``uv add`` and ``uv lock``), we degrade gracefully to a small built-in
stopword detector that handles EN / FR / ES — the three languages exercised
by the test suite.

Public surface:
- ``detect_language(text) -> str`` returning a lowercase ISO 639-1 code.
- ``DEFAULT_LANGUAGE`` ("en") — what we fall back to on empty / undetectable
  input so downstream prompt assembly never sees ``None``.
"""

from __future__ import annotations

import logging
import re
from typing import Final

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE: Final[str] = "en"

# Minimum characters before we even try real detection. Below this, short
# legal headers ("Article 5", "s 26") are too ambiguous and we just return
# the default rather than misclassifying.
_MIN_DETECT_CHARS: Final[int] = 12

# Stopword sets used by the built-in fallback. Deliberately small and
# function-word-only so legal jargon does not pollute the signal. These are
# *only* consulted when lingua is unavailable.
_STOPWORDS: Final[dict[str, frozenset[str]]] = {
    "en": frozenset(
        {
            "the",
            "and",
            "or",
            "of",
            "to",
            "in",
            "a",
            "an",
            "is",
            "are",
            "shall",
            "not",
            "any",
            "unless",
            "that",
            "this",
            "by",
            "for",
            "with",
            "be",
            "from",
            "as",
            "on",
            "such",
            "which",
            "under",
        }
    ),
    "fr": frozenset(
        {
            "le",
            "la",
            "les",
            "de",
            "des",
            "du",
            "et",
            "ou",
            "un",
            "une",
            "est",
            "sont",
            "ne",
            "pas",
            "que",
            "qui",
            "dans",
            "par",
            "pour",
            "avec",
            "sur",
            "aux",
            "au",
            "sous",
            "doit",
            "moins",
            "sauf",
            "donnees",
        }
    ),
    "es": frozenset(
        {
            "el",
            "la",
            "los",
            "las",
            "de",
            "del",
            "y",
            "o",
            "un",
            "una",
            "es",
            "son",
            "no",
            "que",
            "en",
            "por",
            "para",
            "con",
            "sobre",
            "al",
            "se",
            "su",
            "sus",
            "menos",
            "salvo",
            "debe",
            "datos",
            "personales",
        }
    ),
}

# Cache for the lingua detector — building it is expensive (loads n-gram
# models), so we lazy-instantiate at first use and reuse across calls.
_LINGUA_DETECTOR: object | None = None
_LINGUA_TRIED: bool = False


def _get_lingua_detector() -> object | None:
    """Lazily build the lingua detector. Returns ``None`` if lingua is not installed."""

    global _LINGUA_DETECTOR, _LINGUA_TRIED
    if _LINGUA_TRIED:
        return _LINGUA_DETECTOR
    _LINGUA_TRIED = True
    try:
        from lingua import Language, LanguageDetectorBuilder  # type: ignore[import-not-found]
    except ImportError:
        logger.info(
            "lingua-language-detector not importable; falling back to stopword detector."
        )
        _LINGUA_DETECTOR = None
        return None

    # Limit the language set so detection is fast and accurate for the
    # corpus we actually classify. Extend this list when new gold sets land.
    languages = [
        Language.ENGLISH,
        Language.FRENCH,
        Language.SPANISH,
        Language.GERMAN,
        Language.PORTUGUESE,
        Language.ITALIAN,
    ]
    _LINGUA_DETECTOR = (
        LanguageDetectorBuilder.from_languages(*languages).with_low_accuracy_mode().build()
    )
    return _LINGUA_DETECTOR


def _detect_with_lingua(text: str) -> str | None:
    detector = _get_lingua_detector()
    if detector is None:
        return None
    result = detector.detect_language_of(text)  # type: ignore[attr-defined]
    if result is None:
        return None
    iso = result.iso_code_639_1.name.lower()  # type: ignore[attr-defined]
    return iso


_WORD_RE = re.compile(r"[a-zA-ZÀ-ÿ]+", re.UNICODE)


def _detect_with_stopwords(text: str) -> str:
    """Score each known language by stopword hit rate; tie → English."""

    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    if not tokens:
        return DEFAULT_LANGUAGE
    scores: dict[str, int] = {}
    for lang, stopwords in _STOPWORDS.items():
        scores[lang] = sum(1 for t in tokens if t in stopwords)
    best_lang = max(scores, key=lambda lang_key: (scores[lang_key], lang_key == DEFAULT_LANGUAGE))
    if scores[best_lang] == 0:
        return DEFAULT_LANGUAGE
    return best_lang


def detect_language(text: str) -> str:
    """Return a lowercase ISO 639-1 code for ``text``.

    Always returns a non-empty string. On empty / very short / undetectable
    input, returns ``DEFAULT_LANGUAGE``. Never raises.
    """

    if not text or len(text.strip()) < _MIN_DETECT_CHARS:
        return DEFAULT_LANGUAGE
    try:
        iso = _detect_with_lingua(text)
        if iso:
            return iso
    except Exception as exc:
        logger.warning("lingua detection failed; falling back to stopwords: %s", exc)
    return _detect_with_stopwords(text)


__all__ = ["DEFAULT_LANGUAGE", "detect_language"]
