"""Unit tests for `rie_retrieval.multilingual` — pure helpers, no model load."""

from __future__ import annotations

import pytest
from rie_retrieval.multilingual import LanguageAwareQueryExpander, detect_language

# ── detect_language ────────────────────────────────────────────────────────


def test_detect_chinese() -> None:
    assert detect_language("个人数据不得传输至境外") == "zh"


def test_detect_japanese_is_cjk() -> None:
    # Hiragana / Katakana fall in the CJK bucket -> "zh" representative.
    assert detect_language("これは個人データです") == "zh"


def test_detect_cyrillic() -> None:
    assert detect_language("Передача персональных данных запрещена") == "ru"


def test_detect_arabic() -> None:
    assert detect_language("لا يجوز نقل البيانات الشخصية") == "ar"


def test_detect_latin_defaults_or_langdetect() -> None:
    # Latin-script always resolves to one of en/fr/es (langdetect if present,
    # else the "en" default). Never "unknown" for real prose.
    out = detect_language("An organisation shall not transfer personal data outside the country.")
    assert out in {"en", "fr", "es"}


def test_detect_empty_is_unknown() -> None:
    assert detect_language("") == "unknown"
    assert detect_language("   ") == "unknown"


def test_detect_digits_punct_only_is_unknown() -> None:
    assert detect_language("123 -- !!! 456") == "unknown"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("跨境数据传输", "zh"),
        ("Кириллица", "ru"),
        ("بيانات", "ar"),
    ],
)
def test_detect_script_table(text: str, expected: str) -> None:
    assert detect_language(text) == expected


# ── LanguageAwareQueryExpander ─────────────────────────────────────────────


def test_expand_appends_other_language_keywords() -> None:
    expander = LanguageAwareQueryExpander()
    kbl = {
        "en": ["shall not transfer"],
        "fr": ["ne doit pas transférer"],
        "es": ["no podrá transferir"],
        "zh": ["不得传输"],
    }
    out = expander.expand("outbound data transfer prohibition", kbl)
    # Original preserved verbatim at the front.
    assert out.startswith("outbound data transfer prohibition")
    # Cross-lingual variants appended.
    assert "ne doit pas transférer" in out
    assert "no podrá transferir" in out
    assert "不得传输" in out
    assert "shall not transfer" in out


def test_expand_no_keywords_is_identity() -> None:
    expander = LanguageAwareQueryExpander()
    assert expander.expand("a query", {}) == "a query"


def test_expand_dedupes_against_query_and_across_langs() -> None:
    expander = LanguageAwareQueryExpander()
    kbl = {"en": ["transfer", "transfer"], "fr": ["transfer"]}
    out = expander.expand("transfer rules", kbl)
    # "transfer" already in the query (case-insensitive) -> not re-appended.
    assert out == "transfer rules"


def test_expand_is_deterministic_sorted_langs() -> None:
    expander = LanguageAwareQueryExpander()
    kbl = {"zh": ["不得传输"], "fr": ["interdit"], "en": ["banned"]}
    out1 = expander.expand("q", kbl)
    out2 = expander.expand("q", kbl)
    assert out1 == out2
    # Sorted by lang key: en, fr, zh.
    assert out1 == "q banned interdit 不得传输"


def test_expand_respects_max_per_lang() -> None:
    expander = LanguageAwareQueryExpander(max_keywords_per_lang=1)
    kbl = {"en": ["one", "two", "three"]}
    out = expander.expand("q", kbl)
    assert out == "q one"


def test_expand_empty_query_just_keywords() -> None:
    expander = LanguageAwareQueryExpander()
    out = expander.expand("", {"en": ["alpha"], "fr": ["beta"]})
    assert out == "alpha beta"
