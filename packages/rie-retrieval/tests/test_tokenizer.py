"""Tests for multilingual BM25 tokenizer."""

from __future__ import annotations

from rie_retrieval.tokenizer import SUPPORTED_LANGUAGES, is_supported_language, tokenize


def test_tokenize_english_stopwords_removed() -> None:
    toks = tokenize("The organisation shall not process personal data.", lang="en")
    assert "the" not in toks
    assert "shall" not in toks
    assert "organisation" in toks or "process" in toks


def test_tokenize_french_elision_split() -> None:
    toks = tokenize("l'État doit protéger les données personnelles", lang="fr")
    assert "etat" in toks or "protéger" in toks or "proteger" in toks
    assert "l" not in toks


def test_tokenize_cjk_bigram_fallback() -> None:
    toks = tokenize("个人信息保护", lang="zh")
    assert len(toks) >= 2


def test_tokenize_empty_returns_empty() -> None:
    assert tokenize("") == []


def test_supported_languages() -> None:
    assert "en" in SUPPORTED_LANGUAGES
    assert is_supported_language("fr")
    assert not is_supported_language("xx")
