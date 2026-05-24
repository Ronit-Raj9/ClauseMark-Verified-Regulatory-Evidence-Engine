"""Tests for clause language detection."""

from __future__ import annotations

from rie_classify.language import DEFAULT_LANGUAGE, detect_language


def test_detect_language_empty_defaults_to_en() -> None:
    assert detect_language("") == DEFAULT_LANGUAGE
    assert detect_language("   ") == DEFAULT_LANGUAGE


def test_detect_language_short_text_defaults_to_en() -> None:
    assert detect_language("Article 5") == DEFAULT_LANGUAGE


def test_detect_language_french_stopwords() -> None:
    text = (
        "Le responsable du traitement doit obtenir le consentement de la personne "
        "concernée pour le traitement des données personnelles."
    )
    assert detect_language(text) == "fr"


def test_detect_language_spanish_stopwords() -> None:
    text = (
        "El responsable del tratamiento debe obtener el consentimiento del interesado "
        "para el tratamiento de datos personales en cualquier caso."
    )
    assert detect_language(text) == "es"


def test_detect_language_english_prose() -> None:
    text = (
        "An organisation shall not process personal data unless the data subject "
        "has given consent or another lawful basis applies under this section."
    )
    assert detect_language(text) == "en"
