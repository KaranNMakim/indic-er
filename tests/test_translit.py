"""Script detection and transliteration."""

from __future__ import annotations

import pytest

from app.core.script_detect import detect_script, is_indic, profile_script
from app.core.translit import has_indic, to_latin, transliterate


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Jayesh Patel", "latin"),
        ("જયેશ પટેલ", "gujarati"),
        ("जयेश पटेल", "devanagari"),
        ("ராஜேஷ்", "tamil"),
        ("380008", "unknown"),
        ("", "unknown"),
    ],
)
def test_detect_script(text: str, expected: str) -> None:
    assert detect_script(text) == expected


def test_mixed_script_is_reported_as_indic_and_still_transliterates() -> None:
    """ "જયેશ Patel" has more Latin letters, but treating it as Latin would
    leave the Gujarati half uncomparable."""
    profile = profile_script("જયેશ Patel")
    assert profile.is_mixed
    script, latin = to_latin("જયેશ Patel")
    assert script == "gujarati"
    assert latin.lower() == "jayesh patel"


@pytest.mark.parametrize(
    ("indic", "latin"),
    [
        ("જયેશ પટેલ", "jayesh patel"),
        ("जयेश पटेल", "jayesh patel"),
        ("राम शर्मा", "ram sharma"),
        ("लक्ष्मी", "lakshmi"),
        ("संजय", "sanjay"),
        ("ભાવના શાહ", "bhavana shah"),
        ("ராஜேஷ்", "rajesh"),
    ],
)
def test_transliteration(indic: str, latin: str) -> None:
    assert transliterate(indic) == latin


def test_final_schwa_is_deleted_in_devanagari() -> None:
    """राम is "ram", not "rama" — Hindi deletes the word-final inherent vowel."""
    assert transliterate("राम") == "ram"


def test_virama_suppresses_the_inherent_vowel() -> None:
    """लक्ष्मी is a three-consonant cluster: l-a-ksh-m-i."""
    assert transliterate("लक्ष्मी") == "lakshmi"


def test_matra_replaces_the_inherent_vowel() -> None:
    assert transliterate("पटेल") == "patel"


def test_anusvara_becomes_n() -> None:
    assert transliterate("संजय") == "sanjay"


def test_latin_passes_through_untouched() -> None:
    script, latin = to_latin("Jayesh R. Patel")
    assert script == "latin"
    assert latin == "Jayesh R. Patel"


def test_latin_city_variants_are_folded() -> None:
    _, latin = to_latin("Amdavad")
    assert latin == "ahmedabad"
    _, latin = to_latin("Bombay 400028")
    assert latin.startswith("mumbai")


def test_indic_city_variant_folds_through_transliteration() -> None:
    """अमदावाद transliterates to "amadavad", which the variant list folds."""
    assert transliterate("अमदावाद") == "ahmedabad"


def test_has_indic_and_is_indic() -> None:
    assert has_indic("જયેશ Patel")
    assert not has_indic("Jayesh Patel")
    assert is_indic("જયેશ")
    assert not is_indic("Jayesh")


def test_empty_input() -> None:
    assert transliterate("") == ""
    assert to_latin("") == ("unknown", "")
