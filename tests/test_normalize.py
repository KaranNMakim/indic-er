"""Name, phone and record normalization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.models import InputRecord
from app.core.normalize import (
    is_common_surname,
    normalize_name,
    normalize_phone,
    normalize_record,
    normalize_records,
    phone_e164,
    strip_business_suffixes,
)

# --------------------------------------------------------------- honorifics #


@pytest.mark.parametrize(
    "name",
    [
        "Shri Jayesh Patel",
        "Jayeshbhai Patel",
        "Shri Jayeshbhai Patel",
        "Mr. Jayesh Patel",
        "જયેશભાઈ પટેલ",
        "Jayesh bhai Patel",
    ],
)
def test_honorifics_and_suffixes_converge_on_the_same_tokens(name: str) -> None:
    assert normalize_name(name).tokens == ("jayesh", "patel")


def test_what_was_stripped_is_kept() -> None:
    """An auditor must be able to see why a token disappeared."""
    result = normalize_name("Shri Jayeshbhai Patel")
    assert result.honorifics_removed == ("shri",)
    assert result.suffixes_removed == ("bhai",)


def test_female_suffix_does_not_eat_the_stem_vowel() -> None:
    """Kokilaben is Kokila + ben, not Kokil + aben."""
    assert normalize_name("Kokilaben Shah").tokens == ("kokila", "shah")
    assert normalize_name("કોકિલાબેન શાહ").tokens == ("kokila", "shah")


def test_suffix_exceptions_are_protected() -> None:
    """Juneja ends in "ja" but is not Jun + eja."""
    for name in ("Juneja", "Gowda", "Panda", "Ahuja"):
        assert normalize_name(name).tokens == (name.lower(),)


def test_short_stems_are_not_stripped() -> None:
    """Stripping "bai" from "Bai" would leave nothing."""
    assert normalize_name("Bhai").tokens == ("bhai",)


def test_name_that_is_only_honorifics_is_not_emptied() -> None:
    result = normalize_name("Shri Smt")
    assert result.tokens


def test_punctuation_initials_survive() -> None:
    assert normalize_name("J. R. Patel").tokens == ("j", "r", "patel")


def test_phonetic_key_ignores_honorifics() -> None:
    """The single most common failure in naive Indian-name matching."""
    assert (
        normalize_name("Jayeshbhai Patel").phonetic_key
        == normalize_name("Jayesh Patel").phonetic_key
    )


def test_display_form() -> None:
    assert normalize_name("જયેશ પટેલ").display == "Jayesh Patel"


# ----------------------------------------------------------------- business #


@pytest.mark.parametrize(
    ("raw", "tokens"),
    [
        ("Patel Trading Pvt. Ltd.", ("patel", "trading")),
        ("Patel Trading Company", ("patel", "trading")),
        ("Patel & Sons", ("patel",)),
        ("M/s Patel Enterprises", ("patel", "enterprises")),
        ("Patel Textiles LLP", ("patel", "textiles")),
    ],
)
def test_only_the_legal_wrapper_is_stripped(raw: str, tokens: tuple[str, ...]) -> None:
    """Legal forms go; sector words stay.

    "Pvt Ltd", "& Sons" and "LLP" say nothing about which business this is.
    "Trading" and "Textiles" do — stripping them merged Patel Trading with
    Patel Textiles, two different firms sharing a proprietor's surname, and
    reduced the canonical name of a resolved business to the bare word
    "Patel", which is the least identifying part of any Indian business name.
    """
    assert normalize_name(raw, kind="business").tokens == tokens


def test_two_sectors_under_one_surname_stay_apart() -> None:
    trading = normalize_name("Patel Trading Pvt Ltd", kind="business")
    textiles = normalize_name("Patel Textiles Pvt Ltd", kind="business")
    assert trading.tokens != textiles.tokens


def test_legal_suffix_is_captured_not_discarded() -> None:
    result = normalize_name("Patel Trading Pvt Ltd", kind="business")
    assert result.legal_suffix
    assert "pvt ltd" in result.legal_suffix


def test_business_suffix_only_name_is_not_emptied() -> None:
    cleaned, suffix = strip_business_suffixes("enterprises pvt ltd")
    assert cleaned
    assert suffix


def test_person_names_keep_business_words() -> None:
    """ "Trading" in a person's name is not a legal suffix."""
    assert normalize_name("Trading Patel", kind="person").tokens == ("trading", "patel")


# -------------------------------------------------------------------- phone #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+91 98250 12345", "9825012345"),
        ("098250-12345", "9825012345"),
        ("0091 9825012345", "9825012345"),
        ("9825012345", "9825012345"),
        ("+91-79-26580011", "7926580011"),
        ("98250 12345 ", "9825012345"),
    ],
)
def test_phone_normalization(raw: str, expected: str) -> None:
    assert normalize_phone(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "12345", "abc", "0000000000", "9999999999", "0"])
def test_unusable_phones_return_none(raw: str | None) -> None:
    """A junk phone must not become a matching signal."""
    assert normalize_phone(raw) is None


def test_phone_e164() -> None:
    assert phone_e164("9825012345") == "+919825012345"
    assert phone_e164(None) is None


# ------------------------------------------------------------------ records #


def test_normalize_record_end_to_end(jayesh_records: list[InputRecord]) -> None:
    record = normalize_record(jayesh_records[1])
    assert record.name.script == "gujarati"
    assert record.name.tokens == ("jayesh", "patel")
    assert record.address is not None
    assert record.address.city == "ahmedabad"
    assert record.phone_last10 == "9825012345"


def test_normalize_records_rejects_duplicate_ids() -> None:
    records = [InputRecord(id="a", name="One"), InputRecord(id="a", name="Two")]
    with pytest.raises(ValueError, match="duplicate record id"):
        normalize_records(records)


def test_missing_fields_are_none_not_empty(jayesh_records: list[InputRecord]) -> None:
    """A missing phone must be absent, never a falsy value that scores 0."""
    record = normalize_record(jayesh_records[2])
    assert record.phone_last10 is None


def test_unmasked_identifier_is_rejected() -> None:
    """indic-er refuses to hold a full identifier, even if asked."""
    with pytest.raises(ValidationError, match="masked"):
        InputRecord(id="a", name="Jayesh Patel", identifier_masked="ABCDE1234F")
    ok = InputRecord(id="a", name="Jayesh Patel", identifier_masked="XXXXX1234F")
    assert ok.identifier_masked == "XXXXX1234F"


def test_blank_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InputRecord(id="a", name="   ")


def test_is_common_surname() -> None:
    assert is_common_surname("Patel")
    assert is_common_surname("shah")
    assert not is_common_surname("Jayesh")
