"""Address parsing.

The three addresses in ``SAME_PLACE`` are the project's motivating example:
the same flat written three ways, by three different source systems.
"""

from __future__ import annotations

import pytest

from app.core.address import parse_address

SAME_PLACE = [
    "12, Shreeji Soc., Nr. Swaminarayan Mandir, Maninagar, Ahmedabad 380008",
    "12 Shriji Society, opp Swaminarayan Temple, Maninagar, Amdavad, 380 008",
    "Flat 12 Shriji Soc, Maninagar, nr Swaminarayan temple, A'bad - 380008",
]


def test_none_in_none_out() -> None:
    assert parse_address(None) is None
    assert parse_address("") is None
    assert parse_address("   ") is None


@pytest.mark.parametrize("raw", SAME_PLACE)
def test_the_same_place_parses_consistently(raw: str) -> None:
    parsed = parse_address(raw)
    assert parsed is not None
    assert parsed.pincode == "380008"
    assert parsed.city == "ahmedabad"
    assert parsed.locality == "maninagar"
    assert parsed.landmark is not None
    assert "swaminarayan" in parsed.landmark


def test_core_tokens_agree_across_the_variants() -> None:
    """Order-insensitive comparison is the point of ``core_tokens``."""
    cores = [set(parse_address(raw).core_tokens) for raw in SAME_PLACE]
    shared = set.intersection(*cores)
    assert {"swaminarayan", "temple"} <= shared


def test_core_tokens_exclude_everything_generic() -> None:
    """What is left out is the point.

    `society` is a building *type* and appears in a third of all addresses;
    `maninagar` and `ahmedabad` are the locality and city, already extracted
    into their own fields and already scored by `pincode`. Leaving them in
    counted the same evidence twice and — worse — outvoted the one token that
    distinguishes two households, so two different Nilesh Pawars in Dadar West
    scored 0.93 on address.
    """
    core = set(parse_address(SAME_PLACE[0]).core_tokens)
    assert "shreeji" in core
    assert "society" not in core
    assert "maninagar" not in core
    assert "ahmedabad" not in core


@pytest.mark.parametrize(
    ("raw", "pincode"),
    [
        ("Maninagar, Ahmedabad 380008", "380008"),
        ("Maninagar, Ahmedabad 380 008", "380008"),
        ("Maninagar, Ahmedabad - 380-008", "380008"),
        ("380008, Maninagar", "380008"),
        ("Maninagar, Ahmedabad", None),
        ("Maninagar 012345", None),  # a pincode cannot start with 0
        ("Phone 9825012345, Maninagar", None),  # 10 digits is not a pincode
    ],
)
def test_pincode_extraction(raw: str, pincode: str | None) -> None:
    assert parse_address(raw).pincode == pincode


def test_abbreviations_are_expanded() -> None:
    parsed = parse_address("B-402, Sun Apt., Nr. Bus Stand, M.G. Rd, Surat 395003")
    assert "apartment" in parsed.expanded
    assert "road" in parsed.expanded
    assert parsed.house == "b-402"


@pytest.mark.parametrize(
    ("marker", "raw"),
    [
        ("near", "12 Shreeji Soc, nr Swaminarayan temple, Maninagar"),
        ("opposite", "12 Shreeji Soc, opp. Swaminarayan temple, Maninagar"),
        ("behind", "12 Shreeji Soc, behind Swaminarayan temple, Maninagar"),
        ("beside", "12 Shreeji Soc, beside Swaminarayan temple, Maninagar"),
    ],
)
def test_every_landmark_marker_captures_the_phrase(marker: str, raw: str) -> None:
    parsed = parse_address(raw)
    assert parsed.landmark == "swaminarayan temple"
    assert marker not in (parsed.landmark or "")


def test_mandir_and_temple_are_folded() -> None:
    a = parse_address("12 Shreeji Soc, nr Swaminarayan Mandir, Maninagar")
    b = parse_address("12 Shreeji Soc, nr Swaminarayan Temple, Maninagar")
    assert a.landmark == b.landmark


@pytest.mark.parametrize(
    ("raw", "city"),
    [
        ("Dadar West, Bombay 400028", "mumbai"),
        ("Dadar West, Mumbai 400028", "mumbai"),
        ("Jayanagar, Bangalore 560011", "bengaluru"),
        ("Jayanagar, Bengaluru 560011", "bengaluru"),
        ("T Nagar, Madras 600017", "chennai"),
        ("Maninagar, A'bad 380008", "ahmedabad"),
    ],
)
def test_city_aliases_fold(raw: str, city: str) -> None:
    assert parse_address(raw).city == city


def test_multiword_city_wins_over_its_substring() -> None:
    assert parse_address("Lajpat Nagar, New Delhi 110024").city == "delhi"


def test_state_is_recognised_from_its_code() -> None:
    assert parse_address("Sector 21, Gandhinagar, GJ 382021").state == "gujarat"
    assert parse_address("Sector 21, Gandhinagar, Gujarat 382021").state == "gujarat"


@pytest.mark.parametrize(
    ("raw", "house"),
    [
        ("12, Shreeji Soc", "12"),
        ("B-402, Sun Apt", "b-402"),
        ("15/A Krishna Nivas", "15/a"),
        ("Flat 12 Shriji Soc", "12"),
        ("Shreeji Society, Maninagar", None),
    ],
)
def test_house_number_extraction(raw: str, house: str | None) -> None:
    assert parse_address(raw).house == house


def test_building_drops_stopwords() -> None:
    """ "flat" appears in half of all Indian addresses and identifies nothing."""
    assert parse_address("Flat 12 Shriji Soc, Maninagar").building == "shriji society"


def test_locality_found_by_marker() -> None:
    parsed = parse_address("12 Sun Apt, Satellite Nagar, Ahmedabad 380015")
    assert parsed.locality == "satellite nagar"


def test_locality_found_by_position_when_no_marker() -> None:
    parsed = parse_address("12 Sun Apartment, Maninagar, Ahmedabad 380008")
    assert parsed.locality == "maninagar"


def test_district_prefix() -> None:
    assert parse_address("Maninagar, Ahmedabad 380008").district_prefix == "380"
    assert parse_address("Maninagar").district_prefix is None


def test_indic_script_address() -> None:
    parsed = parse_address("૧૨, શ્રીજી સોસાયટી, મણિનગર, અમદાવાદ")
    assert parsed is not None
    assert parsed.core_tokens


def test_is_empty_for_unparseable_junk() -> None:
    assert parse_address("...").is_empty


def test_to_dict_round_trip() -> None:
    payload = parse_address(SAME_PLACE[0]).to_dict()
    assert payload["pincode"] == "380008"
    assert isinstance(payload["core_tokens"], list)
