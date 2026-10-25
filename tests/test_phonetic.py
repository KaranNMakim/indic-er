"""The phonetic key is the load-bearing piece of this project.

These are the equivalences the brief names explicitly, plus the separations
that stop the key from merging everything.
"""

from __future__ import annotations

import pytest

from app.core.phonetic import indic_key, phonetic_key, phonetic_tokens, token_phonetic_equal

EQUIVALENT_GROUPS = [
    pytest.param(["Patel", "Patil", "Pattel", "Pateel", "Pathel"], id="patel"),
    pytest.param(["Jayesh", "Jaishe", "Jaysh", "Jayes"], id="jayesh"),
    pytest.param(["Shah", "Sah", "Chah", "Shaha"], id="shah-sibilants"),
    pytest.param(["Lakshmi", "Laxmi", "Lakshmee"], id="lakshmi"),
    pytest.param(["Agarwal", "Aggarwal", "Agrawal"], id="agarwal"),
    pytest.param(["Krishna", "Krushna", "Krishnaa"], id="krishna"),
    pytest.param(["Mohammed", "Muhammad", "Mohamad"], id="mohammed"),
    pytest.param(["Reddy", "Reddi", "Redy"], id="reddy"),
    pytest.param(["Gupta", "Gupte"], id="gupta"),
    pytest.param(["Bhavna", "Bhavana", "Bhawna"], id="bhavna-v-w"),
    pytest.param(["Phone", "Fone"], id="ph-f"),
    pytest.param(["Zamindar", "Jamindar"], id="z-j"),
]


@pytest.mark.parametrize("variants", EQUIVALENT_GROUPS)
def test_variants_share_a_key(variants: list[str]) -> None:
    keys = {indic_key(v) for v in variants}
    assert len(keys) == 1, f"{variants} produced {keys}"
    assert keys != {""}


def test_documented_keys_are_stable() -> None:
    """Pin the actual values, so a config edit that changes them is visible."""
    assert indic_key("Patel") == "PTL"
    assert indic_key("Jayesh") == "JS"
    assert indic_key("Shah") == "S"
    assert indic_key("Lakshmi") == "LKSM"
    assert indic_key("Agarwal") == "AGRWL"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Patel", "Shah"),
        ("Jayesh", "Nilesh"),
        ("Kumar", "Kamal"),
        ("Yadav", "Jadav"),  # word-initial y is a consonant, not a vowel
        ("Hemant", "Emant"),  # word-initial h is kept
        ("Vora", "Bora"),  # b/v deliberately NOT folded — see config notes
    ],
)
def test_distinct_names_keep_distinct_keys(left: str, right: str) -> None:
    assert indic_key(left) != indic_key(right)


def test_initial_vowel_is_marked_not_dropped() -> None:
    """Agarwal must not collide with Garwal."""
    assert indic_key("Agarwal") != indic_key("Garwal")
    assert indic_key("Agarwal").startswith("A")


def test_whole_name_key_is_order_insensitive() -> None:
    assert phonetic_key(("jayesh", "patel")) == phonetic_key(("patel", "jayesh"))


def test_phonetic_tokens_drops_unusable_tokens() -> None:
    assert phonetic_tokens(("patel", "", "123", "shah")) == ("PTL", "S")


@pytest.mark.parametrize("token", ["", "   ", "123", "!!!"])
def test_unusable_input_yields_empty_key(token: str) -> None:
    assert indic_key(token) == ""


def test_every_initial_vowel_collapses_to_one_marker() -> None:
    """Which initial vowel it was does not survive, by design.

    Transliteration picks the vowel almost arbitrarily (Ishwar / Eshwar,
    Omkar / Umkar), so distinguishing them would split true variants.
    """
    assert indic_key("Aaee") == indic_key("Oo") == "A"
    assert indic_key("Ishwar") == indic_key("Eshwar")


def test_min_key_length_fallback() -> None:
    """Defensive branch: a config that marks nothing still yields a key."""
    from dataclasses import replace

    from app.core.config import phonetic_config
    from app.core.phonetic import _indic_key

    config = replace(phonetic_config(), initial_vowel_marker="")
    assert _indic_key("aaee", config) == "A"


def test_token_phonetic_equal() -> None:
    assert token_phonetic_equal("Patel", "Patil")
    assert not token_phonetic_equal("Patel", "Shah")
    assert not token_phonetic_equal("", "")


def test_sibilants_fold_which_soundex_never_does() -> None:
    assert indic_key("Shah") == indic_key("Chah") == indic_key("Sah")


def test_the_key_collides_on_purpose() -> None:
    """A documented, accepted cost of a recall-first blocking key.

    Jayesh and Joshi share a key. That makes them a candidate *pair*, which
    scoring then rejects — see ``test_features.py``. The failure that would
    actually matter is the opposite one, asserted above: two spellings of the
    same name landing in different buckets.
    """
    assert indic_key("Jayesh") == indic_key("Joshi")
    assert indic_key("Jayesh") != indic_key("Jayant")
