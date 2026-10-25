"""Blocking.

Blocking recall is a ceiling that scoring can never raise, so these tests are
mostly about pairs that MUST be proposed.
"""

from __future__ import annotations

from app.core.blocking import (
    build_blocks,
    candidate_pairs,
    describe_strategies,
    record_keys,
)
from app.core.config import weights_config
from app.core.models import InputRecord
from app.core.normalize import normalize_records


def norm(records: list[InputRecord]):
    return normalize_records(records)


def pair_ids(result) -> set[tuple[str, str]]:
    return set(result.pairs)


# Phonetic keys strip digits, so "Name1"/"Name2" would all collapse to one key
# and make a blocking test meaningless. These are genuinely distinct sounds.
_SYLLABLES_A = ["ja", "ni", "su", "ra", "ki", "ma", "pa", "ta", "va", "ha", "go", "bi"]
_SYLLABLES_B = [
    "yesh",
    "lesh",
    "resh",
    "mesh",
    "nish",
    "dip",
    "pak",
    "ram",
    "jit",
    "pal",
    "kesh",
    "veer",
    "nand",
    "bir",
    "sad",
]


def distinct_given_names(count: int) -> list[str]:
    names = [f"{a}{b}".title() for a in _SYLLABLES_A for b in _SYLLABLES_B]
    assert len(names) >= count, "widen the syllable lists"
    return names[:count]


def test_keys_cover_every_enabled_strategy() -> None:
    records = norm(
        [
            InputRecord(
                id="a",
                name="Jayesh Patel",
                address="12 Shreeji Soc, Maninagar, Ahmedabad 380008",
                phone="9825012345",
            )
        ]
    )
    keys = record_keys(records["a"], weights_config().blocking_strategies)
    prefixes = {key.split(":", 1)[0] for key in keys}
    assert {"ps", "pf", "pt", "pin", "tel"} <= prefixes


def test_single_letter_phonetic_keys_are_not_used() -> None:
    """An initial blocks against almost everything and is pure cost."""
    records = norm([InputRecord(id="a", name="J. R. Patel")])
    keys = record_keys(records["a"], weights_config().blocking_strategies)
    assert not any(key.startswith("pt:") and len(key) == 4 for key in keys)


def test_a_reordered_name_still_blocks_together() -> None:
    records = norm(
        [
            InputRecord(id="a", name="Jayesh Patel"),
            InputRecord(id="b", name="Patel Jayesh"),
        ]
    )
    assert ("a", "b") in pair_ids(candidate_pairs(records))


def test_a_transliterated_name_still_blocks_together() -> None:
    records = norm(
        [
            InputRecord(id="a", name="Jayeshbhai Patel"),
            InputRecord(id="b", name="જયેશ પટેલ"),
        ]
    )
    assert ("a", "b") in pair_ids(candidate_pairs(records))


def test_a_shared_phone_blocks_records_with_nothing_else_in_common() -> None:
    """The strategy that saves a pair when the name AND address are both wrong."""
    records = norm(
        [
            InputRecord(id="a", name="Jayesh Patel", phone="9825012345"),
            InputRecord(id="b", name="Murugan Pillai", phone="+91 98250 12345"),
        ]
    )
    result = candidate_pairs(records)
    assert ("a", "b") in pair_ids(result)
    assert "phone_last10" in describe_strategies(result.pairs[("a", "b")])


def test_a_shared_pincode_blocks_wildly_different_spellings() -> None:
    records = norm(
        [
            InputRecord(id="a", name="Jayesh Patel", address="Maninagar, Ahmedabad 380008"),
            InputRecord(id="b", name="Jaishe Pattel", address="Maninagar, Ahmedabad 380008"),
        ]
    )
    assert ("a", "b") in pair_ids(candidate_pairs(records))


def test_a_dropped_middle_name_still_blocks() -> None:
    records = norm(
        [
            InputRecord(id="a", name="Jayesh Ramesh Patel"),
            InputRecord(id="b", name="Jayesh Patel"),
        ]
    )
    assert ("a", "b") in pair_ids(candidate_pairs(records))


def test_unrelated_records_are_not_proposed() -> None:
    records = norm(
        [
            InputRecord(id="a", name="Jayesh Patel", address="Maninagar, Ahmedabad 380008"),
            InputRecord(id="b", name="Murugan Pillai", address="T Nagar, Chennai 600017"),
        ]
    )
    assert pair_ids(candidate_pairs(records)) == set()


def test_every_pair_records_which_strategies_proposed_it() -> None:
    records = norm(
        [
            InputRecord(
                id="a",
                name="Jayesh Patel",
                address="Maninagar, Ahmedabad 380008",
                phone="9825012345",
            ),
            InputRecord(
                id="b",
                name="Jayesh Patil",
                address="Maninagar, Ahmedabad 380008",
                phone="9825012345",
            ),
        ]
    )
    strategies = describe_strategies(candidate_pairs(records).pairs[("a", "b")])
    assert "pincode" in strategies
    assert "phone_last10" in strategies
    assert len(set(strategies)) == len(strategies)


def test_oversized_blocks_are_skipped_not_truncated() -> None:
    """Skipping a key is honest; truncating it drops specific records silently."""
    weights = weights_config()
    records = norm(
        [
            InputRecord(id=f"r{i}", name="Patel", address="Maninagar, Ahmedabad 380008")
            for i in range(weights.max_block_size + 5)
        ]
    )
    result = candidate_pairs(records, weights)
    assert result.skipped_blocks
    assert result.pair_count == 0


def test_records_in_an_oversized_block_remain_reachable_by_other_keys() -> None:
    """Skipping the pincode key must not strand a record.

    125 Patels share one pincode, so `pin:380008` is oversized and skipped.
    The record in a different pincode is still proposed against its namesake,
    because the phonetic keys are selective enough not to be skipped.
    """
    weights = weights_config()
    records = norm(
        [
            InputRecord(id=f"r{i}", name=f"{given} Patel", address="Maninagar, Ahmedabad 380008")
            for i, given in enumerate(distinct_given_names(weights.max_block_size + 5))
        ]
        + [InputRecord(id="twin", name="Jayesh Patel", address="Bopal, Ahmedabad 380058")]
    )
    result = candidate_pairs(records, weights)
    assert any("twin" in pair for pair in result.pairs)


def test_candidate_cap_sets_the_truncated_flag() -> None:
    weights = weights_config().with_overrides()
    object.__setattr__(weights, "max_candidate_pairs", 3)
    records = norm([InputRecord(id=f"r{i}", name="Jayesh Patel") for i in range(10)])
    result = candidate_pairs(records, weights)
    assert result.truncated
    assert result.pair_count <= 3


def test_blocking_stats_report_the_reduction() -> None:
    records = norm(
        [
            InputRecord(
                id=f"r{i}",
                name=f"{given} {surname}",
                address=f"Loc{i}, Pune 4110{i:02d}",
            )
            for i, (given, surname) in enumerate(
                zip(distinct_given_names(40), distinct_given_names(80)[40:], strict=True)
            )
        ]
    )
    result = candidate_pairs(records)
    stats = result.stats(len(records))
    assert stats["all_pairs"] == 780
    assert stats["candidate_pairs"] < stats["all_pairs"]
    assert stats["reduction_ratio"] > 0.9


def test_build_blocks_groups_by_key() -> None:
    records = norm(
        [InputRecord(id="a", name="Jayesh Patel"), InputRecord(id="b", name="Jaishe Patil")]
    )
    blocks = build_blocks(records)
    shared = [key for key, members in blocks.items() if len(members) == 2]
    assert shared


def test_describe_strategies_passes_unknown_codes_through() -> None:
    assert describe_strategies(("direct",)) == ("direct",)
