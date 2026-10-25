"""Pairwise features and pair scoring."""

from __future__ import annotations

import pytest

from app.core.config import weights_config
from app.core.features import (
    COMMON_ONLY_RARITY,
    _phonetic_token_score,
    _string_token_score,
    align_tokens,
    combine,
    compute_features,
    decide,
    f_address_core,
    f_landmark,
    f_name_phonetic,
    f_name_rarity,
    f_name_string,
    f_phone,
    f_pincode,
    score_pair,
)
from app.core.models import FeatureValue, InputRecord
from app.core.normalize import normalize_record


def rec(record_id: str, name: str, address: str | None = None, phone: str | None = None):
    return normalize_record(InputRecord(id=record_id, name=name, address=address, phone=phone))


# ------------------------------------------------------------- token scorers #


def test_string_token_score() -> None:
    assert _string_token_score("patel", "patel") == 1.0
    assert 0.8 < _string_token_score("patel", "patil") < 1.0
    assert _string_token_score("", "patel") == 0.0


def test_an_initial_matches_a_full_token_but_not_perfectly() -> None:
    assert _string_token_score("j", "jayesh") == pytest.approx(0.90)
    assert _string_token_score("j", "nilesh") == 0.0


def test_phonetic_token_score_gives_partial_credit() -> None:
    assert _phonetic_token_score("patel", "patil") == 1.0
    partial = _phonetic_token_score("subramanian", "subramaniam")
    assert 0.8 < partial < 1.0
    assert _phonetic_token_score("j", "jayesh") == pytest.approx(0.90)


# ---------------------------------------------------------------- alignment #


def test_alignment_is_order_insensitive() -> None:
    forward = align_tokens(("jayesh", "patel"), ("patel", "jayesh"), _string_token_score)
    assert forward.score == pytest.approx(1.0)


def test_an_extra_middle_initial_is_not_held_against_a_pair() -> None:
    """ "Jayesh R. Patel" and "Jayesh Patel" are the same name."""
    alignment = align_tokens(("jayesh", "r", "patel"), ("jayesh", "patel"), _string_token_score)
    assert alignment.score == pytest.approx(1.0)


def test_a_much_longer_name_is_penalised() -> None:
    """ "Patel" alone tells us less than "Jayesh Kumar Patel" does."""
    alignment = align_tokens(("patel",), ("jayesh", "kumar", "patel"), _string_token_score)
    assert alignment.score < 0.8


def test_alignment_of_empty_input() -> None:
    assert align_tokens((), ("patel",), _string_token_score).score == 0.0
    assert align_tokens(("patel",), (), _string_token_score).score == 0.0


def test_matched_tokens_only_reports_close_matches() -> None:
    alignment = align_tokens(("jayesh", "patel"), ("jayesh", "shah"), _string_token_score)
    assert alignment.matched == (("jayesh", "jayesh"),)


# ----------------------------------------------------------------- features #


def test_name_features_on_the_motivating_example() -> None:
    left = rec("a", "Jayeshbhai Patel")
    right = rec("b", "જયેશ પટેલ")
    assert f_name_string(left, right).value == pytest.approx(1.0)
    assert f_name_phonetic(left, right).value == pytest.approx(1.0)


def test_rarity_is_high_when_a_distinctive_token_agrees() -> None:
    feature = f_name_rarity(rec("a", "Jayesh Patel"), rec("b", "Jayesh Patil"))
    assert feature.value == 1.0
    assert "jayesh" in feature.detail


def test_rarity_is_low_when_only_a_common_surname_agrees() -> None:
    """The feature that stops a Patel block collapsing into one entity."""
    feature = f_name_rarity(rec("a", "Jayesh Patel"), rec("b", "Nilesh Patel"))
    assert feature.value == COMMON_ONLY_RARITY
    assert "common surname" in feature.detail


def test_rarity_is_none_when_no_token_agrees() -> None:
    feature = f_name_rarity(rec("a", "Jayesh Patel"), rec("b", "Murugan Pillai"))
    assert feature.value is None


def test_pincode_exact_district_and_different() -> None:
    a = rec("a", "X", "Maninagar, Ahmedabad 380008")
    b = rec("b", "Y", "Maninagar, Ahmedabad 380008")
    c = rec("c", "Z", "Bopal, Ahmedabad 380058")
    d = rec("d", "W", "Sector 12, Gandhinagar 382012")
    assert f_pincode(a, b).value == 1.0
    assert f_pincode(a, c).value == 0.5
    assert f_pincode(a, d).value == 0.0


def test_phone_is_binary() -> None:
    a = rec("a", "X", phone="+91 98250 12345")
    b = rec("b", "Y", phone="098250-12345")
    c = rec("c", "Z", phone="9900011122")
    assert f_phone(a, b).value == 1.0
    assert f_phone(a, c).value == 0.0


def test_landmark_agreement_survives_abbreviation_and_synonym() -> None:
    a = rec("a", "X", "12 Shreeji Soc, Nr. Swaminarayan Mandir, Maninagar")
    b = rec("b", "Y", "12 Shreeji Society, opp Swaminarayan Temple, Maninagar")
    assert f_landmark(a, b).value == pytest.approx(1.0)


def test_address_core_agrees_across_written_variants() -> None:
    a = rec("a", "X", "12, Shreeji Soc., Nr. Swaminarayan Mandir, Maninagar, Ahmedabad 380008")
    b = rec("b", "Y", "Shreeji Society, Maninagar, Amdavad, 380 008")
    assert f_address_core(a, b).value > 0.85


# ----------------------------------------------- missing fields are neutral #


@pytest.mark.parametrize(
    ("feature", "left", "right"),
    [
        ("phone", rec("a", "X", phone="9825012345"), rec("b", "Y")),
        ("pincode", rec("a", "X", "Maninagar 380008"), rec("b", "Y", "Maninagar")),
        ("address_core", rec("a", "X", "Maninagar 380008"), rec("b", "Y")),
        ("landmark", rec("a", "X", "nr temple, Maninagar"), rec("b", "Y", "Maninagar")),
    ],
)
def test_a_missing_field_yields_none_not_zero(feature, left, right) -> None:
    features = compute_features(left, right)
    assert features[feature].value is None


def test_address_with_no_identifying_tokens_is_none() -> None:
    assert f_address_core(rec("a", "X", "..."), rec("b", "Y", "...")).value is None


def test_none_features_are_excluded_from_the_mean() -> None:
    """The core promise: absence must not drag a score down."""
    present = {
        "a": FeatureValue(name="a", value=1.0, weight=0.5),
        "b": FeatureValue(name="b", value=1.0, weight=0.5),
    }
    with_missing = {
        "a": FeatureValue(name="a", value=1.0, weight=0.5),
        "b": FeatureValue(name="b", value=None, weight=0.5),
    }
    assert combine(present) == pytest.approx(1.0)
    assert combine(with_missing) == pytest.approx(1.0)


def test_a_wrong_value_does_drag_the_score_down() -> None:
    disagreeing = {
        "a": FeatureValue(name="a", value=1.0, weight=0.5),
        "b": FeatureValue(name="b", value=0.0, weight=0.5),
    }
    assert combine(disagreeing) == pytest.approx(0.5)


def test_combine_with_nothing_computable() -> None:
    assert combine({"a": FeatureValue(name="a", value=None, weight=1.0)}) == 0.0
    assert combine({}) == 0.0


def test_zero_weight_features_are_ignored() -> None:
    features = {
        "a": FeatureValue(name="a", value=1.0, weight=0.0),
        "b": FeatureValue(name="b", value=0.2, weight=1.0),
    }
    assert combine(features) == pytest.approx(0.2)


# ------------------------------------------------------------ pair scoring #


def test_score_pair_merges_the_motivating_example() -> None:
    weights = weights_config()
    left = rec(
        "r1",
        "Jayeshbhai Patel",
        "12, Shreeji Soc., Nr. Swaminarayan Mandir, Maninagar, Ahmedabad 380008",
        "+91 98250 12345",
    )
    right = rec(
        "r2",
        "જયેશ પટેલ",
        "12 Shriji Society, opp Swaminarayan Temple, Maninagar, Ahmedabad",
        "9825012345",
    )
    pair = score_pair(left, right, weights)
    assert pair.decision == "merge"
    assert pair.score > 0.9


def test_two_different_people_with_the_same_name_do_not_merge() -> None:
    """The hardest and most important case in Indian entity resolution.

    Identical names, different address, different pincode, different phone.
    The name weights sum to less than the merge threshold precisely so this
    pair cannot be merged on its name alone.
    """
    weights = weights_config()
    left = rec("a", "Jayesh Patel", "12 Shreeji Soc, Maninagar, Ahmedabad 380008", "9825012345")
    right = rec("b", "Jayesh Patel", "5 Sun Apt, Bopal, Ahmedabad 380058", "9811122233")
    pair = score_pair(left, right, weights)
    assert pair.decision != "merge"
    assert pair.score < weights.auto_merge


def test_identical_names_alone_cannot_reach_the_merge_threshold() -> None:
    name_weight = sum(
        spec.weight for name, spec in weights_config().features.items() if name.startswith("name_")
    )
    assert name_weight < weights_config().auto_merge


def test_name_only_records_do_merge_a_documented_limitation() -> None:
    """With no other field present there is nothing else to go on."""
    pair = score_pair(rec("a", "Jayesh Patel"), rec("b", "Jayesh Patil"))
    assert pair.decision == "merge"


def test_pair_id_is_order_independent() -> None:
    weights = weights_config()
    left, right = rec("b", "Jayesh Patel"), rec("a", "Jayesh Patel")
    assert score_pair(left, right, weights).pair_id == "a::b"


def test_decide_bands() -> None:
    weights = weights_config()
    assert decide(weights.auto_merge, weights) == "merge"
    assert decide(weights.auto_merge - 0.01, weights) == "review"
    assert decide(weights.review_low, weights) == "review"
    assert decide(weights.review_low - 0.01, weights) == "reject"


def test_unimplemented_feature_in_config_fails_loudly() -> None:
    weights = weights_config()
    broken = weights.with_overrides(feature_weights={"phone": 0.1})
    object.__setattr__(
        broken,
        "features",
        {**broken.features, "astrology": broken.features["phone"]},
    )
    with pytest.raises(KeyError, match="no function implements it"):
        compute_features(rec("a", "X"), rec("b", "Y"), broken)


def test_feature_to_dict_is_json_safe() -> None:
    pair = score_pair(rec("a", "Jayesh Patel"), rec("b", "Jayesh Patil"))
    payload = pair.to_dict()
    assert payload["pair_id"] == "a::b"
    assert payload["features"]["name_phonetic"]["contribution"] > 0
