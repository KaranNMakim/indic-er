"""The pipeline end to end, and explanations."""

from __future__ import annotations

import pytest

from app.core.config import weights_config
from app.core.explain import explain, feature_table, highlights
from app.core.models import InputRecord
from app.core.normalize import normalize_record
from app.core.pipeline import match_two, resolve


def test_the_motivating_example_resolves_to_one_entity(
    jayesh_records: list[InputRecord],
) -> None:
    """Three records, three scripts, three source systems, one person."""
    result = resolve(jayesh_records)
    assert len(result.entities) == 1
    entity = result.entities[0]
    assert entity.size == 3
    assert entity.canonical_name == "Jayesh Patel"
    assert set(entity.sources) == {"kyc", "wallet", "logistics"}
    assert entity.canonical_address.pincode == "380008"
    assert entity.canonical_phone == "9825012345"


def test_a_different_person_with_the_same_name_stays_separate(
    jayesh_records: list[InputRecord],
) -> None:
    other = InputRecord(
        id="x2",
        name="Jayesh Patel",
        address="5, Sun Apt, Bopal, Ahmedabad 380058",
        phone="9811122233",
        source="kyc",
    )
    result = resolve([*jayesh_records, other])
    entity_ids = {e.entity_id for e in result.entities if "x2" in e.member_ids}
    assert len(entity_ids) == 1
    entity = next(e for e in result.entities if "x2" in e.member_ids)
    assert entity.size == 1


def test_an_unrelated_record_is_not_merged(
    jayesh_records: list[InputRecord], unrelated_record: InputRecord
) -> None:
    result = resolve([*jayesh_records, unrelated_record])
    assert len(result.entities) == 2
    assert {e.size for e in result.entities} == {3, 1}


def test_resolve_is_deterministic(jayesh_records: list[InputRecord]) -> None:
    """No randomness, no network, no LLM: the same input, the same answer."""
    first = resolve(jayesh_records).to_dict(include_normalized=False)
    second = resolve(jayesh_records).to_dict(include_normalized=False)
    assert first["entities"] == second["entities"]


def test_record_order_does_not_change_the_entities(
    jayesh_records: list[InputRecord],
) -> None:
    forward = resolve(jayesh_records)
    backward = resolve(list(reversed(jayesh_records)))
    assert {e.entity_id for e in forward.entities} == {e.entity_id for e in backward.entities}


def test_empty_input() -> None:
    result = resolve([])
    assert result.entities == []
    assert result.stats["duplicate_rate"] == 0.0


def test_a_single_record_is_one_entity() -> None:
    result = resolve([InputRecord(id="a", name="Jayesh Patel")])
    assert len(result.entities) == 1
    assert result.stats["duplicate_rate"] == 0.0


def test_duplicate_rate_is_the_headline_number(
    jayesh_records: list[InputRecord],
) -> None:
    """Three records collapsing to one entity is a 67% duplicate rate."""
    result = resolve(jayesh_records)
    assert result.stats["duplicate_rate"] == pytest.approx(0.6667, abs=1e-3)


def test_stats_account_for_every_pair(jayesh_records: list[InputRecord]) -> None:
    stats = resolve(jayesh_records).stats
    assert (
        stats["pairs_merged"] + stats["pairs_review"] + stats["pairs_rejected"]
        == stats["pairs_scored"]
    )
    assert stats["timings_ms"]["total"] >= 0


def test_normalized_detail_is_returned_for_the_ui(
    jayesh_records: list[InputRecord],
) -> None:
    result = resolve(jayesh_records)
    payload = result.to_dict()
    assert payload["normalized"]["r2"]["name"]["script"] == "gujarati"
    assert payload["normalized"]["r1"]["name"]["suffixes_removed"] == ["bhai"]


def test_review_pairs_are_surfaced_in_score_order() -> None:
    records = [
        InputRecord(
            id="a",
            name="Jayesh Patel",
            address="12 Shreeji Soc, Maninagar, Ahmedabad 380008",
            phone="9825012345",
        ),
        InputRecord(
            id="b",
            name="Jayesh Patel",
            address="5 Sun Apt, Bopal, Ahmedabad 380058",
            phone="9811122233",
        ),
        InputRecord(
            id="c",
            name="Jayesh Patel",
            address="9 Moon Apt, Vastrapur, Ahmedabad 380015",
            phone="9700011122",
        ),
    ]
    result = resolve(records)
    scores = [p.score for p in result.review_pairs]
    assert scores == sorted(scores, reverse=True)


def test_a_human_override_changes_the_entities() -> None:
    """The point of the review queue: decisions shape the output."""
    records = [
        InputRecord(
            id="a", name="Jayesh Patel", address="12 Shreeji Soc, Maninagar, Ahmedabad 380008"
        ),
        InputRecord(id="b", name="Jayesh Patel", address="5 Sun Apt, Bopal, Ahmedabad 380058"),
    ]
    before = resolve(records)
    pair_id = before.review_pairs[0].pair_id if before.review_pairs else "a::b"
    after = resolve(records, overrides={pair_id: "accept"})
    assert len(after.entities) < len(before.entities) or after.entities[0].size == 2
    assert after.stats["overrides_applied"] == 1


def test_overrides_are_counted_in_stats(jayesh_records: list[InputRecord]) -> None:
    result = resolve(jayesh_records, overrides={"r1::r3": "reject"})
    assert result.stats["overrides_applied"] == 1


def test_a_rejected_override_splits_an_entity(
    jayesh_records: list[InputRecord],
) -> None:
    baseline = resolve(jayesh_records)
    assert len(baseline.entities) == 1
    overridden = resolve(
        jayesh_records, overrides={"r1::r2": "reject", "r1::r3": "reject", "r2::r3": "reject"}
    )
    assert len(overridden.entities) == 3


def test_explanations_can_be_switched_off(jayesh_records: list[InputRecord]) -> None:
    result = resolve(jayesh_records, explanations=False)
    assert all(not pair.explanation for pair in result.pairs)


# --------------------------------------------------------------- match_two #


def test_match_two_scores_a_pair_blocking_would_never_propose() -> None:
    """``/match`` must answer about any two records, not only candidates."""
    left = InputRecord(id="a", name="Jayesh Patel", phone="9825012345")
    right = InputRecord(id="b", name="Murugan Pillai", address="T Nagar, Chennai 600017")
    pair, _, _ = match_two(left, right)
    assert pair.blocked_by == ("direct",)
    assert pair.decision == "reject"
    assert pair.explanation


def test_match_two_rejects_the_same_id_twice() -> None:
    record = InputRecord(id="a", name="Jayesh Patel")
    with pytest.raises(ValueError, match="two different record ids"):
        match_two(record, record)


# ------------------------------------------------------------ explanations #


def test_explanation_names_the_evidence(jayesh_records: list[InputRecord]) -> None:
    result = resolve(jayesh_records)
    pair = next(p for p in result.pairs if p.pair_id == "r1::r2")
    text = pair.explanation
    assert "Same entity" in text
    assert "phone" in text.lower()
    assert "sound identical" in text or "name sound" in text


def test_explanation_says_what_it_could_not_compare() -> None:
    left = normalize_record(InputRecord(id="a", name="Jayesh Patel", phone="9825012345"))
    right = normalize_record(InputRecord(id="b", name="Jayesh Patil"))
    from app.core.features import score_pair

    pair = score_pair(left, right)
    text = explain(pair, left, right)
    assert "Not comparable on" in text
    assert "left out of the score rather than counted against" in text


def test_explanation_explains_a_rejection() -> None:
    left = normalize_record(
        InputRecord(
            id="a", name="Jayesh Patel", address="Maninagar, Ahmedabad 380008", phone="9825012345"
        )
    )
    right = normalize_record(
        InputRecord(
            id="b", name="Nilesh Patel", address="Sector 12, Gandhinagar 382012", phone="9900011122"
        )
    )
    from app.core.features import score_pair

    text = explain(score_pair(left, right), left, right)
    assert "Different entities" in text
    assert "Against:" in text


def test_explanation_flags_a_common_surname_only_match() -> None:
    left = normalize_record(InputRecord(id="a", name="Jayesh Patel"))
    right = normalize_record(InputRecord(id="b", name="Nilesh Patel"))
    from app.core.features import score_pair

    text = explain(score_pair(left, right), left, right)
    assert "common surname" in text


def test_explanation_defers_to_a_human_decision() -> None:
    left = normalize_record(InputRecord(id="a", name="Jayesh Patel"))
    right = normalize_record(InputRecord(id="b", name="Jayesh Patil"))
    from app.core.features import score_pair

    pair = score_pair(left, right)
    pair.override = "human_accept"
    text = explain(pair, left, right)
    assert "A reviewer accepted this pair" in text
    assert "human decision stands" in text


def test_explanation_when_nothing_is_comparable() -> None:
    left = normalize_record(InputRecord(id="a", name="..."))
    right = normalize_record(InputRecord(id="b", name="Jayesh Patel"))
    from app.core.features import score_pair

    pair = score_pair(left, right)
    for feature in pair.features.values():
        object.__setattr__(feature, "value", None)
    assert "nothing" in explain(pair, left, right).lower()


def test_explanation_mentions_the_review_band() -> None:
    weights = weights_config()
    records = [
        InputRecord(
            id="a",
            name="Jayesh Patel",
            address="12 Shreeji Soc, Maninagar, Ahmedabad 380008",
            phone="9825012345",
        ),
        InputRecord(
            id="b",
            name="Jayesh Patel",
            address="5 Sun Apt, Bopal, Ahmedabad 380058",
            phone="9811122233",
        ),
    ]
    result = resolve(records)
    review = [p for p in result.pairs if p.decision == "review"]
    if review:
        assert f"{weights.review_low:.2f}" in review[0].explanation


def test_highlights_point_at_the_tokens_that_carried_the_match() -> None:
    left = normalize_record(InputRecord(id="a", name="Jayeshbhai Patel"))
    right = normalize_record(InputRecord(id="b", name="જયેશ પટેલ"))
    from app.core.features import score_pair

    marks = highlights(score_pair(left, right), left, right)
    assert "jayesh" in marks["left"]
    assert "patel" in marks["right"]
    assert marks["left_name"] == "Jayesh Patel"


def test_feature_table_puts_what_mattered_first(
    jayesh_records: list[InputRecord],
) -> None:
    result = resolve(jayesh_records)
    pair = next(p for p in result.pairs if p.pair_id == "r1::r2")
    rows = feature_table(pair)
    contributions = [row["contribution"] for row in rows if row["computed"]]
    assert contributions == sorted(contributions, reverse=True)
    assert all(row["computed"] for row in rows[: len(contributions)])
    assert {"name", "label", "value", "weight", "contribution", "detail"} <= set(rows[0])
