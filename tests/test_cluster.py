"""Clustering, conflict resolution and canonical records."""

from __future__ import annotations

import pytest

from app.core.cluster import (
    HUMAN_ACCEPT,
    apply_overrides,
    build_entity,
    cluster_records,
    conflict_evidence,
    entity_id_for,
    find_conflicts,
)
from app.core.config import weights_config
from app.core.features import score_pair
from app.core.models import InputRecord, PairScore
from app.core.normalize import normalize_records


def norm(rows: list[tuple]):
    """rows of (id, name, address, phone)."""
    return normalize_records(
        [
            InputRecord(id=rid, name=name, address=address, phone=phone)
            for rid, name, address, phone in rows
        ]
    )


def pair(left: str, right: str, score: float, decision: str = "merge") -> PairScore:
    return PairScore(left_id=left, right_id=right, score=score, decision=decision)


# ---------------------------------------------------------------- overrides #


def test_overrides_replace_the_machine_decision() -> None:
    weights = weights_config()
    pairs = [pair("a", "b", 0.60, "review"), pair("c", "d", 0.95, "merge")]
    result = apply_overrides(pairs, {"a::b": "accept", "c::d": "reject"}, weights)
    assert result[0].decision == "merge"
    assert result[0].override == HUMAN_ACCEPT
    assert result[1].decision == "reject"


def test_overrides_leave_the_score_untouched() -> None:
    """The eval must still be able to see what the model thought."""
    pairs = [pair("a", "b", 0.60, "review")]
    apply_overrides(pairs, {"a::b": "accept"}, weights_config())
    assert pairs[0].score == 0.60


def test_no_overrides_is_a_no_op() -> None:
    pairs = [pair("a", "b", 0.9)]
    assert apply_overrides(pairs, None, weights_config()) is pairs


# ---------------------------------------------------------------- conflicts #


def test_conflict_evidence_needs_both_sides_present() -> None:
    """A missing field is never evidence of a conflict."""
    weights = weights_config()
    records = norm(
        [
            ("a", "Jayesh Patel", "Maninagar, Ahmedabad 380008", "9825012345"),
            ("b", "Jayesh Patel", None, None),
        ]
    )
    assert conflict_evidence(records["a"], records["b"], weights) == set()


def test_conflict_evidence_detects_hard_disagreement() -> None:
    weights = weights_config()
    records = norm(
        [
            ("a", "Jayesh Patel", "12 Shreeji Soc, Maninagar, Ahmedabad 380008", "9825012345"),
            ("b", "Jayesh Patel", "5 Sun Apt, Bopal, Ahmedabad 380058", "9811122233"),
        ]
    )
    evidence = conflict_evidence(records["a"], records["b"], weights)
    assert "pincode_differs" in evidence
    assert "phone_differs" in evidence


def test_find_conflicts_matches_a_configured_rule() -> None:
    weights = weights_config()
    records = norm(
        [
            ("a", "Jayesh Patel", "12 Shreeji Soc, Maninagar, Ahmedabad 380008", "9825012345"),
            ("b", "Jayesh Patel", "5 Sun Apt, Bopal, Ahmedabad 380058", "9811122233"),
        ]
    )
    conflicts = find_conflicts(["a", "b"], records, weights)
    assert conflicts
    assert conflicts[0][2] == "different_pincode_and_phone"


def test_conflicts_can_be_turned_off() -> None:
    weights = weights_config().with_overrides(conflicts_enabled=False)
    records = norm(
        [
            ("a", "Jayesh Patel", "Maninagar, Ahmedabad 380008", "9825012345"),
            ("b", "Jayesh Patel", "Bopal, Ahmedabad 380058", "9811122233"),
        ]
    )
    assert find_conflicts(["a", "b"], records, weights) == []


# --------------------------------------------------------------- clustering #


def test_transitive_chain_is_split_when_the_ends_conflict() -> None:
    """The failure connected components cannot avoid on its own.

    A ~ B and B ~ C both pass, so plain components merge all three — but A and
    C have different pincodes AND different phones. The weakest link is cut.
    """
    weights = weights_config()
    records = norm(
        [
            ("a", "Jayesh Patel", "12 Shreeji Soc, Maninagar, Ahmedabad 380008", "9825012345"),
            ("b", "Jayesh Patel", "12 Shreeji Soc, Maninagar, Ahmedabad 380008", None),
            ("c", "Jayesh Patel", "5 Sun Apt, Bopal, Ahmedabad 380058", "9811122233"),
        ]
    )
    pairs = [pair("a", "b", 0.95), pair("b", "c", 0.74)]
    outcome = cluster_records(records, pairs, weights)
    members = {frozenset(e.member_ids) for e in outcome.entities}
    assert frozenset({"a", "b"}) in members
    assert frozenset({"c"}) in members
    assert outcome.splits
    assert outcome.splits[0]["cut_edge"] == ["b", "c"]


def test_a_strong_enough_edge_survives_a_conflict() -> None:
    """A 0.95 pair is better evidence than a phone number someone changed."""
    weights = weights_config()
    records = norm(
        [
            ("a", "Jayesh Patel", "12 Shreeji Soc, Maninagar, Ahmedabad 380008", "9825012345"),
            ("c", "Jayesh Patel", "12 Shreeji Soc, Maninagar, Ahmedabad 380058", "9811122233"),
        ]
    )
    pairs = [pair("a", "c", 0.96)]
    outcome = cluster_records(records, pairs, weights)
    assert len(outcome.entities) == 1
    assert outcome.unresolved_conflicts
    assert "above the override threshold" in outcome.unresolved_conflicts[0]["reason"]


def test_a_reviewer_confirmed_edge_is_never_cut() -> None:
    weights = weights_config()
    records = norm(
        [
            ("a", "Jayesh Patel", "12 Shreeji Soc, Maninagar, Ahmedabad 380008", "9825012345"),
            ("c", "Jayesh Patel", "5 Sun Apt, Bopal, Ahmedabad 380058", "9811122233"),
        ]
    )
    pairs = [pair("a", "c", 0.60, "review")]
    outcome = cluster_records(records, pairs, weights, overrides={"a::c": "accept"})
    assert len(outcome.entities) == 1
    assert outcome.entities[0].review_state == "human_confirmed"
    assert outcome.unresolved_conflicts
    assert "confirmed by a reviewer" in outcome.unresolved_conflicts[0]["reason"]


def test_singletons_are_entities_too() -> None:
    records = norm([("a", "Jayesh Patel", None, None), ("b", "Murugan Pillai", None, None)])
    outcome = cluster_records(records, [], weights_config())
    assert len(outcome.entities) == 2
    assert all(e.size == 1 for e in outcome.entities)
    assert all(e.cluster_score == 1.0 for e in outcome.entities)


def test_review_and_reject_pairs_do_not_merge() -> None:
    records = norm([("a", "Jayesh Patel", None, None), ("b", "Jayesh Patil", None, None)])
    outcome = cluster_records(records, [pair("a", "b", 0.60, "review")], weights_config())
    assert len(outcome.entities) == 2


def test_oversized_component_skips_the_conflict_check(monkeypatch) -> None:
    import app.core.cluster as cluster_module

    monkeypatch.setattr(cluster_module, "MAX_CONFLICT_CHECK_SIZE", 2)
    records = norm([(f"r{i}", "Jayesh Patel", None, None) for i in range(4)])
    pairs = [pair("r0", f"r{i}", 0.9) for i in range(1, 4)]
    outcome = cluster_records(records, pairs, weights_config())
    assert outcome.skipped_large_components == [4]
    assert len(outcome.entities) == 1


def test_cluster_stats() -> None:
    records = norm(
        [
            ("a", "Jayesh Patel", None, None),
            ("b", "Jayesh Patil", None, None),
            ("z", "Murugan Pillai", None, None),
        ]
    )
    outcome = cluster_records(records, [pair("a", "b", 0.9)], weights_config())
    stats = outcome.stats()
    assert stats["entities"] == 2
    assert stats["merged_entities"] == 1
    assert stats["singletons"] == 1
    assert stats["largest_entity"] == 2


# ----------------------------------------------------------------- entities #


def test_entity_id_is_deterministic_and_order_free() -> None:
    assert entity_id_for(["b", "a"]) == entity_id_for(["a", "b"])
    assert entity_id_for(["a"]) != entity_id_for(["b"])


def test_canonical_name_prefers_the_most_complete_latin_form() -> None:
    weights = weights_config()
    records = norm(
        [
            ("a", "J. Patil", None, None),
            ("b", "જયેશ પટેલ", None, None),
            ("c", "Jayeshbhai Patel", None, None),
        ]
    )
    pairs = [
        score_pair(records["a"], records["c"], weights),
        score_pair(records["b"], records["c"], weights),
    ]
    for p in pairs:
        p.decision = "merge"
    outcome = cluster_records(records, pairs, weights)
    entity = next(e for e in outcome.entities if e.size == 3)
    assert entity.canonical_name == "Jayesh Patel"
    assert "J. Patil" in entity.variants
    assert set(entity.scripts) == {"latin", "gujarati"}


def test_canonical_record_keeps_every_variant_and_source() -> None:
    weights = weights_config()
    records = normalize_records(
        [
            InputRecord(id="a", name="Jayeshbhai Patel", source="kyc"),
            InputRecord(id="b", name="જયેશ પટેલ", source="wallet"),
        ]
    )
    outcome = cluster_records(records, [pair("a", "b", 0.98)], weights)
    entity = outcome.entities[0]
    assert set(entity.sources) == {"kyc", "wallet"}
    assert len(entity.variants) == 2
    assert entity.merge_evidence
    assert entity.merge_evidence[0]["score"] == 0.98


def test_canonical_address_is_the_most_complete_one() -> None:
    records = norm(
        [
            ("a", "Jayesh Patel", "Maninagar, Ahmedabad", None),
            (
                "b",
                "Jayesh Patel",
                "12, Shreeji Soc, Nr. Swaminarayan Mandir, Maninagar, Ahmedabad 380008",
                None,
            ),
        ]
    )
    outcome = cluster_records(records, [pair("a", "b", 0.95)], weights_config())
    entity = outcome.entities[0]
    assert entity.canonical_address is not None
    assert entity.canonical_address.pincode == "380008"
    assert entity.canonical_address.landmark is not None


def test_canonical_phone_is_the_most_frequent() -> None:
    records = norm(
        [
            ("a", "Jayesh Patel", None, "9825012345"),
            ("b", "Jayesh Patel", None, "9825012345"),
            ("c", "Jayesh Patel", None, None),
        ]
    )
    pairs = [pair("a", "b", 0.99), pair("b", "c", 0.95)]
    outcome = cluster_records(records, pairs, weights_config())
    assert outcome.entities[0].canonical_phone == "9825012345"


@pytest.mark.parametrize("strategy", ["most_complete", "most_frequent", "longest"])
def test_every_canonical_name_strategy_works(strategy: str) -> None:
    weights = weights_config().with_overrides()
    object.__setattr__(weights, "canonical_name_strategy", strategy)
    records = norm(
        [
            ("a", "Jayesh Patel", None, None),
            ("b", "Jayesh Patel", None, None),
            ("c", "Jayeshkumar Ramesh Patel", None, None),
        ]
    )
    pairs = [pair("a", "b", 0.99), pair("b", "c", 0.9)]
    outcome = cluster_records(records, pairs, weights)
    assert outcome.entities[0].canonical_name


def test_build_entity_rejects_an_unknown_cluster() -> None:
    import networkx as nx

    with pytest.raises(ValueError, match="no known records"):
        build_entity(["ghost"], {}, nx.Graph(), {})


def test_entity_to_dict_is_json_safe() -> None:
    records = norm([("a", "Jayesh Patel", "Maninagar, Ahmedabad 380008", "9825012345")])
    outcome = cluster_records(records, [], weights_config())
    payload = outcome.entities[0].to_dict()
    assert payload["size"] == 1
    assert payload["canonical_address"]["pincode"] == "380008"
