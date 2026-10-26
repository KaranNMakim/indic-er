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


def test_a_transitive_chain_does_not_drag_in_a_conflicting_record() -> None:
    """The failure connected components cannot avoid on its own.

    A ~ B and B ~ C both pass the threshold, so plain components merge all
    three — but A and C have different pincodes AND different phones. Edges
    are applied strongest-first, so A~B (0.95) lands and the marginal B~C
    (0.74) is refused, because the cluster it would create would hold
    conflicting evidence.
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
    assert outcome.blocked_merges
    assert outcome.blocked_merges[0]["pair_id"] == "b::c"


def test_the_strongest_edge_wins_the_ground_it_is_entitled_to() -> None:
    """Edge order decides who keeps the hub, and it must be score order."""
    weights = weights_config()
    records = norm(
        [
            ("hub", "Jayesh Patel", "12 Shreeji Soc, Maninagar, Ahmedabad 380008", None),
            (
                "strong",
                "Jayesh Patel",
                "12 Shreeji Soc, Maninagar, Ahmedabad 380008",
                "9825012345",
            ),
            ("weak", "Jayesh Patel", "5 Sun Apt, Bopal, Ahmedabad 380058", "9811122233"),
        ]
    )
    pairs = [pair("hub", "weak", 0.75), pair("hub", "strong", 0.93)]
    outcome = cluster_records(records, pairs, weights)
    members = {frozenset(e.member_ids) for e in outcome.entities}
    assert frozenset({"hub", "strong"}) in members
    assert frozenset({"weak"}) in members


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
    assert "override threshold" in outcome.unresolved_conflicts[0]["reason"]


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
    assert "reviewer" in outcome.unresolved_conflicts[0]["reason"]


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


def test_a_hub_cannot_chain_many_pincodes_into_one_entity() -> None:
    """The blowup this algorithm exists to prevent.

    One hub record and seven others in seven different pincodes with seven
    different phones. Connected components would return a single entity of
    eight — on the full dataset that failure produced one entity holding
    4,702 records. Conflict-aware agglomeration keeps the hub with at most
    one of them and refuses the rest.
    """
    rows = [
        (f"r{i}", "Jayesh Patel", f"Loc{i}, Pune 4110{i:02d}", f"98250123{i:02d}") for i in range(8)
    ]
    records = norm(rows)
    pairs = [pair("r0", f"r{i}", 0.80) for i in range(1, 8)]
    outcome = cluster_records(records, pairs, weights_config())
    assert outcome.blocked_merges
    assert max(e.size for e in outcome.entities) <= 2
    assert len(outcome.entities) >= 7


def test_a_blocked_merge_is_explained_on_the_entity() -> None:
    records = norm(
        [
            ("a", "Jayesh Patel", "12 Shreeji Soc, Maninagar, Ahmedabad 380008", "9825012345"),
            ("b", "Jayesh Patel", "5 Sun Apt, Bopal, Ahmedabad 380058", "9811122233"),
        ]
    )
    outcome = cluster_records(records, [pair("a", "b", 0.80)], weights_config())
    assert len(outcome.entities) == 2
    assert any(e.split_reason for e in outcome.entities)
    assert any(e.review_state == "conflict_separated" for e in outcome.entities)


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


def test_merge_conflict_is_evaluated_against_the_union() -> None:
    """Checking before the union is what makes a post-hoc split unnecessary."""
    from app.core.cluster import ClusterSummary, merge_conflict

    weights = weights_config()
    left = ClusterSummary(pincodes={"380008"}, phones={"9825012345"})
    right = ClusterSummary(pincodes={"380058"}, phones={"9811122233"})
    assert (
        merge_conflict(left, right, pair("a", "b", 0.8), weights) == "different_pincode_and_phone"
    )
    same_pin = ClusterSummary(pincodes={"380008"}, phones={"9811122233"})
    assert (
        merge_conflict(left, same_pin, pair("a", "b", 0.8), weights)
        != "different_pincode_and_phone"
    )


def test_a_strong_enough_pair_overrides_a_blocked_merge() -> None:
    records = norm(
        [
            ("a", "Jayesh Patel", "12 Shreeji Soc, Maninagar, Ahmedabad 380008", "9825012345"),
            ("b", "Jayesh Patel", "12 Shreeji Soc, Maninagar, Ahmedabad 380058", "9811122233"),
        ]
    )
    outcome = cluster_records(records, [pair("a", "b", 0.96)], weights_config())
    assert len(outcome.entities) == 1
    assert outcome.unresolved_conflicts
    assert "override threshold" in outcome.unresolved_conflicts[0]["reason"]


def test_union_find_is_order_independent() -> None:
    from app.core.cluster import _UnionFind

    forward = _UnionFind(["a", "b", "c"])
    forward.union("a", "b")
    forward.union("b", "c")
    backward = _UnionFind(["a", "b", "c"])
    backward.union("c", "b")
    backward.union("b", "a")
    # The root key is an internal detail; the grouping is what must match, and
    # entity ids are derived from the sorted members, not from the root.
    assert sorted(forward.groups().values()) == sorted(backward.groups().values())


def test_entity_to_dict_is_json_safe() -> None:
    records = norm([("a", "Jayesh Patel", "Maninagar, Ahmedabad 380008", "9825012345")])
    outcome = cluster_records(records, [], weights_config())
    payload = outcome.entities[0].to_dict()
    assert payload["size"] == 1
    assert payload["canonical_address"]["pincode"] == "380008"
