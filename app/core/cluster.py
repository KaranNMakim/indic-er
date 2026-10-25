"""Clustering: from scored pairs to canonical entities.

Connected components are the obvious way to turn pairwise matches into groups,
and on their own they are dangerous. Transitivity is not guaranteed by a
pairwise score: if A~B and B~C both pass the threshold, A and C are merged even
when they are plainly different people. One hub record — a "Patel" in a big
pincode with a common address — can chain two real entities together, and on a
dataset full of Patels and Shahs the components grow without limit.

So after components are formed, every component is **conflict-checked**. A
rule from ``config/weights.yaml`` names the evidence that must disagree (two
different pincodes *and* two different phones, say). When a rule fires for a
member pair, the weakest ``same_as`` edge on the path between them is cut and
the component is re-split — unless that edge is strong enough to override the
conflict, in which case the conflict is recorded rather than acted on, because
a 0.95 pair is better evidence than a phone that someone changed.

Human decisions sit above all of it: a reviewer's accept is a forced edge that
conflict resolution may not cut, and a reject is a forbidden edge.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from app.core.config import WeightsConfig, weights_config
from app.core.features import f_address_core
from app.core.models import CanonicalEntity, NormalizedRecord, PairScore

log = logging.getLogger(__name__)

# Conflict checking is O(n^2) inside a component. Above this size the check is
# skipped and logged rather than allowed to dominate the run.
MAX_CONFLICT_CHECK_SIZE = 250
MAX_SPLIT_ITERATIONS = 200

HUMAN_ACCEPT = "human_accept"
HUMAN_REJECT = "human_reject"


@dataclass(slots=True)
class ClusterOutcome:
    entities: list[CanonicalEntity] = field(default_factory=list)
    splits: list[dict[str, Any]] = field(default_factory=list)
    unresolved_conflicts: list[dict[str, Any]] = field(default_factory=list)
    skipped_large_components: list[int] = field(default_factory=list)

    def stats(self) -> dict[str, Any]:
        sizes = [e.size for e in self.entities]
        multi = [s for s in sizes if s > 1]
        return {
            "entities": len(self.entities),
            "singletons": sum(1 for s in sizes if s == 1),
            "merged_entities": len(multi),
            "largest_entity": max(sizes, default=0),
            "mean_merged_size": round(sum(multi) / len(multi), 2) if multi else 0.0,
            "conflict_splits": len(self.splits),
            "unresolved_conflicts": len(self.unresolved_conflicts),
            "skipped_large_components": len(self.skipped_large_components),
        }


# --------------------------------------------------------------------------- #
# overrides
# --------------------------------------------------------------------------- #


def apply_overrides(
    pairs: list[PairScore], overrides: dict[str, str] | None, weights: WeightsConfig
) -> list[PairScore]:
    """Replace machine decisions with human ones where a human has decided.

    A reviewer's call is not a nudge to the score — it is the answer. The
    score is kept untouched so the eval can still measure what the model
    thought, and ``override`` records who overruled it.
    """
    if not overrides:
        return pairs
    out: list[PairScore] = []
    for pair in pairs:
        verdict = overrides.get(pair.pair_id)
        if verdict == "accept":
            pair.decision = "merge"
            pair.override = HUMAN_ACCEPT
        elif verdict == "reject":
            pair.decision = "reject"
            pair.override = HUMAN_REJECT
        out.append(pair)
    return out


# --------------------------------------------------------------------------- #
# conflicts
# --------------------------------------------------------------------------- #


def conflict_evidence(
    left: NormalizedRecord, right: NormalizedRecord, weights: WeightsConfig
) -> set[str]:
    """Which conflict conditions two records satisfy.

    Note what is *not* here: a missing field never contributes evidence of a
    conflict. ``address_weak`` requires an address on both sides that actually
    scores badly — an absent address is unknown, not weak, and splitting an
    entity because a field was blank would be exactly the wrong behaviour.
    """
    evidence: set[str] = set()

    pin_left = left.address.pincode if left.address else None
    pin_right = right.address.pincode if right.address else None
    if pin_left and pin_right and pin_left != pin_right:
        evidence.add("pincode_differs")

    if left.phone_last10 and right.phone_last10 and left.phone_last10 != right.phone_last10:
        evidence.add("phone_differs")

    address = f_address_core(left, right)
    if address.value is not None and address.value < weights.address_weak_below:
        evidence.add("address_weak")

    return evidence


def find_conflicts(
    members: list[str], records: dict[str, NormalizedRecord], weights: WeightsConfig
) -> list[tuple[str, str, str]]:
    """Member pairs that trip a conflict rule, as ``(left, right, rule)``."""
    if not weights.conflicts_enabled or not weights.conflict_rules:
        return []
    found: list[tuple[str, str, str]] = []
    for i, left_id in enumerate(members):
        left = records.get(left_id)
        if left is None:
            continue
        for right_id in members[i + 1 :]:
            right = records.get(right_id)
            if right is None:
                continue
            evidence = conflict_evidence(left, right, weights)
            if not evidence:
                continue
            for rule in weights.conflict_rules:
                if set(rule.requires) <= evidence:
                    found.append((left_id, right_id, rule.name))
                    break
    return found


def _resolve_component(
    graph: nx.Graph,
    members: list[str],
    records: dict[str, NormalizedRecord],
    weights: WeightsConfig,
    outcome: ClusterOutcome,
) -> None:
    """Cut the weakest bridging edges until no conflict remains.

    Operates on ``graph`` in place. Human-accepted edges are protected: a
    reviewer has already looked at that pair and said yes.
    """
    if len(members) > MAX_CONFLICT_CHECK_SIZE:
        outcome.skipped_large_components.append(len(members))
        log.warning(
            "conflict_check_skipped",
            extra={"component_size": len(members), "limit": MAX_CONFLICT_CHECK_SIZE},
        )
        return

    immovable: set[tuple[str, str]] = set()
    for _ in range(MAX_SPLIT_ITERATIONS):
        conflicts = find_conflicts(members, records, weights)
        actionable = [c for c in conflicts if (c[0], c[1]) not in immovable]
        if not actionable:
            return

        progressed = False
        for left_id, right_id, rule in actionable:
            if not graph.has_node(left_id) or not graph.has_node(right_id):
                continue
            if not nx.has_path(graph, left_id, right_id):
                continue
            path = nx.shortest_path(graph, left_id, right_id)
            edges = [
                (path[i], path[i + 1], graph.edges[path[i], path[i + 1]])
                for i in range(len(path) - 1)
            ]
            cuttable = [e for e in edges if e[2].get("override") != HUMAN_ACCEPT]
            if not cuttable:
                immovable.add((left_id, right_id))
                outcome.unresolved_conflicts.append(
                    {
                        "left_id": left_id,
                        "right_id": right_id,
                        "rule": rule,
                        "reason": "every edge on the path was confirmed by a reviewer",
                    }
                )
                continue
            weakest = min(cuttable, key=lambda e: e[2].get("score", 0.0))
            if weakest[2].get("score", 0.0) >= weights.conflict_override_score:
                immovable.add((left_id, right_id))
                outcome.unresolved_conflicts.append(
                    {
                        "left_id": left_id,
                        "right_id": right_id,
                        "rule": rule,
                        "reason": (
                            f"kept: weakest link on the path still scores "
                            f"{weakest[2]['score']:.2f}, above the override "
                            f"threshold {weights.conflict_override_score:.2f}"
                        ),
                    }
                )
                continue
            graph.remove_edge(weakest[0], weakest[1])
            outcome.splits.append(
                {
                    "rule": rule,
                    "conflict_between": [left_id, right_id],
                    "cut_edge": [weakest[0], weakest[1]],
                    "cut_edge_score": round(weakest[2].get("score", 0.0), 4),
                    "reason": (
                        f"{rule}: {left_id} and {right_id} were in one cluster but "
                        f"disagree on hard evidence; cut the weakest link "
                        f"({weakest[0]} ~ {weakest[1]}, {weakest[2].get('score', 0.0):.2f})"
                    ),
                }
            )
            progressed = True
            break

        if not progressed:
            return

    log.warning("conflict_resolution_hit_iteration_limit", extra={"size": len(members)})


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #


def cluster_records(
    records: dict[str, NormalizedRecord],
    pairs: list[PairScore],
    weights: WeightsConfig | None = None,
    overrides: dict[str, str] | None = None,
) -> ClusterOutcome:
    """Turn scored pairs into canonical entities."""
    weights = weights or weights_config()
    pairs = apply_overrides(pairs, overrides, weights)
    outcome = ClusterOutcome()

    graph = nx.Graph()
    graph.add_nodes_from(records)
    for pair in pairs:
        if pair.decision != "merge":
            continue
        graph.add_edge(
            pair.left_id,
            pair.right_id,
            score=pair.score,
            override=pair.override,
            pair_id=pair.pair_id,
        )

    # Resolve conflicts component by component, then re-read the components:
    # cutting an edge can turn one component into two.
    for component in list(nx.connected_components(graph)):
        if len(component) > 1:
            _resolve_component(graph, sorted(component), records, weights, outcome)

    pairs_by_id = {pair.pair_id: pair for pair in pairs}
    split_members = {member for split in outcome.splits for member in split["cut_edge"]}

    for component in nx.connected_components(graph):
        members = sorted(component)
        entity = build_entity(
            members,
            records,
            graph,
            pairs_by_id,
            was_split=bool(split_members & set(members)),
            splits=outcome.splits,
        )
        outcome.entities.append(entity)

    outcome.entities.sort(key=lambda e: (-e.size, e.canonical_name))
    return outcome


def build_entity(
    members: list[str],
    records: dict[str, NormalizedRecord],
    graph: nx.Graph,
    pairs_by_id: dict[str, PairScore],
    *,
    was_split: bool = False,
    splits: list[dict[str, Any]] | None = None,
) -> CanonicalEntity:
    """Build the canonical record for one cluster, with merge provenance."""
    members_records = [records[m] for m in members if m in records]
    if not members_records:
        raise ValueError(f"cluster {members} contains no known records")

    weights = weights_config()
    kind = Counter(r.kind for r in members_records).most_common(1)[0][0]
    canonical = _pick_canonical(members_records, weights)

    edge_scores: list[float] = []
    evidence: list[dict[str, Any]] = []
    human_confirmed = False
    subgraph = graph.subgraph(members)
    for left, right, data in subgraph.edges(data=True):
        edge_scores.append(data.get("score", 0.0))
        if data.get("override") == HUMAN_ACCEPT:
            human_confirmed = True
        pair = pairs_by_id.get(data.get("pair_id", ""))
        evidence.append(
            {
                "pair_id": data.get("pair_id") or f"{left}::{right}",
                "left_id": left,
                "right_id": right,
                "score": round(data.get("score", 0.0), 4),
                "override": data.get("override"),
                "top_features": _top_features(pair) if pair else [],
            }
        )

    review_state = "auto"
    if human_confirmed:
        review_state = "human_confirmed"
    elif was_split:
        review_state = "conflict_split"

    split_reason = None
    if was_split and splits:
        member_set = set(members)
        for split in splits:
            if member_set & set(split["cut_edge"]):
                split_reason = split["reason"]
                break

    return CanonicalEntity(
        entity_id=entity_id_for(members),
        kind=kind,
        canonical_name=canonical.name.display,
        member_ids=tuple(members),
        cluster_score=(sum(edge_scores) / len(edge_scores)) if edge_scores else 1.0,
        review_state=review_state,
        canonical_address=_pick_address(members_records),
        canonical_phone=_pick_phone(members_records),
        variants=tuple(dict.fromkeys(r.raw_name for r in members_records)),
        scripts=tuple(sorted({r.name.script for r in members_records})),
        sources=tuple(sorted({r.source for r in members_records})),
        merge_evidence=tuple(sorted(evidence, key=lambda e: -e["score"])),
        split_reason=split_reason,
    )


def entity_id_for(members: list[str]) -> str:
    """Deterministic id, so the same cluster keeps its id across runs."""
    digest = hashlib.sha1("|".join(sorted(members)).encode()).hexdigest()[:10]
    return f"ent_{digest}"


def _pick_canonical(members: list[NormalizedRecord], weights: WeightsConfig) -> NormalizedRecord:
    """Choose the record whose name becomes the canonical one."""
    strategy = weights.canonical_name_strategy

    def latin_bonus(record: NormalizedRecord) -> int:
        if not weights.canonical_prefer_latin:
            return 0
        return 1 if record.name.script == "latin" else 0

    if strategy == "most_frequent":
        counts = Counter(r.name.display for r in members)
        winner = counts.most_common(1)[0][0]
        return next(r for r in members if r.name.display == winner)
    if strategy == "longest":
        return max(members, key=lambda r: (len(r.name.display), latin_bonus(r), r.record_id))
    # most_complete (default): the most name tokens, no bare initials, Latin
    # preferred, record id as a deterministic tiebreak.
    return max(
        members,
        key=lambda r: (
            sum(1 for t in r.name.tokens if len(t) > 1),
            latin_bonus(r),
            len(r.name.display),
            r.record_id,
        ),
    )


def _pick_address(members: list[NormalizedRecord]):
    """The most complete address in the cluster."""
    addresses = [r.address for r in members if r.address is not None]
    if not addresses:
        return None
    return max(
        addresses,
        key=lambda a: (
            sum(
                1
                for value in (a.house, a.building, a.landmark, a.locality, a.city, a.pincode)
                if value
            ),
            len(a.raw),
        ),
    )


def _pick_phone(members: list[NormalizedRecord]) -> str | None:
    phones = [r.phone_last10 for r in members if r.phone_last10]
    if not phones:
        return None
    return Counter(phones).most_common(1)[0][0]


def _top_features(pair: PairScore, limit: int = 3) -> list[dict[str, Any]]:
    usable = [f for f in pair.features.values() if f.value is not None]
    usable.sort(key=lambda f: f.contribution, reverse=True)
    return [
        {"name": f.name, "value": round(f.value, 3), "contribution": round(f.contribution, 4)}
        for f in usable[:limit]
    ]
