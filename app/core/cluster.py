"""Clustering: from scored pairs to canonical entities.

Connected components are the obvious way to turn pairwise matches into groups,
and on their own they are dangerous. Transitivity is not guaranteed by a
pairwise score: if A~B and B~C both pass the threshold, A and C are merged even
when they are plainly different people. On 5,000 synthetic records, plain
components collapsed 4,702 of them into a single "entity" — one hub record in
a dense Patel block is enough to chain the whole dataset together.

So this module does **conflict-aware agglomeration** instead. Merge edges are
applied strongest-first, and each one is refused if the cluster it would
create trips a conflict rule from ``config/weights.yaml`` — for instance a
cluster holding two different pincodes *and* two different phones. Because the
check runs *before* the union rather than after, a bad merge never happens and
there is nothing to unpick:

* it is a single pass, ``O(E log E)``, over the merge edges
* each check is a couple of set unions against a running per-cluster summary
* the strongest evidence wins the ground it is entitled to, and the marginal
  edge is the one refused
* a refusal is recorded with its reason, so "why are these two separate?" has
  an answer

Human decisions sit above all of it: a reviewer's accept is a forced edge that
no conflict rule may refuse, and a reject is a forbidden edge.

The earlier implementation built the components first and then tried to cut
the weakest edge on the path between each conflicting pair. It worked on small
inputs and was both slow and convoluted on real ones — thousands of iterations,
each re-scanning a huge graph. Checking before the union is cheaper and easier
to explain.
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

HUMAN_ACCEPT = "human_accept"
HUMAN_REJECT = "human_reject"

# The exact pairwise conflict scan used by `find_conflicts` is O(n^2); it is
# used for reporting on a single cluster, never in the hot path.
MAX_CONFLICT_CHECK_SIZE = 250


@dataclass(slots=True)
class ClusterOutcome:
    entities: list[CanonicalEntity] = field(default_factory=list)
    blocked_merges: list[dict[str, Any]] = field(default_factory=list)
    unresolved_conflicts: list[dict[str, Any]] = field(default_factory=list)
    transitivity_checks_skipped: int = 0

    def stats(self) -> dict[str, Any]:
        sizes = [e.size for e in self.entities]
        multi = [s for s in sizes if s > 1]
        return {
            "entities": len(self.entities),
            "singletons": sum(1 for s in sizes if s == 1),
            "merged_entities": len(multi),
            "largest_entity": max(sizes, default=0),
            "mean_merged_size": round(sum(multi) / len(multi), 2) if multi else 0.0,
            "blocked_merges": len(self.blocked_merges),
            "blocked_by_transitivity": sum(
                1 for b in self.blocked_merges if b["rule"] == "transitivity"
            ),
            "unresolved_conflicts": len(self.unresolved_conflicts),
            "transitivity_checks_skipped": self.transitivity_checks_skipped,
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
    for pair in pairs:
        verdict = overrides.get(pair.pair_id)
        if verdict == "accept":
            pair.decision = "merge"
            pair.override = HUMAN_ACCEPT
        elif verdict == "reject":
            pair.decision = "reject"
            pair.override = HUMAN_REJECT
    return pairs


# --------------------------------------------------------------------------- #
# conflicts
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ClusterSummary:
    """Just enough about a cluster to judge the next merge into it."""

    pincodes: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)

    @classmethod
    def for_record(cls, record: NormalizedRecord | None) -> ClusterSummary:
        summary = cls()
        if record is None:
            return summary
        if record.address and record.address.pincode:
            summary.pincodes.add(record.address.pincode)
        if record.phone_last10:
            summary.phones.add(record.phone_last10)
        return summary

    def merged_with(self, other: ClusterSummary) -> ClusterSummary:
        return ClusterSummary(
            pincodes=self.pincodes | other.pincodes,
            phones=self.phones | other.phones,
        )

    def absorb(self, other: ClusterSummary) -> None:
        self.pincodes |= other.pincodes
        self.phones |= other.phones


def conflict_evidence(
    left: NormalizedRecord, right: NormalizedRecord, weights: WeightsConfig
) -> set[str]:
    """Which conflict conditions two records satisfy.

    Note what is *not* here: a missing field never contributes evidence of a
    conflict. ``address_weak`` requires an address on both sides that actually
    scores badly — an absent address is unknown, not weak, and separating two
    records because a field was blank would be exactly the wrong behaviour.
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
    """Member pairs inside one cluster that trip a rule, as ``(l, r, rule)``.

    Reporting only — ``/entities/{id}`` uses it to explain a cluster. The hot
    path uses :func:`merge_conflict`, which is incremental.
    """
    if not weights.conflicts_enabled or not weights.conflict_rules:
        return []
    members = members[:MAX_CONFLICT_CHECK_SIZE]
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
            for rule in weights.conflict_rules:
                if set(rule.requires) <= evidence:
                    found.append((left_id, right_id, rule.name))
                    break
    return found


def build_rejection_index(
    pairs: list[PairScore], weights: WeightsConfig
) -> dict[str, dict[str, float]]:
    """``{record: {other_record: score}}`` for every pair below the veto floor.

    Only pairs that were *actually scored* appear here. A pair blocking never
    proposed is absent, and absence must not be read as disagreement: blocking
    recall is 99.6%, not 100%, and letting a missing candidate veto a merge
    would turn the blocking ceiling into a precision ceiling as well.

    The floor is `transitivity.veto_below`, not `review_low`. Tying the two
    together meant that widening the review band quietly weakened the veto.

    A human rejection is included whatever it scored — that is the strongest
    possible evidence that two records are not the same entity.
    """
    index: dict[str, dict[str, float]] = {}
    for pair in pairs:
        # A pair we are merging on can never also be evidence against itself.
        # Without this guard, any threshold below `veto_below` puts merge
        # edges into the rejection index, where they veto their own merge —
        # every sweep point at or under the veto floor then returns identical
        # numbers, because nothing below it can ever merge.
        if pair.decision == "merge" and pair.override != HUMAN_REJECT:
            continue
        rejected = (
            pair.override == HUMAN_REJECT
            or pair.score < weights.transitivity_veto_below
        )
        if not rejected:
            continue
        index.setdefault(pair.left_id, {})[pair.right_id] = pair.score
        index.setdefault(pair.right_id, {})[pair.left_id] = pair.score
    return index


def transitivity_clash(
    left_members: set[str],
    right_members: set[str],
    rejected_with: dict[str, dict[str, float]],
    weights: WeightsConfig,
    outcome: ClusterOutcome | None = None,
) -> tuple[str, str, float] | None:
    """The scored-and-rejected pair this merge would bring together, if any.

    A pairwise score carries no transitivity guarantee: A~B and B~C can both
    clear the threshold while A~C was compared directly and rejected.
    Agglomeration on its own will still put all three in one cluster, and in
    the evaluation that was the whole residual false-positive class — every
    sampled false-positive *pair* was decided `review` or `reject`, yet the
    records ended up sharing an entity through a chain of other edges.

    Iterates the smaller cluster and intersects each member's rejection set
    with the larger cluster, so the cost is bounded by the rejections actually
    recorded rather than by the product of the cluster sizes.
    """
    if not weights.transitivity_enabled or not rejected_with:
        return None

    small, large = (
        (left_members, right_members)
        if len(left_members) <= len(right_members)
        else (right_members, left_members)
    )
    if len(small) * len(large) > weights.transitivity_max_pair_checks:
        log.warning(
            "transitivity_check_skipped",
            extra={
                "left": len(left_members),
                "right": len(right_members),
                "limit": weights.transitivity_max_pair_checks,
            },
        )
        if outcome is not None:
            outcome.transitivity_checks_skipped += 1
        return None

    for member in sorted(small):
        rejections = rejected_with.get(member)
        if not rejections:
            continue
        clashing = rejections.keys() & large
        if clashing:
            other = min(clashing)
            left_id, right_id = sorted((member, other))
            return (left_id, right_id, rejections[other])
    return None


def merge_conflict(
    left: ClusterSummary,
    right: ClusterSummary,
    pair: PairScore,
    weights: WeightsConfig,
) -> str | None:
    """The rule the merged cluster would trip, or ``None``.

    Evaluated against the *union* of the two clusters, so a cluster can never
    come to hold a combination of evidence that a rule forbids — which is what
    makes a post-hoc split unnecessary.
    """
    if not weights.conflicts_enabled or not weights.conflict_rules:
        return None

    merged = left.merged_with(right)
    evidence: set[str] = set()
    if len(merged.pincodes) > 1:
        evidence.add("pincode_differs")
    if len(merged.phones) > 1:
        evidence.add("phone_differs")
    address = pair.features.get("address_core")
    if (
        address is not None
        and address.value is not None
        and address.value < weights.address_weak_below
    ):
        evidence.add("address_weak")
    if not evidence:
        return None

    for rule in weights.conflict_rules:
        if set(rule.requires) <= evidence:
            return rule.name
    return None


# --------------------------------------------------------------------------- #
# union-find
# --------------------------------------------------------------------------- #


class _UnionFind:
    """Union by size with path compression. Deterministic tie-breaks."""

    def __init__(self, items) -> None:
        self._parent: dict[str, str] = {item: item for item in items}
        self._size: dict[str, int] = dict.fromkeys(self._parent, 1)

    def add(self, item: str) -> None:
        if item not in self._parent:
            self._parent[item] = item
            self._size[item] = 1

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> str:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return root_left
        if self._size[root_left] < self._size[root_right] or (
            self._size[root_left] == self._size[root_right] and root_right < root_left
        ):
            root_left, root_right = root_right, root_left
        self._parent[root_right] = root_left
        self._size[root_left] += self._size[root_right]
        return root_left

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for item in self._parent:
            out.setdefault(self.find(item), []).append(item)
        return {root: sorted(members) for root, members in out.items()}


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #


def cluster_records(
    records: dict[str, NormalizedRecord],
    pairs: list[PairScore],
    weights: WeightsConfig | None = None,
    overrides: dict[str, str] | None = None,
) -> ClusterOutcome:
    """Turn scored pairs into canonical entities, strongest evidence first."""
    weights = weights or weights_config()
    pairs = apply_overrides(pairs, overrides, weights)
    outcome = ClusterOutcome()

    union = _UnionFind(records)
    summaries: dict[str, ClusterSummary] = {
        record_id: ClusterSummary.for_record(record) for record_id, record in records.items()
    }
    # Cluster membership, kept alongside union-find so the transitivity check
    # can ask "who is already in there?" without walking every record.
    members: dict[str, set[str]] = {record_id: {record_id} for record_id in records}
    rejected_with = build_rejection_index(pairs, weights)

    # Human-confirmed edges first, then by descending score, then by pair id
    # so the result never depends on dict ordering.
    merge_edges = sorted(
        (p for p in pairs if p.decision == "merge"),
        key=lambda p: (p.override == HUMAN_ACCEPT, p.score, p.pair_id),
        reverse=True,
    )

    graph = nx.Graph()
    graph.add_nodes_from(records)

    for pair in merge_edges:
        if pair.left_id not in summaries or pair.right_id not in summaries:
            continue
        root_left, root_right = union.find(pair.left_id), union.find(pair.right_id)
        if root_left == root_right:
            # Already together by a stronger route; keep the edge as evidence.
            _add_edge(graph, pair)
            continue

        forced = pair.override == HUMAN_ACCEPT

        # Transitivity: would this merge put a scored-and-rejected pair into
        # one cluster? Checked first, because it is the stronger objection —
        # the two records in question were compared directly and the answer
        # was no, where a conflict rule only infers a clash from field values.
        clash = (
            None
            if forced
            else transitivity_clash(
                members[root_left], members[root_right], rejected_with, weights, outcome
            )
        )
        if clash is not None:
            left_clash, right_clash, clash_score = clash
            if pair.score < weights.transitivity_override_score:
                outcome.blocked_merges.append(
                    {
                        "rule": "transitivity",
                        "pair_id": pair.pair_id,
                        "left_id": pair.left_id,
                        "right_id": pair.right_id,
                        "score": round(pair.score, 4),
                        "reason": (
                            f"transitivity: merging on {pair.left_id} ~ {pair.right_id} "
                            f"({pair.score:.2f}) would put {left_clash} and "
                            f"{right_clash} in one entity, and that pair was scored "
                            f"at {clash_score:.2f} — below the veto floor of "
                            f"{weights.transitivity_veto_below:.2f}"
                        ),
                    }
                )
                continue
            outcome.unresolved_conflicts.append(
                {
                    "rule": "transitivity",
                    "pair_id": pair.pair_id,
                    "left_id": pair.left_id,
                    "right_id": pair.right_id,
                    "score": round(pair.score, 4),
                    "reason": (
                        f"merged despite {left_clash} ~ {right_clash} scoring "
                        f"{clash_score:.2f}: the merging pair scores {pair.score:.2f}, "
                        f"at or above the transitivity override threshold "
                        f"{weights.transitivity_override_score:.2f}"
                    ),
                }
            )

        rule = merge_conflict(summaries[root_left], summaries[root_right], pair, weights)
        if rule is not None:
            strong = pair.score >= weights.conflict_override_score
            if not forced and not strong:
                outcome.blocked_merges.append(
                    {
                        "rule": rule,
                        "pair_id": pair.pair_id,
                        "left_id": pair.left_id,
                        "right_id": pair.right_id,
                        "score": round(pair.score, 4),
                        "reason": (
                            f"{rule}: merging these clusters would put conflicting "
                            f"evidence in one entity, and the pair only scores "
                            f"{pair.score:.2f} (the override threshold is "
                            f"{weights.conflict_override_score:.2f})"
                        ),
                    }
                )
                continue
            outcome.unresolved_conflicts.append(
                {
                    "rule": rule,
                    "pair_id": pair.pair_id,
                    "left_id": pair.left_id,
                    "right_id": pair.right_id,
                    "score": round(pair.score, 4),
                    "reason": (
                        "merged despite the conflict: confirmed by a reviewer"
                        if forced
                        else (
                            f"merged despite the conflict: the pair scores "
                            f"{pair.score:.2f}, at or above the override threshold "
                            f"{weights.conflict_override_score:.2f}"
                        )
                    ),
                }
            )

        new_root = union.union(root_left, root_right)
        merged = summaries[root_left].merged_with(summaries[root_right])
        summaries[new_root] = merged
        members[new_root] = members[root_left] | members[root_right]
        for stale in (root_left, root_right):
            if stale != new_root:
                del members[stale]
        _add_edge(graph, pair)

    pairs_by_id = {pair.pair_id: pair for pair in pairs}
    blocked_by_member: dict[str, str] = {}
    for blocked in outcome.blocked_merges:
        blocked_by_member.setdefault(blocked["left_id"], blocked["reason"])
        blocked_by_member.setdefault(blocked["right_id"], blocked["reason"])

    for members in union.groups().values():
        outcome.entities.append(
            build_entity(members, records, graph, pairs_by_id, blocked_by_member)
        )

    outcome.entities.sort(key=lambda e: (-e.size, e.canonical_name, e.entity_id))
    log.info(
        "clustered",
        extra={
            "records": len(records),
            "merge_edges": len(merge_edges),
            "entities": len(outcome.entities),
            "blocked_merges": len(outcome.blocked_merges),
        },
    )
    return outcome


def _add_edge(graph: nx.Graph, pair: PairScore) -> None:
    graph.add_edge(
        pair.left_id,
        pair.right_id,
        score=pair.score,
        override=pair.override,
        pair_id=pair.pair_id,
    )


def build_entity(
    members: list[str],
    records: dict[str, NormalizedRecord],
    graph: nx.Graph,
    pairs_by_id: dict[str, PairScore],
    blocked_by_member: dict[str, str] | None = None,
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
    if graph.number_of_nodes():
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

    split_reason = (blocked_by_member or {}).get(members[0]) if len(members) >= 1 else None
    if split_reason is None and blocked_by_member:
        for member in members:
            if member in blocked_by_member:
                split_reason = blocked_by_member[member]
                break

    review_state = "auto"
    if human_confirmed:
        review_state = "human_confirmed"
    elif split_reason:
        review_state = "conflict_separated"

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

    # most_complete (default): the most name tokens that are not bare
    # initials, then the spelling the cluster agrees on, then Latin script,
    # then length, with the record id as a deterministic tiebreak.
    #
    # The consensus term matters more than it looks. Without it the tiebreak
    # fell through to length, so a cluster of "Jayeshbhai Patel" / "જયેશ પટેલ"
    # / "Shri Jayesh R. Pattel" published **Pattel** as the canonical name —
    # the one spelling in the group that is a typo, chosen because the stray
    # middle initial made that record's display string the longest. The
    # canonical form is what downstream systems adopt, so it should be the
    # spelling the sources agree on, not the longest string among them.
    frequency = Counter(
        token for record in members for token in record.name.tokens if len(token) > 1
    )

    def consensus(record: NormalizedRecord) -> float:
        """Mean support for this record's tokens across the cluster.

        A mean, not a sum: a sum lets an extra token buy credit, so
        "Jayesh R. Pattel" tied with "Jayesh Patel" on the strength of the
        stray initial. Bare initials are excluded from the count entirely —
        they are not a spelling anyone agreed on.
        """
        tokens = [t for t in record.name.tokens if len(t) > 1]
        if not tokens:
            return 0.0
        return sum(frequency[t] for t in tokens) / len(tokens)

    return max(
        members,
        key=lambda r: (
            sum(1 for t in r.name.tokens if len(t) > 1),
            consensus(r),
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
