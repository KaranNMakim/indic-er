"""The resolution pipeline.

    normalize -> block -> score -> cluster -> canonicalise

One function, ``resolve``, runs the whole thing and returns everything the API
and the web app need, including the intermediate detail — normalization steps,
candidate pairs, per-feature contributions, conflict splits — because a POV
that only returns answers cannot be evaluated or trusted.

No LLM, no network, no randomness: the same input always produces the same
output.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.core.blocking import candidate_pairs, describe_strategies
from app.core.cluster import cluster_records
from app.core.config import WeightsConfig, weights_config
from app.core.explain import explain
from app.core.features import score_pair
from app.core.models import InputRecord, NormalizedRecord, PairScore, ResolutionResult
from app.core.normalize import normalize_records

log = logging.getLogger(__name__)


def resolve(
    records: list[InputRecord],
    weights: WeightsConfig | None = None,
    overrides: dict[str, str] | None = None,
    *,
    explanations: bool = True,
) -> ResolutionResult:
    """Resolve a batch of records into canonical entities.

    ``overrides`` maps ``pair_id`` to ``"accept"`` or ``"reject"`` and comes
    from the review queue. It is applied before clustering, so a reviewer's
    decision shapes the entities rather than being annotated onto them.
    """
    weights = weights or weights_config()
    started = time.perf_counter()

    normalized = normalize_records(records)
    t_normalized = time.perf_counter()

    blocking = candidate_pairs(normalized, weights)
    t_blocked = time.perf_counter()

    pairs: list[PairScore] = []
    for (left_id, right_id), strategies in blocking.pairs.items():
        pair = score_pair(
            normalized[left_id],
            normalized[right_id],
            weights,
            blocked_by=describe_strategies(strategies),
        )
        pairs.append(pair)
    t_scored = time.perf_counter()

    outcome = cluster_records(normalized, pairs, weights, overrides)
    t_clustered = time.perf_counter()

    if explanations:
        for pair in pairs:
            if pair.decision != "reject" or pair.override:
                pair.explanation = explain(
                    pair, normalized[pair.left_id], normalized[pair.right_id], weights
                )

    review_pairs = sorted(
        (p for p in pairs if p.decision == "review"),
        key=lambda p: -p.score,
    )
    merged_pairs = [p for p in pairs if p.decision == "merge"]

    stats: dict[str, Any] = {
        "input_records": len(records),
        "blocking": blocking.stats(len(normalized)),
        "pairs_scored": len(pairs),
        "pairs_merged": len(merged_pairs),
        "pairs_review": len(review_pairs),
        "pairs_rejected": len(pairs) - len(merged_pairs) - len(review_pairs),
        "clustering": outcome.stats(),
        "thresholds": {
            "auto_merge": weights.auto_merge,
            "review_low": weights.review_low,
        },
        "overrides_applied": sum(1 for p in pairs if p.override),
        "duplicate_rate": _duplicate_rate(len(normalized), len(outcome.entities)),
        "blocked_merges": outcome.blocked_merges[:200],
        "unresolved_conflicts": outcome.unresolved_conflicts,
        "timings_ms": {
            "normalize": round((t_normalized - started) * 1000, 1),
            "block": round((t_blocked - t_normalized) * 1000, 1),
            "score": round((t_scored - t_blocked) * 1000, 1),
            "cluster": round((t_clustered - t_scored) * 1000, 1),
            "total": round((time.perf_counter() - started) * 1000, 1),
        },
    }

    log.info(
        "resolved",
        extra={
            "records": len(records),
            "candidate_pairs": blocking.pair_count,
            "entities": len(outcome.entities),
            "merged": stats["clustering"]["merged_entities"],
            "review": len(review_pairs),
            "ms": stats["timings_ms"]["total"],
        },
    )

    return ResolutionResult(
        entities=outcome.entities,
        pairs=pairs,
        review_pairs=review_pairs,
        normalized=normalized,
        stats=stats,
    )


def _duplicate_rate(record_count: int, entity_count: int) -> float:
    """Share of records that were duplicates of something else.

    The headline number a stakeholder actually asks for: "how much of my data
    is the same thing twice?"
    """
    if record_count <= 0:
        return 0.0
    return round(1 - (entity_count / record_count), 4)


def match_two(
    left: InputRecord,
    right: InputRecord,
    weights: WeightsConfig | None = None,
) -> tuple[PairScore, NormalizedRecord, NormalizedRecord]:
    """Score exactly two records, bypassing blocking.

    ``/match`` must answer for any two records the caller asks about, even a
    pair that blocking would never have proposed — otherwise the endpoint
    silently returns nothing for the most interesting questions.
    """
    weights = weights or weights_config()
    if left.record_id == right.record_id:
        raise ValueError("match needs two different record ids")
    normalized = normalize_records([left, right])
    left_norm = normalized[left.record_id]
    right_norm = normalized[right.record_id]
    pair = score_pair(left_norm, right_norm, weights, blocked_by=("direct",))
    pair.explanation = explain(pair, left_norm, right_norm, weights)
    return pair, left_norm, right_norm
