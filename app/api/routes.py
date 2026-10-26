"""API endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    MAX_PAIRS_IN_RESPONSE,
    EntityResponse,
    MatchRequest,
    MatchResponse,
    MetricsResponse,
    ResolveRequest,
    ResolveResponse,
    ReviewDecisionRequest,
    ReviewQueueResponse,
)
from app.api.state import AppState, get_state
from app.core.cluster import find_conflicts
from app.core.explain import feature_table, highlights
from app.core.models import CanonicalEntity
from app.core.pipeline import match_two, resolve

log = logging.getLogger(__name__)
router = APIRouter()


def _weights_for(state: AppState, auto_merge: float | None, review_low: float | None):
    if auto_merge is None and review_low is None:
        return state.weights
    return state.weights.with_overrides(auto_merge=auto_merge, review_low=review_low)


# --------------------------------------------------------------------------- #
# resolve
# --------------------------------------------------------------------------- #


@router.post(
    "/resolve",
    response_model=ResolveResponse,
    tags=["resolution"],
    summary="Resolve a batch of records into canonical entities",
)
def resolve_endpoint(request: ResolveRequest) -> ResolveResponse:
    """Run the full pipeline over a batch.

    Stateless by default: nothing is written to the graph, so the playground
    can run experiments without disturbing the seeded demo. Set
    ``enqueue_review`` to push borderline pairs into the persistent queue.
    """
    state = get_state()
    weights = _weights_for(state, request.auto_merge, request.review_low)
    overrides = state.review.overrides() if request.apply_reviews else None

    try:
        result = resolve(
            request.records,
            weights,
            overrides=overrides,
            explanations=request.explanations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    decided = [p for p in result.pairs if p.decision in {"merge", "review"}]
    decided.sort(key=lambda p: -p.score)
    truncated = len(decided) > MAX_PAIRS_IN_RESPONSE

    enqueued = 0
    if request.enqueue_review:
        enqueued = state.review.enqueue(result.review_pairs)

    return ResolveResponse(
        entities=[entity.to_dict() for entity in result.entities],
        pairs=[pair.to_dict() for pair in decided[:MAX_PAIRS_IN_RESPONSE]],
        review_pairs=[pair.to_dict() for pair in result.review_pairs[:MAX_PAIRS_IN_RESPONSE]],
        stats=result.stats,
        normalized=(
            {k: v.to_dict() for k, v in result.normalized.items()}
            if request.include_normalized
            else None
        ),
        truncated_pairs=truncated,
        enqueued=enqueued,
    )


@router.post(
    "/match",
    response_model=MatchResponse,
    tags=["resolution"],
    summary="Score two records, with a per-feature explanation",
)
def match_endpoint(request: MatchRequest) -> MatchResponse:
    """Score exactly two records.

    Bypasses blocking, so it answers for any two records the caller asks
    about — including a pair blocking would never have proposed, which is
    usually the interesting question.
    """
    state = get_state()
    try:
        pair, left, right = match_two(request.left, request.right, state.weights)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    payload = pair.to_dict()
    response = MatchResponse(
        pair=payload,
        explanation=pair.explanation,
        features=feature_table(pair),
        highlights=highlights(pair, left, right),
        left=left.to_dict(),
        right=right.to_dict(),
        thresholds={
            "auto_merge": state.weights.auto_merge,
            "review_low": state.weights.review_low,
        },
    )

    if request.explain_with_llm:
        llm_response = state.llm.explain_pair(payload)
        response.llm_explanation = {
            "text": llm_response.text,
            "model": llm_response.model,
            "mode": llm_response.mode,
            "advisory": True,
        }
    if request.adjudicate:
        verdict = state.llm.adjudicate_pair(payload)
        response.adjudication = {
            "verdict": verdict.verdict,
            "confidence": verdict.confidence,
            "rationale": verdict.rationale,
            "model": verdict.model,
            "mode": verdict.mode,
            "advisory": True,
        }
    return response


# --------------------------------------------------------------------------- #
# entities
# --------------------------------------------------------------------------- #


@router.get("/entities", tags=["entities"], summary="List the seeded entities")
def list_entities(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_size: int = Query(default=1, ge=1),
    kind: str | None = Query(default=None, pattern="^(person|business)$"),
    q: str | None = Query(default=None, max_length=120),
) -> dict[str, Any]:
    state = get_state()
    result = state.ensure_resolved()
    entities: list[CanonicalEntity] = [e for e in result.entities if e.size >= min_size]
    if kind:
        entities = [e for e in entities if e.kind == kind]
    if q:
        needle = q.lower()
        entities = [
            e
            for e in entities
            if needle in e.canonical_name.lower()
            or any(needle in variant.lower() for variant in e.variants)
        ]
    return {
        "total": len(entities),
        "limit": limit,
        "offset": offset,
        "entities": [e.to_dict() for e in entities[offset : offset + limit]],
    }


@router.get(
    "/entities/{entity_id}",
    response_model=EntityResponse,
    tags=["entities"],
    summary="One canonical entity, with its members, evidence and provenance",
)
def get_entity(entity_id: str) -> EntityResponse:
    """Accepts an entity id or the id of any record inside it."""
    state = get_state()
    result = state.ensure_resolved()
    entity = state.find_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"no entity or record {entity_id!r}")

    members = [
        result.normalized[member].to_dict()
        for member in entity.member_ids
        if member in result.normalized
    ]
    conflicts = [
        {"left_id": left, "right_id": right, "rule": rule}
        for left, right, rule in find_conflicts(
            list(entity.member_ids), result.normalized, state.weights
        )
    ]
    node_ids = [*entity.member_ids, entity.entity_id]
    provenance = []
    for node_id in node_ids:
        node = state.graph.get_node(node_id)
        if node is not None:
            provenance.append(
                {"node": node_id, "class": node.class_name, **node.provenance.as_row()}
            )

    return EntityResponse(
        entity=entity.to_dict(),
        members=members,
        evidence=list(entity.merge_evidence),
        conflicts=conflicts,
        graph=state.graph.cytoscape(node_ids),
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# review
# --------------------------------------------------------------------------- #


@router.get(
    "/review-queue",
    response_model=ReviewQueueResponse,
    tags=["review"],
    summary="Borderline pairs awaiting a human decision",
)
def review_queue(
    state_filter: str = Query(default="pending", alias="state"),
    limit: int | None = Query(default=None, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    order: str | None = Query(default=None, pattern="^(uncertainty|score_desc|score_asc)$"),
) -> ReviewQueueResponse:
    state = get_state()
    state.ensure_resolved()
    weights = state.weights
    items = state.review.list_items(
        state=state_filter, limit=limit, offset=offset, order=order, weights=weights
    )
    enriched: list[dict[str, Any]] = []
    result = state.result
    for item in items:
        payload = item.to_dict()
        if result is not None:
            left = result.normalized.get(item.left_id)
            right = result.normalized.get(item.right_id)
            payload["left"] = left.to_dict() if left else None
            payload["right"] = right.to_dict() if right else None
        enriched.append(payload)

    return ReviewQueueResponse(
        items=enriched,
        counts=state.review.counts(),
        threshold_shift=state.review.threshold_shift(weights),
        page_size=limit or weights.review_page_size,
        offset=offset,
        order=order or weights.review_order,
    )


@router.post(
    "/review/{pair_id}",
    tags=["review"],
    summary="Accept or reject a borderline pair",
)
def decide_review(pair_id: str, request: ReviewDecisionRequest) -> dict[str, Any]:
    """Record a decision and re-resolve, so the effect is immediate.

    The decision becomes a hard override applied *before* clustering on the
    next run, which is what makes review work compound rather than accumulate
    as a pile of annotations.
    """
    state = get_state()
    try:
        item = state.review.decide(pair_id, request.decision, request.reviewer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = state.resolve_seed(enqueue=False)
    return {
        "decided": item.to_dict(),
        "counts": state.review.counts(),
        "threshold_shift": state.review.threshold_shift(state.weights),
        "entities": result.stats["clustering"],
        "overrides_applied": result.stats["overrides_applied"],
    }


@router.post("/review/{pair_id}/undo", tags=["review"], summary="Undo a decision")
def undo_review(pair_id: str) -> dict[str, Any]:
    state = get_state()
    if state.review.get(pair_id) is None:
        raise HTTPException(status_code=404, detail=f"pair not in queue: {pair_id}")
    state.review.undo(pair_id)
    state.resolve_seed(enqueue=False)
    return {"counts": state.review.counts()}


# --------------------------------------------------------------------------- #
# metrics and demo data
# --------------------------------------------------------------------------- #


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["meta"],
    summary="Live stats for the seeded data, plus the full evaluation results",
)
def metrics() -> MetricsResponse:
    """Two different things, kept clearly apart.

    *Seeded* numbers describe the live demo dataset in this process.
    *Evaluation* numbers come from `make eval` over the full 5,000-record set
    with ground truth, and are the ones worth quoting.
    """
    state = get_state()
    result = state.ensure_resolved()
    evaluation = state.evaluation()
    return MetricsResponse(
        seeded={
            "source": state.seed_source,
            "records": len(state.records),
            **{
                key: result.stats[key]
                for key in (
                    "blocking",
                    "pairs_scored",
                    "pairs_merged",
                    "pairs_review",
                    "pairs_rejected",
                    "clustering",
                    "duplicate_rate",
                    "overrides_applied",
                    "timings_ms",
                )
            },
        },
        review={
            "counts": state.review.counts(),
            "threshold_shift": state.review.threshold_shift(state.weights),
        },
        thresholds={
            "auto_merge": state.weights.auto_merge,
            "review_low": state.weights.review_low,
        },
        features=[
            {
                "name": spec.name,
                "weight": spec.weight,
                "description": spec.description,
            }
            for spec in state.weights.features.values()
        ],
        evaluation=evaluation,
        evaluation_available=evaluation is not None,
        note=(
            ""
            if evaluation is not None
            else "Run `make eval` to populate the ground-truth comparison."
        ),
    )


@router.get("/showcase", tags=["meta"], summary="The curated demo records")
def showcase() -> dict[str, Any]:
    """Fourteen hand-built records that exercise every hard case at once.

    What the playground loads when you press "Load example".
    """
    state = get_state()
    records = state.showcase_records()
    return {
        "records": [record.model_dump(by_alias=True, exclude_none=False) for record in records],
        "count": len(records),
    }


@router.get("/seed", tags=["meta"], summary="The seeded dataset and its resolution")
def seed_info() -> dict[str, Any]:
    state = get_state()
    result = state.ensure_resolved()
    return {
        "source": state.seed_source,
        "records": len(state.records),
        "entities": len(result.entities),
        "stats": result.stats,
        "graph": state.graph.stats(),
    }


@router.post("/seed/resolve", tags=["meta"], summary="Re-run the seeded resolution")
def reseed() -> dict[str, Any]:
    state = get_state()
    result = state.resolve_seed()
    return {"entities": len(result.entities), "stats": result.stats["clustering"]}


@router.get("/graph/{entity_id}", tags=["entities"], summary="Cytoscape elements")
def entity_graph(entity_id: str) -> dict[str, Any]:
    state = get_state()
    state.ensure_resolved()
    entity = state.find_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"no entity {entity_id!r}")
    return state.graph.cytoscape([*entity.member_ids, entity.entity_id])
