"""Request and response models for the API.

Response shapes are capped deliberately. A 5,000-record resolve produces
~300,000 scored pairs, and serialising all of them would make the endpoint
unusable for the thing it exists to do. Merged and review-band pairs are
returned (they are the decisions); rejected pairs are counted, not listed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.models import InputRecord

MAX_RESOLVE_RECORDS = 6000
MAX_PAIRS_IN_RESPONSE = 2000


class ResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[InputRecord] = Field(min_length=1)
    include_normalized: bool = Field(
        default=True,
        description="Return the normalization trace for each record (the web app shows it)",
    )
    apply_reviews: bool = Field(
        default=True, description="Apply stored human decisions as overrides"
    )
    enqueue_review: bool = Field(
        default=False, description="Add review-band pairs to the persistent queue"
    )
    explanations: bool = Field(default=True)
    auto_merge: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Override the merge threshold for this call"
    )
    review_low: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("records")
    @classmethod
    def _not_too_many(cls, value: list[InputRecord]) -> list[InputRecord]:
        if len(value) > MAX_RESOLVE_RECORDS:
            raise ValueError(
                f"at most {MAX_RESOLVE_RECORDS} records per call; "
                f"got {len(value)}. Split the batch."
            )
        return value

    @field_validator("records")
    @classmethod
    def _unique_ids(cls, value: list[InputRecord]) -> list[InputRecord]:
        seen: set[str] = set()
        for record in value:
            if record.record_id in seen:
                raise ValueError(f"duplicate record id: {record.record_id!r}")
            seen.add(record.record_id)
        return value


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: InputRecord
    right: InputRecord
    explain_with_llm: bool = Field(
        default=False,
        description=(
            "Also return an LLM-phrased explanation. Advisory only — it never "
            "changes the score or the decision. Deterministic in LLM_MODE=mock."
        ),
    )
    adjudicate: bool = Field(
        default=False,
        description="Ask the LLM for a second opinion. Advisory only.",
    )


class ReviewDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "reject"]
    reviewer: str = Field(default="demo", max_length=64)


class FeatureRow(BaseModel):
    name: str
    label: str
    value: float | None
    weight: float
    contribution: float
    detail: str
    computed: bool


class MatchResponse(BaseModel):
    pair: dict[str, Any]
    explanation: str
    features: list[FeatureRow]
    highlights: dict[str, Any]
    left: dict[str, Any]
    right: dict[str, Any]
    thresholds: dict[str, float]
    llm_explanation: dict[str, Any] | None = None
    adjudication: dict[str, Any] | None = None


class ResolveResponse(BaseModel):
    entities: list[dict[str, Any]]
    pairs: list[dict[str, Any]]
    review_pairs: list[dict[str, Any]]
    stats: dict[str, Any]
    normalized: dict[str, Any] | None = None
    truncated_pairs: bool = False
    enqueued: int = 0


class EntityResponse(BaseModel):
    entity: dict[str, Any]
    members: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    graph: dict[str, Any]
    provenance: list[dict[str, Any]]


class ReviewQueueResponse(BaseModel):
    items: list[dict[str, Any]]
    counts: dict[str, int]
    threshold_shift: dict[str, Any]
    page_size: int
    offset: int
    order: str


class MetricsResponse(BaseModel):
    seeded: dict[str, Any]
    review: dict[str, Any]
    thresholds: dict[str, float]
    features: list[dict[str, Any]]
    evaluation: dict[str, Any] | None = None
    evaluation_available: bool = False
    note: str = ""
