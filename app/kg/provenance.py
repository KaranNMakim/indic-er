"""Provenance for every graph fact.

The rule the ontology states and this module enforces: nothing is written to
the graph without ``source``, ``timestamp``, ``confidence`` and ``method``.
Without it an entity-resolution system is unauditable — you can see that two
records were merged but not why, by what, or how sure anyone was. For a
KYC-shaped use case that is the difference between a usable system and a
liability.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Method(StrEnum):
    """How a fact came to be believed."""

    RULE = "rule"
    MODEL = "model"
    HUMAN = "human"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Provenance(BaseModel):
    """Who said this, when, how, and how sure they were."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(description="Dataset, file or actor the fact came from")
    timestamp: datetime = Field(default_factory=utcnow)
    confidence: float = Field(ge=0.0, le=1.0)
    method: Method
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific payload: feature contributions, reviewer id, rule name",
    )

    @field_validator("source")
    @classmethod
    def _source_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provenance.source must not be blank")
        return value.strip()

    @classmethod
    def from_rule(cls, source: str, confidence: float, rule: str, **detail: Any) -> Provenance:
        return cls(
            source=source,
            confidence=confidence,
            method=Method.RULE,
            detail={"rule": rule, **detail},
        )

    @classmethod
    def from_human(cls, reviewer: str, decision: str, **detail: Any) -> Provenance:
        return cls(
            source=f"reviewer:{reviewer}",
            confidence=1.0,
            method=Method.HUMAN,
            detail={"decision": decision, "reviewer": reviewer, **detail},
        )

    @classmethod
    def from_model(cls, source: str, confidence: float, model: str, **detail: Any) -> Provenance:
        return cls(
            source=source,
            confidence=confidence,
            method=Method.MODEL,
            detail={"model": model, **detail},
        )

    def as_row(self) -> dict[str, Any]:
        """Flat form for storage."""
        return {
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "method": str(self.method),
            "detail": self.detail,
        }

    def describe(self) -> str:
        """One line for a human reading an audit trail."""
        when = self.timestamp.strftime("%Y-%m-%d %H:%M:%SZ")
        label = {
            Method.RULE: self.detail.get("rule", "rule"),
            Method.HUMAN: f"{self.detail.get('reviewer', 'reviewer')} "
            f"{self.detail.get('decision', '')}".strip(),
            Method.MODEL: self.detail.get("model", "model"),
        }[self.method]
        return f"[{self.method}] {label} — {self.source}, conf {self.confidence:.2f}, {when}"
