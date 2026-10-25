"""Record models: what goes in, what normalization produces, what comes out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EntityKind = Literal["person", "business"]


class InputRecord(BaseModel):
    """One row from a source system.

    Everything but ``name`` is optional, because real source data is patchy —
    and a missing field must never be scored as a disagreement.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    record_id: str = Field(alias="id", description="Stable id within the batch")
    name: str
    kind: EntityKind = "person"
    address: str | None = None
    phone: str | None = None
    identifier_scheme: str | None = None
    identifier_masked: str | None = None
    source: str = "input"

    @field_validator("record_id", "name", mode="before")
    @classmethod
    def _stringify(cls, value: Any) -> Any:
        return str(value) if value is not None else value

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be blank")
        return value.strip()

    @field_validator("identifier_masked")
    @classmethod
    def _identifier_must_be_masked(cls, value: str | None) -> str | None:
        """Refuse to accept an unmasked identifier.

        indic-er is not a place to put a PAN or an Aadhaar number. If a caller
        sends something that does not look masked, the request fails rather
        than quietly storing it.
        """
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if not any(ch in stripped for ch in "X*x"):
            raise ValueError(
                "identifier_masked must be masked (e.g. 'XXXXX1234'); "
                "indic-er never stores a full identifier"
            )
        return stripped


@dataclass(slots=True)
class ParsedAddress:
    """An Indian address, pulled apart into the bits that carry signal."""

    raw: str
    house: str | None = None
    building: str | None = None
    landmark: str | None = None
    locality: str | None = None
    city: str | None = None
    state: str | None = None
    pincode: str | None = None
    core_tokens: tuple[str, ...] = ()
    expanded: str = ""

    @property
    def district_prefix(self) -> str | None:
        return self.pincode[:3] if self.pincode else None

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.house, self.building, self.landmark, self.locality, self.city, self.pincode)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "house": self.house,
            "building": self.building,
            "landmark": self.landmark,
            "locality": self.locality,
            "city": self.city,
            "state": self.state,
            "pincode": self.pincode,
            "core_tokens": list(self.core_tokens),
            "expanded": self.expanded,
        }


@dataclass(slots=True)
class NormalizedName:
    """The result of the name pipeline, with every step kept for the UI."""

    raw: str
    script: str
    transliterated: str
    tokens: tuple[str, ...] = ()
    phonetic_tokens: tuple[str, ...] = ()
    phonetic_key: str = ""
    honorifics_removed: tuple[str, ...] = ()
    suffixes_removed: tuple[str, ...] = ()
    legal_suffix: str | None = None

    @property
    def display(self) -> str:
        return " ".join(t.title() for t in self.tokens) or self.transliterated

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "script": self.script,
            "transliterated": self.transliterated,
            "tokens": list(self.tokens),
            "phonetic_tokens": list(self.phonetic_tokens),
            "phonetic_key": self.phonetic_key,
            "honorifics_removed": list(self.honorifics_removed),
            "suffixes_removed": list(self.suffixes_removed),
            "legal_suffix": self.legal_suffix,
            "display": self.display,
        }


@dataclass(slots=True)
class NormalizedRecord:
    """A source record plus everything normalization derived from it."""

    record_id: str
    kind: EntityKind
    source: str
    raw_name: str
    name: NormalizedName
    address: ParsedAddress | None = None
    phone_raw: str | None = None
    phone_last10: str | None = None
    identifier_scheme: str | None = None
    identifier_masked: str | None = None

    @property
    def phonetic_key(self) -> str:
        return self.name.phonetic_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "kind": self.kind,
            "source": self.source,
            "raw_name": self.raw_name,
            "name": self.name.to_dict(),
            "address": self.address.to_dict() if self.address else None,
            "phone_raw": self.phone_raw,
            "phone_last10": self.phone_last10,
            "identifier_scheme": self.identifier_scheme,
            "identifier_masked": self.identifier_masked,
        }


@dataclass(slots=True)
class FeatureValue:
    """One feature's contribution to a pair score."""

    name: str
    value: float | None
    weight: float
    detail: str = ""
    matched_tokens: tuple[tuple[str, str], ...] = ()

    @property
    def contribution(self) -> float:
        """Weighted contribution, 0 when the feature could not be computed."""
        return 0.0 if self.value is None else self.value * self.weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "weight": self.weight,
            "contribution": self.contribution,
            "detail": self.detail,
            "matched_tokens": [list(pair) for pair in self.matched_tokens],
        }


Decision = Literal["merge", "review", "reject"]


@dataclass(slots=True)
class PairScore:
    """A scored candidate pair."""

    left_id: str
    right_id: str
    score: float
    decision: Decision
    features: dict[str, FeatureValue] = field(default_factory=dict)
    blocked_by: tuple[str, ...] = ()
    override: str | None = None
    explanation: str = ""

    @property
    def pair_id(self) -> str:
        left, right = sorted((self.left_id, self.right_id))
        return f"{left}::{right}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "score": round(self.score, 4),
            "decision": self.decision,
            "features": {name: fv.to_dict() for name, fv in self.features.items()},
            "blocked_by": list(self.blocked_by),
            "override": self.override,
            "explanation": self.explanation,
        }


@dataclass(slots=True)
class CanonicalEntity:
    """A resolved entity: one real-world person or business."""

    entity_id: str
    kind: EntityKind
    canonical_name: str
    member_ids: tuple[str, ...]
    cluster_score: float
    review_state: str = "auto"
    canonical_address: ParsedAddress | None = None
    canonical_phone: str | None = None
    variants: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    merge_evidence: tuple[dict[str, Any], ...] = ()
    split_reason: str | None = None

    @property
    def size(self) -> int:
        return len(self.member_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "canonical_name": self.canonical_name,
            "member_ids": list(self.member_ids),
            "size": self.size,
            "cluster_score": round(self.cluster_score, 4),
            "review_state": self.review_state,
            "canonical_address": (
                self.canonical_address.to_dict() if self.canonical_address else None
            ),
            "canonical_phone": self.canonical_phone,
            "variants": list(self.variants),
            "scripts": list(self.scripts),
            "sources": list(self.sources),
            "merge_evidence": list(self.merge_evidence),
            "split_reason": self.split_reason,
        }


@dataclass(slots=True)
class ResolutionResult:
    """Everything one ``/resolve`` call produced."""

    entities: list[CanonicalEntity]
    pairs: list[PairScore]
    review_pairs: list[PairScore]
    normalized: dict[str, NormalizedRecord]
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_normalized: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "entities": [e.to_dict() for e in self.entities],
            "pairs": [p.to_dict() for p in self.pairs],
            "review_pairs": [p.to_dict() for p in self.review_pairs],
            "stats": self.stats,
        }
        if include_normalized:
            payload["normalized"] = {k: v.to_dict() for k, v in self.normalized.items()}
        return payload
