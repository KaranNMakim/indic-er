"""Explanations.

Every match returns per-feature contributions and a sentence a non-technical
reviewer can act on. This is rule-based and deterministic — the same pair
always produces the same words, with no model involved — because the
explanation has to be trustworthy in an audit, and because the demo must read
well with no API key.

``LLMClient.explain_pair`` can rephrase this more fluently when
``LLM_MODE=live``, but it never changes the decision or the numbers.
"""

from __future__ import annotations

from typing import Any

from app.core.config import WeightsConfig, weights_config
from app.core.models import FeatureValue, NormalizedRecord, PairScore

VERDICT = {
    "merge": "Same entity",
    "review": "Needs a human look",
    "reject": "Different entities",
}

FEATURE_LABEL = {
    "name_string": "name spelling",
    "name_phonetic": "name sound",
    "name_rarity": "name distinctiveness",
    "address_core": "address",
    "house_number": "house number",
    "pincode": "pincode",
    "landmark": "landmark",
    "phone": "phone",
}


def explain(
    pair: PairScore,
    left: NormalizedRecord,
    right: NormalizedRecord,
    weights: WeightsConfig | None = None,
) -> str:
    """A plain-language explanation of one pair's outcome."""
    weights = weights or weights_config()
    usable = [f for f in pair.features.values() if f.value is not None]
    missing = [f for f in pair.features.values() if f.value is None]

    if pair.override:
        who = "accepted" if pair.override == "human_accept" else "rejected"
        return (
            f"A reviewer {who} this pair, so the human decision stands. "
            f"The model had scored it {pair.score:.2f} "
            f"({_band_name(pair.score, weights)})."
        )

    if not usable:
        return "These two records share no comparable field, so there is nothing to match on."

    strong = sorted(usable, key=lambda f: f.contribution, reverse=True)
    # Order the supporting evidence the way a reviewer weighs it, not the way
    # the model weights it. An exact phone or pincode match is a hard
    # identifier: it carries a modest weight because it is often missing, but
    # when it does agree it is the most convincing line in the explanation and
    # must not be pushed out of the list by three middling name features.
    supporting = sorted(
        (f for f in strong if (f.value or 0) >= 0.75),
        key=lambda f: (_is_hard_identifier(f), f.contribution),
        reverse=True,
    )
    against = [f for f in strong if (f.value or 0) <= 0.35]

    parts: list[str] = [
        f"{VERDICT[pair.decision]} — score {pair.score:.2f}, {_band_name(pair.score, weights)}."
    ]

    if supporting:
        parts.append("In favour: " + _join(_phrase(f, left, right) for f in supporting[:3]) + ".")
    if against:
        parts.append("Against: " + _join(_phrase(f, left, right) for f in against[:3]) + ".")
    if not supporting and not against:
        parts.append(
            "Every signal is middling: " + _join(_phrase(f, left, right) for f in strong[:2]) + "."
        )

    if missing:
        names = _join(FEATURE_LABEL.get(f.name, f.name) for f in missing)
        parts.append(
            f"Not comparable on {names}, so {'it was' if len(missing) == 1 else 'they were'} "
            "left out of the score rather than counted against the pair."
        )

    if pair.decision == "review":
        parts.append(
            f"This sits in the review band "
            f"({weights.review_low:.2f}–{weights.auto_merge:.2f}), where a "
            "reviewer's decision is worth more than a threshold tweak."
        )

    return " ".join(parts)


HARD_IDENTIFIERS = frozenset({"phone", "pincode"})


def _is_hard_identifier(feature: FeatureValue) -> bool:
    """A field that identifies rather than merely resembles, and that agrees."""
    return feature.name in HARD_IDENTIFIERS and (feature.value or 0) >= 1.0


def _band_name(score: float, weights: WeightsConfig) -> str:
    if score >= weights.auto_merge:
        return f"at or above the auto-merge threshold of {weights.auto_merge:.2f}"
    if score >= weights.review_low:
        return f"inside the review band starting at {weights.review_low:.2f}"
    return f"below the review band floor of {weights.review_low:.2f}"


def _phrase(feature: FeatureValue, left: NormalizedRecord, right: NormalizedRecord) -> str:
    label = FEATURE_LABEL.get(feature.name, feature.name)
    value = feature.value or 0.0

    if feature.name == "phone":
        return (
            f"the same phone number (…{left.phone_last10[-4:]})"
            if value >= 1.0
            else "different phone numbers"
        )
    if feature.name == "pincode":
        if value >= 1.0:
            pin = left.address.pincode if left.address else ""
            return f"the same pincode ({pin})"
        if value >= 0.5:
            return "different pincodes in the same sorting district"
        return "different pincodes"
    if feature.name == "name_phonetic" and value >= 0.95:
        matched = ", ".join(f"{a}/{b}" for a, b in feature.matched_tokens[:2])
        return f"the names sound identical ({matched})" if matched else "the names sound identical"
    if feature.name == "name_rarity" and value < 0.5:
        return "the only shared name token is a very common surname"
    if feature.name == "landmark" and value >= 0.8:
        mark = left.address.landmark if left.address else ""
        return f"the same landmark ({mark})"
    if feature.name == "address_core" and value >= 0.8:
        return "the addresses agree once abbreviations are expanded"

    return f"{label} at {value:.0%}"


def _join(items) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def highlights(pair: PairScore, left: NormalizedRecord, right: NormalizedRecord) -> dict[str, Any]:
    """Which tokens the UI should highlight on each side, and why.

    Returned as ``{"left": {token: feature}, "right": {...}}`` so the
    playground can colour the exact tokens that carried the match instead of
    showing a bare number.
    """
    left_tokens: dict[str, str] = {}
    right_tokens: dict[str, str] = {}
    for feature in pair.features.values():
        if feature.value is None or not feature.matched_tokens:
            continue
        for a, b in feature.matched_tokens:
            left_tokens.setdefault(a, feature.name)
            right_tokens.setdefault(b, feature.name)
    return {
        "left": left_tokens,
        "right": right_tokens,
        "left_name": left.name.display,
        "right_name": right.name.display,
    }


def feature_table(pair: PairScore) -> list[dict[str, Any]]:
    """Per-feature contributions, ordered by how much they mattered."""
    rows = [
        {
            "name": feature.name,
            "label": FEATURE_LABEL.get(feature.name, feature.name),
            "value": feature.value,
            "weight": feature.weight,
            "contribution": round(feature.contribution, 4),
            "detail": feature.detail,
            "computed": feature.value is not None,
        }
        for feature in pair.features.values()
    ]
    rows.sort(key=lambda row: (row["computed"] is False, -row["contribution"]))
    return rows
