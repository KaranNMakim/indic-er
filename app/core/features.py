"""Pairwise features.

Two rules govern everything here.

**A missing field is not a disagreement.** Every feature returns ``None`` when
it cannot be computed, and ``None`` features are dropped before the weighted
mean, so the remaining weights renormalise. A logistics record with no phone
must not score worse than one with a *wrong* phone. Treating absence as zero
is the most common way a scoring model quietly becomes useless on real data,
where a third of the fields are blank.

**Token alignment, not string distance.** Indian names reorder, gain and lose
middle names, and abbreviate to initials constantly:

    "Jayesh R. Patel"  /  "Patel Jayesh"  /  "J. Patel"

A raw edit distance between those strings is noise. So names are compared
token to token, best-match greedily, scored on the *shorter* side (an extra
middle name is not evidence against a match) with a coverage penalty when one
name is substantially longer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rapidfuzz.distance import JaroWinkler

from app.core.config import WeightsConfig, names_config, weights_config
from app.core.models import Decision, FeatureValue, NormalizedRecord, PairScore
from app.core.phonetic import indic_key

# An initial matching a full token ("J." vs "Jayesh") is strong evidence, but
# not as strong as the full token agreeing.
INITIAL_MATCH = 0.90

# Below this, two aligned tokens are not considered "the same token" when
# deciding which tokens to highlight or whether a match rests only on a
# common surname.
TOKEN_MATCH_FLOOR = 0.88

# A name agreement carried only by Patel/Shah/Kumar scores this instead of 1.0.
COMMON_ONLY_RARITY = 0.22


@dataclass(frozen=True, slots=True)
class Alignment:
    """The result of matching two token lists against each other."""

    score: float
    pairs: tuple[tuple[str, str, float], ...]

    @property
    def matched(self) -> tuple[tuple[str, str], ...]:
        return tuple((a, b) for a, b, s in self.pairs if s >= TOKEN_MATCH_FLOOR)


# --------------------------------------------------------------------------- #
# token scorers
# --------------------------------------------------------------------------- #


def _string_token_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if len(left) == 1 or len(right) == 1:
        # An initial against a full name: only the first letter can agree.
        return INITIAL_MATCH if left[0] == right[0] else 0.0
    return JaroWinkler.similarity(left, right)


def _phonetic_token_score(left: str, right: str) -> float:
    key_left, key_right = indic_key(left), indic_key(right)
    if not key_left or not key_right:
        return 0.0
    if key_left == key_right:
        return 1.0
    if len(left) == 1 or len(right) == 1:
        return INITIAL_MATCH if key_left[0] == key_right[0] else 0.0
    # Partial credit: "Subramanian"/"Subramaniam" differ in one final class.
    return JaroWinkler.similarity(key_left, key_right)


def align_tokens(
    left: Sequence[str],
    right: Sequence[str],
    scorer: Callable[[str, str], float],
    penalty: Callable[[Sequence[str], Sequence[str]], float] | None = None,
) -> Alignment:
    """Greedy best-match alignment, scored on the shorter token list.

    Greedy rather than optimal (Hungarian) because names are 2-4 tokens long,
    the two agree in practice, and greedy is far easier to explain to someone
    asking why a pair matched.

    ``penalty`` discounts the result when the shorter side carries too little
    to be convincing. Names and addresses need different discounts — see
    :func:`_coverage_penalty` and :func:`_informativeness_penalty`.
    """
    if not left or not right:
        return Alignment(score=0.0, pairs=())

    short, long = (left, right) if len(left) <= len(right) else (right, left)
    flipped = short is right

    remaining = list(long)
    pairs: list[tuple[str, str, float]] = []
    for token in short:
        if not remaining:
            break
        best_index, best_score = 0, -1.0
        for index, candidate in enumerate(remaining):
            value = scorer(token, candidate)
            if value > best_score:
                best_index, best_score = index, value
        partner = remaining.pop(best_index)
        pair = (partner, token) if flipped else (token, partner)
        pairs.append((pair[0], pair[1], best_score))

    if not pairs:
        return Alignment(score=0.0, pairs=())

    mean = sum(score for _, _, score in pairs) / len(pairs)
    discount = (penalty or _coverage_penalty)(left, right)
    return Alignment(score=mean * discount, pairs=tuple(pairs))


def _coverage_penalty(left: Sequence[str], right: Sequence[str]) -> float:
    """For names: penalise when one side has far more tokens than the other.

    A *ratio* is the right shape here because a name is a closed set — if one
    record says "Jayesh Kumar Patel" and the other says only "Patel", the
    second record has withheld most of the name and the agreement is weaker.

    Initials are excluded from the count: "Jayesh R. Patel" and "Jayesh Patel"
    are the same name written two ways and must not be penalised.
    """
    n_left = sum(1 for t in left if len(t) > 1) or len(left)
    n_right = sum(1 for t in right if len(t) > 1) or len(right)
    short, long = min(n_left, n_right), max(n_left, n_right)
    if long == short:
        return 1.0
    return min(1.0, 0.6 + 0.4 * (short / long))


def _informativeness_penalty(left: Sequence[str], right: Sequence[str]) -> float:
    """For addresses: judge the shorter side's substance, not the ratio.

    A ratio penalty would be wrong here, and wrongly in the direction this
    whole module is built to avoid. Addresses are *open* sets: one source
    records a landmark and a building, another records neither. If three
    address tokens agree exactly, the match is confirmed — the fact that one
    record also mentions a temple is a missing field on the other side, not
    evidence against the pair.

    What does genuinely weaken an address match is thinness in absolute terms:
    agreeing on the single token "maninagar" is a locality, not an address.
    """
    shortest = min(len(left), len(right))
    return min(1.0, 0.55 + 0.15 * shortest)


def _landmark_penalty(left: Sequence[str], right: Sequence[str]) -> float:
    """For landmarks: a two-word phrase is already a full landmark.

    "Swaminarayan temple" agreeing on both words is as much agreement as a
    landmark can offer, so it must score 1.0 — the address penalty would cut
    it to 0.85 purely for being short. A single shared word ("temple",
    "masjid") is a category, not a landmark, so that case is discounted.
    """
    return 1.0 if min(len(left), len(right)) >= 2 else 0.75


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #


def f_name_string(left: NormalizedRecord, right: NormalizedRecord) -> FeatureValue:
    alignment = align_tokens(left.name.tokens, right.name.tokens, _string_token_score)
    return FeatureValue(
        name="name_string",
        value=alignment.score,
        weight=0.0,
        detail=_describe_alignment(alignment, "spelling"),
        matched_tokens=alignment.matched,
    )


def f_name_phonetic(left: NormalizedRecord, right: NormalizedRecord) -> FeatureValue:
    alignment = align_tokens(left.name.tokens, right.name.tokens, _phonetic_token_score)
    return FeatureValue(
        name="name_phonetic",
        value=alignment.score,
        weight=0.0,
        detail=_describe_alignment(alignment, "sound"),
        matched_tokens=alignment.matched,
    )


def f_name_rarity(left: NormalizedRecord, right: NormalizedRecord) -> FeatureValue:
    """Is the name agreement carried by anything distinctive?

    Patel, Shah and Kumar are borne by millions of people. Two records that
    agree *only* on such a surname have told us almost nothing, and this is
    the feature that stops a common-surname block from collapsing into one
    giant entity.
    """
    alignment = align_tokens(left.name.tokens, right.name.tokens, _phonetic_token_score)
    matched = alignment.matched
    if not matched:
        return FeatureValue(
            name="name_rarity",
            value=None,
            weight=0.0,
            detail="no name token agreed closely enough to assess",
        )
    common = names_config().common_surnames
    distinctive = [a for a, b in matched if a not in common and b not in common]
    if distinctive:
        return FeatureValue(
            name="name_rarity",
            value=1.0,
            weight=0.0,
            detail=f"distinctive token agrees: {', '.join(sorted(set(distinctive)))}",
            matched_tokens=matched,
        )
    shared = ", ".join(sorted({a for a, _ in matched}))
    return FeatureValue(
        name="name_rarity",
        value=COMMON_ONLY_RARITY,
        weight=0.0,
        detail=f"only a very common surname agrees ({shared}) — weak evidence",
        matched_tokens=matched,
    )


def f_address_core(left: NormalizedRecord, right: NormalizedRecord) -> FeatureValue:
    """Order-tolerant similarity of the identifying address tokens."""
    if left.address is None or right.address is None:
        return FeatureValue(
            name="address_core", value=None, weight=0.0, detail="address missing on one side"
        )
    a, b = left.address.core_tokens, right.address.core_tokens
    if not a or not b:
        return FeatureValue(
            name="address_core",
            value=None,
            weight=0.0,
            detail="address had no identifying tokens",
        )
    alignment = align_tokens(a, b, _string_token_score, _informativeness_penalty)
    overlap = len(set(a) & set(b))
    return FeatureValue(
        name="address_core",
        value=alignment.score,
        weight=0.0,
        detail=(
            f"{overlap} of {min(len(a), len(b))} address tokens match exactly"
            if overlap
            else "no address token matches exactly"
        ),
        matched_tokens=alignment.matched,
    )


def f_pincode(left: NormalizedRecord, right: NormalizedRecord) -> FeatureValue:
    """Exact pincode, or same sorting district as a soft signal."""
    pin_left = left.address.pincode if left.address else None
    pin_right = right.address.pincode if right.address else None
    if not pin_left or not pin_right:
        return FeatureValue(
            name="pincode", value=None, weight=0.0, detail="pincode missing on one side"
        )
    if pin_left == pin_right:
        return FeatureValue(
            name="pincode", value=1.0, weight=0.0, detail=f"same pincode {pin_left}"
        )
    if left.address.district_prefix == right.address.district_prefix:
        return FeatureValue(
            name="pincode",
            value=0.5,
            weight=0.0,
            detail=f"different pincode, same sorting district {left.address.district_prefix}xxx",
        )
    return FeatureValue(
        name="pincode",
        value=0.0,
        weight=0.0,
        detail=f"different pincode ({pin_left} vs {pin_right})",
    )


def f_landmark(left: NormalizedRecord, right: NormalizedRecord) -> FeatureValue:
    """Landmark phrase agreement.

    Low weight but high value when it fires: "near Swaminarayan temple" is
    often the most distinctive thing in an Indian address.
    """
    mark_left = left.address.landmark if left.address else None
    mark_right = right.address.landmark if right.address else None
    if not mark_left or not mark_right:
        return FeatureValue(
            name="landmark", value=None, weight=0.0, detail="no landmark on one side"
        )
    alignment = align_tokens(
        mark_left.split(), mark_right.split(), _string_token_score, _landmark_penalty
    )
    return FeatureValue(
        name="landmark",
        value=alignment.score,
        weight=0.0,
        detail=f"landmark '{mark_left}' vs '{mark_right}'",
        matched_tokens=alignment.matched,
    )


def f_phone(left: NormalizedRecord, right: NormalizedRecord) -> FeatureValue:
    if not left.phone_last10 or not right.phone_last10:
        return FeatureValue(
            name="phone", value=None, weight=0.0, detail="phone missing on one side"
        )
    if left.phone_last10 == right.phone_last10:
        return FeatureValue(
            name="phone",
            value=1.0,
            weight=0.0,
            detail=f"same number (…{left.phone_last10[-4:]})",
        )
    return FeatureValue(name="phone", value=0.0, weight=0.0, detail="different phone numbers")


FEATURE_FUNCTIONS: dict[str, Callable[[NormalizedRecord, NormalizedRecord], FeatureValue]] = {
    "name_string": f_name_string,
    "name_phonetic": f_name_phonetic,
    "name_rarity": f_name_rarity,
    "address_core": f_address_core,
    "pincode": f_pincode,
    "landmark": f_landmark,
    "phone": f_phone,
}


def _describe_alignment(alignment: Alignment, what: str) -> str:
    if not alignment.pairs:
        return "nothing to compare"
    best = max(alignment.pairs, key=lambda p: p[2])
    worst = min(alignment.pairs, key=lambda p: p[2])
    if best is worst or best[2] == worst[2]:
        return f"{what}: '{best[0]}' ~ '{best[1]}' at {best[2]:.0%}"
    return (
        f"{what}: best '{best[0]}' ~ '{best[1]}' at {best[2]:.0%}, "
        f"weakest '{worst[0]}' ~ '{worst[1]}' at {worst[2]:.0%}"
    )


# --------------------------------------------------------------------------- #
# pair scoring
# --------------------------------------------------------------------------- #


def compute_features(
    left: NormalizedRecord, right: NormalizedRecord, weights: WeightsConfig | None = None
) -> dict[str, FeatureValue]:
    """Compute every configured feature, carrying its weight along."""
    weights = weights or weights_config()
    out: dict[str, FeatureValue] = {}
    for name, spec in weights.features.items():
        func = FEATURE_FUNCTIONS.get(name)
        if func is None:
            raise KeyError(
                f"weights.yaml configures feature {name!r} but no function implements it"
            )
        value = func(left, right)
        out[name] = FeatureValue(
            name=name,
            value=value.value,
            weight=spec.weight,
            detail=value.detail,
            matched_tokens=value.matched_tokens,
        )
    return out


def combine(features: dict[str, FeatureValue]) -> float:
    """Weighted mean over the features that could be computed.

    Features with ``value is None`` are excluded from *both* the numerator and
    the denominator — that is what makes a missing field neutral rather than
    damaging.
    """
    usable = [f for f in features.values() if f.value is not None and f.weight > 0]
    total_weight = sum(f.weight for f in usable)
    if total_weight <= 0:
        return 0.0
    return sum(f.value * f.weight for f in usable) / total_weight


def decide(score: float, weights: WeightsConfig) -> Decision:
    if score >= weights.auto_merge:
        return "merge"
    if score >= weights.review_low:
        return "review"
    return "reject"


def score_pair(
    left: NormalizedRecord,
    right: NormalizedRecord,
    weights: WeightsConfig | None = None,
    blocked_by: tuple[str, ...] = (),
) -> PairScore:
    """Score one candidate pair, keeping every feature's contribution."""
    weights = weights or weights_config()
    features = compute_features(left, right, weights)
    score = combine(features)
    return PairScore(
        left_id=left.record_id,
        right_id=right.record_id,
        score=score,
        decision=decide(score, weights),
        features=features,
        blocked_by=blocked_by,
    )
