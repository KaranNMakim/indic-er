"""Blocking: which pairs are worth scoring at all.

5,000 records is 12,497,500 pairs. Scoring all of them would take minutes and
tell us nothing we could not get from a few tens of thousands, because almost
every pair is obviously unrelated. Blocking generates candidates from cheap
exact keys and lets scoring do the expensive work only where it might matter.

The keys are chosen so that a true match is reachable by at least one of them
even when several fields are wrong:

===========================  =========================================
strategy                     survives
===========================  =========================================
``phonetic_sorted``          token reorder, spelling, honorifics
``phonetic_first_token``     a dropped middle name
``phonetic_token_any``       a missing or extra token on either side
``pincode``                  a completely different name spelling
``phone_last10``             a different name *and* a different address
===========================  =========================================

Recall at this stage is a ceiling that scoring can never raise: a true pair
that lands in no shared block is lost for good. ``make eval`` therefore
reports blocking recall separately from pipeline precision.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from app.core.config import WeightsConfig, weights_config
from app.core.models import NormalizedRecord
from app.core.phonetic import indic_key

log = logging.getLogger(__name__)


@dataclass(slots=True)
class BlockingResult:
    """Candidate pairs plus enough detail to explain and measure the stage."""

    pairs: dict[tuple[str, str], tuple[str, ...]] = field(default_factory=dict)
    blocks: dict[str, list[str]] = field(default_factory=dict)
    skipped_blocks: list[tuple[str, int]] = field(default_factory=list)
    truncated: bool = False

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    def stats(self, record_count: int) -> dict[str, object]:
        all_pairs = record_count * (record_count - 1) // 2
        return {
            "records": record_count,
            "candidate_pairs": self.pair_count,
            "all_pairs": all_pairs,
            "reduction_ratio": (round(1 - (self.pair_count / all_pairs), 6) if all_pairs else 0.0),
            "blocks": len(self.blocks),
            "skipped_oversized_blocks": len(self.skipped_blocks),
            "largest_skipped": (max((size for _, size in self.skipped_blocks), default=0)),
            "truncated": self.truncated,
        }


def record_keys(record: NormalizedRecord, strategies: tuple[str, ...]) -> list[str]:
    """Every blocking key this record belongs to."""
    keys: list[str] = []
    phonetic = record.name.phonetic_tokens

    if "phonetic_sorted" in strategies and phonetic:
        keys.append(f"ps:{'-'.join(sorted(phonetic))}")
    if "phonetic_first_token" in strategies and phonetic:
        keys.append(f"pf:{phonetic[0]}")
    if "phonetic_token_any" in strategies:
        # Single-letter keys come from initials and match far too much.
        keys.extend(f"pt:{token}" for token in set(phonetic) if len(token) > 1)
    if "pincode" in strategies and record.address and record.address.pincode:
        keys.append(f"pin:{record.address.pincode}")
    if "phone_last10" in strategies and record.phone_last10:
        keys.append(f"tel:{record.phone_last10}")
    if "locality_phonetic" in strategies and record.address and record.address.locality:
        locality_key = indic_key(record.address.locality.split()[0])
        if locality_key:
            keys.append(f"loc:{locality_key}")
    return keys


def build_blocks(
    records: dict[str, NormalizedRecord], weights: WeightsConfig | None = None
) -> dict[str, list[str]]:
    """Group record ids by blocking key."""
    weights = weights or weights_config()
    blocks: dict[str, list[str]] = defaultdict(list)
    for record_id, record in records.items():
        for key in record_keys(record, weights.blocking_strategies):
            blocks[key].append(record_id)
    return dict(blocks)


def candidate_pairs(
    records: dict[str, NormalizedRecord], weights: WeightsConfig | None = None
) -> BlockingResult:
    """Generate candidate pairs, recording which strategies produced each.

    Oversized blocks are skipped rather than truncated. A pincode shared by
    400 records contributes ~80,000 pairs on its own and almost no true
    matches that the phonetic keys have not already proposed — but a truncated
    block would silently drop *specific* records, which is worse than dropping
    the key, because those records are still reachable through their more
    selective keys.
    """
    weights = weights or weights_config()
    blocks = build_blocks(records, weights)
    result = BlockingResult(blocks=blocks)

    for key, members in sorted(blocks.items()):
        if len(members) < 2:
            continue
        if len(members) > weights.max_block_size:
            result.skipped_blocks.append((key, len(members)))
            continue
        strategy = key.split(":", 1)[0]
        ordered = sorted(members)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                existing = result.pairs.get((left, right))
                if existing is None:
                    if len(result.pairs) >= weights.max_candidate_pairs:
                        result.truncated = True
                        log.warning(
                            "blocking_truncated",
                            extra={"limit": weights.max_candidate_pairs},
                        )
                        return result
                    result.pairs[(left, right)] = (strategy,)
                elif strategy not in existing:
                    result.pairs[(left, right)] = (*existing, strategy)

    if result.skipped_blocks:
        log.info(
            "blocking_skipped_oversized",
            extra={
                "count": len(result.skipped_blocks),
                "max_block_size": weights.max_block_size,
            },
        )
    return result


STRATEGY_LABELS = {
    "ps": "phonetic_sorted",
    "pf": "phonetic_first_token",
    "pt": "phonetic_token_any",
    "pin": "pincode",
    "tel": "phone_last10",
    "loc": "locality_phonetic",
}


def describe_strategies(codes: tuple[str, ...]) -> tuple[str, ...]:
    """Turn internal key prefixes into the names used in weights.yaml."""
    return tuple(STRATEGY_LABELS.get(code, code) for code in codes)
