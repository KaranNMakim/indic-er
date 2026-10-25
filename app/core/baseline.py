"""The baseline, for honest comparison.

This is what a competent team ships in a weekend and what most in-house
deduplication actually is: exact match on the name, then Jaro-Winkler on the
**raw** name strings, blocked on the first two characters.

It is implemented properly rather than strawmanned — the same clustering step,
the same output shape — because the comparison is the whole point of the POV.
If the pipeline cannot beat this by a wide margin on Indian data, it is not
worth buying.

What it deliberately does not do (and what the pipeline does):

* no transliteration, so a Gujarati record never matches its Latin twin
* no honorific stripping, so "Jayeshbhai Patel" is a different person
* no phonetics, so Patel/Patil are different people
* no address or phone evidence, so it cannot separate two real Jayesh Patels
  and cannot join two records whose names were typed differently
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
from rapidfuzz.distance import JaroWinkler

from app.core.config import WeightsConfig, weights_config
from app.core.models import InputRecord

BASELINE_BLOCK_PREFIX = 2


@dataclass(slots=True)
class BaselineResult:
    clusters: list[tuple[str, ...]] = field(default_factory=list)
    matched_pairs: dict[tuple[str, str], float] = field(default_factory=dict)
    candidate_pairs: int = 0
    mode: str = "fuzzy"

    def stats(self) -> dict[str, object]:
        sizes = [len(c) for c in self.clusters]
        return {
            "mode": self.mode,
            "entities": len(self.clusters),
            "singletons": sum(1 for s in sizes if s == 1),
            "merged_entities": sum(1 for s in sizes if s > 1),
            "largest_entity": max(sizes, default=0),
            "matched_pairs": len(self.matched_pairs),
            "candidate_pairs": self.candidate_pairs,
        }


def _raw_key(name: str) -> str:
    """The only "normalization" a naive baseline does: trim and lowercase."""
    return " ".join(name.lower().split())


def run_baseline(
    records: list[InputRecord],
    weights: WeightsConfig | None = None,
    *,
    mode: str = "fuzzy",
) -> BaselineResult:
    """Cluster records by exact name, or by exact plus raw fuzzy name.

    ``mode="exact"`` is the floor: identical strings only.
    ``mode="fuzzy"`` adds Jaro-Winkler above ``baseline.fuzzy_threshold``.
    """
    weights = weights or weights_config()
    result = BaselineResult(mode=mode)
    graph = nx.Graph()
    graph.add_nodes_from(r.record_id for r in records)

    keys = {r.record_id: _raw_key(r.name) for r in records}

    # Exact name match.
    by_exact: dict[str, list[str]] = defaultdict(list)
    for record_id, key in keys.items():
        by_exact[key].append(record_id)
    for members in by_exact.values():
        for i, left in enumerate(sorted(members)):
            for right in sorted(members)[i + 1 :]:
                graph.add_edge(left, right)
                result.matched_pairs[(left, right)] = 1.0

    if mode == "fuzzy":
        threshold = weights.baseline_fuzzy_threshold
        by_prefix: dict[str, list[str]] = defaultdict(list)
        for record_id, key in keys.items():
            by_prefix[key[:BASELINE_BLOCK_PREFIX]].append(record_id)
        for members in by_prefix.values():
            ordered = sorted(members)
            for i, left in enumerate(ordered):
                for right in ordered[i + 1 :]:
                    result.candidate_pairs += 1
                    similarity = JaroWinkler.similarity(keys[left], keys[right])
                    if similarity >= threshold:
                        graph.add_edge(left, right)
                        result.matched_pairs[(left, right)] = similarity

    result.clusters = [tuple(sorted(component)) for component in nx.connected_components(graph)]
    result.clusters.sort(key=lambda c: (-len(c), c[0]))
    return result
