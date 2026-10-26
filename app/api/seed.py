#!/usr/bin/env python
"""Seed the graph store and review queue before the demo boots.

    py -m app.api.seed

Run by `make demo` so the first page load is instant rather than waiting for
a resolution. Safe to re-run: the graph is rebuilt from scratch and human
review decisions are preserved, because they live in their own table and are
re-applied as overrides.
"""

from __future__ import annotations

import sys

from app.api.logging_conf import configure_logging
from app.api.state import get_state
from app.core.config import settings


def main() -> int:
    config = settings()
    configure_logging(config.log_level, config.log_json)

    state = get_state()
    if not state.records:
        print(
            "no seed data found — run `py data/generate.py` (or `make data`) first",
            file=sys.stderr,
        )
        return 1

    print(f"  seed source      {state.seed_source}")
    print(f"  records          {len(state.records)}")

    result = state.resolve_seed()
    clustering = result.stats["clustering"]
    review = state.review.counts()

    print(f"  entities         {clustering['entities']}")
    print(f"  merged entities  {clustering['merged_entities']}")
    print(f"  largest entity   {clustering['largest_entity']}")
    print(f"  duplicate rate   {result.stats['duplicate_rate']:.1%}")
    print(f"  blocked merges   {clustering['blocked_merges']}")
    print(f"  review queue     {review['pending']} pending, {review['total']} total")
    print(f"  graph            {state.graph.stats()}")
    print(f"  resolved in      {result.stats['timings_ms']['total']:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
