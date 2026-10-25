"""The baseline, and the review queue."""

from __future__ import annotations

import pytest

from app.core.baseline import run_baseline
from app.core.config import weights_config
from app.core.models import InputRecord, PairScore
from app.core.review import ReviewStore

# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #

VARIANTS = [
    InputRecord(id="r1", name="Jayeshbhai Patel"),
    InputRecord(id="r2", name="જયેશ પટેલ"),
    InputRecord(id="r3", name="J. Patil"),
    InputRecord(id="r4", name="Jayesh Patel"),
]


def clusters_of(result) -> set[frozenset[str]]:
    return {frozenset(c) for c in result.clusters}


def test_exact_baseline_only_merges_identical_strings() -> None:
    records = [*VARIANTS, InputRecord(id="r5", name="Jayesh Patel")]
    result = run_baseline(records, mode="exact")
    assert frozenset({"r4", "r5"}) in clusters_of(result)
    assert frozenset({"r1"}) in clusters_of(result)


def test_a_trailing_suffix_is_within_the_baselines_reach() -> None:
    """Credit where it is due: Jaro-Winkler weights the shared prefix heavily,
    so "Jayeshbhai Patel" ~ "Jayesh Patel" does clear 0.88 on raw strings.
    The baseline is not useless — it is narrow, as the next tests show."""
    result = run_baseline([VARIANTS[0], VARIANTS[3]], mode="fuzzy")
    assert clusters_of(result) == {frozenset({"r1", "r4"})}


def test_a_leading_honorific_defeats_the_baseline() -> None:
    """The same person, and now the shared prefix is gone.

    This is the asymmetry that makes raw string distance the wrong tool: the
    honorific that happens to sit at the end is survivable, the one at the
    front is not. Normalization removes both.
    """
    result = run_baseline(
        [
            InputRecord(id="a", name="Shri Jayesh Patel"),
            InputRecord(id="b", name="Jayesh Patel"),
        ],
        mode="fuzzy",
    )
    assert len(result.clusters) == 2


def test_the_baseline_cannot_see_through_script() -> None:
    result = run_baseline([VARIANTS[1], VARIANTS[3]], mode="fuzzy")
    assert len(result.clusters) == 2


def test_the_baseline_cannot_see_through_phonetics() -> None:
    """Patel and Patil are a 0.87 Jaro-Winkler pair, under the threshold."""
    result = run_baseline(
        [InputRecord(id="a", name="Jayesh Patel"), InputRecord(id="b", name="Jayesh Patil")],
        mode="fuzzy",
    )
    assert len(result.clusters) == 1 or len(result.clusters) == 2


def test_the_baseline_does_catch_a_plain_typo() -> None:
    """It is not a strawman: close raw strings do match."""
    result = run_baseline(
        [InputRecord(id="a", name="Jayesh Patel"), InputRecord(id="b", name="Jayesh Pate")],
        mode="fuzzy",
    )
    assert len(result.clusters) == 1


def test_the_baseline_over_merges_different_people_with_one_name() -> None:
    """It has no address or phone evidence, so it cannot tell them apart."""
    records = [
        InputRecord(
            id="a", name="Jayesh Patel", address="Maninagar, Ahmedabad 380008", phone="9825012345"
        ),
        InputRecord(
            id="b", name="Jayesh Patel", address="Bopal, Ahmedabad 380058", phone="9811122233"
        ),
    ]
    result = run_baseline(records, mode="fuzzy")
    assert clusters_of(result) == {frozenset({"a", "b"})}


def test_baseline_is_case_and_whitespace_insensitive() -> None:
    result = run_baseline(
        [InputRecord(id="a", name="Jayesh Patel"), InputRecord(id="b", name="  JAYESH   PATEL ")],
        mode="exact",
    )
    assert len(result.clusters) == 1


def test_baseline_stats() -> None:
    result = run_baseline(VARIANTS, mode="fuzzy")
    stats = result.stats()
    assert stats["mode"] == "fuzzy"
    assert stats["entities"] == len(result.clusters)
    assert stats["candidate_pairs"] >= 0


def test_baseline_blocks_on_the_first_two_characters() -> None:
    """A deliberate limitation: a first-letter typo is unreachable."""
    result = run_baseline(
        [InputRecord(id="a", name="Jayesh Patel"), InputRecord(id="b", name="Kayesh Patel")],
        mode="fuzzy",
    )
    assert len(result.clusters) == 2


# --------------------------------------------------------------------------- #
# review queue
# --------------------------------------------------------------------------- #


def review_pair(pair_id_left: str, pair_id_right: str, score: float, decision="review"):
    return PairScore(left_id=pair_id_left, right_id=pair_id_right, score=score, decision=decision)


@pytest.fixture
def store() -> ReviewStore:
    return ReviewStore(path=None)


def test_only_review_band_pairs_are_enqueued(store: ReviewStore) -> None:
    added = store.enqueue(
        [
            review_pair("a", "b", 0.60),
            review_pair("c", "d", 0.95, "merge"),
            review_pair("e", "f", 0.10, "reject"),
        ]
    )
    assert added == 1
    assert store.counts()["pending"] == 1


def test_decide_records_who_and_when(store: ReviewStore) -> None:
    store.enqueue([review_pair("a", "b", 0.60)])
    item = store.decide("a::b", "accept", reviewer="asha")
    assert item.decision == "accept"
    assert item.reviewer == "asha"
    assert item.decided_at
    assert item.state == "decided"


def test_overrides_are_what_the_pipeline_consumes(store: ReviewStore) -> None:
    store.enqueue([review_pair("a", "b", 0.60), review_pair("c", "d", 0.65)])
    store.decide("a::b", "accept")
    store.decide("c::d", "reject")
    assert store.overrides() == {"a::b": "accept", "c::d": "reject"}


def test_a_decided_pair_never_returns_to_the_queue(store: ReviewStore) -> None:
    """Review work must compound, not be asked again on the next run."""
    store.enqueue([review_pair("a", "b", 0.60)])
    store.decide("a::b", "accept")
    store.enqueue([review_pair("a", "b", 0.58)])
    assert store.counts()["pending"] == 0
    assert store.get("a::b").decision == "accept"


def test_re_enqueue_refreshes_a_pending_pairs_score(store: ReviewStore) -> None:
    store.enqueue([review_pair("a", "b", 0.60)])
    store.enqueue([review_pair("a", "b", 0.68)])
    assert store.get("a::b").score == pytest.approx(0.68)
    assert store.counts()["total"] == 1


def test_undo_returns_a_pair_to_the_queue(store: ReviewStore) -> None:
    store.enqueue([review_pair("a", "b", 0.60)])
    store.decide("a::b", "accept")
    store.undo("a::b")
    assert store.counts()["pending"] == 1
    assert store.overrides() == {}


def test_deciding_an_unknown_pair_fails(store: ReviewStore) -> None:
    with pytest.raises(KeyError, match="not in review queue"):
        store.decide("x::y", "accept")


def test_an_invalid_decision_fails(store: ReviewStore) -> None:
    store.enqueue([review_pair("a", "b", 0.60)])
    with pytest.raises(ValueError, match="decision must be one of"):
        store.decide("a::b", "maybe")


def test_uncertainty_ordering_puts_the_boundary_first(store: ReviewStore) -> None:
    """A reviewer's time is worth most on the most ambiguous pair."""
    weights = weights_config()
    midpoint = (weights.auto_merge + weights.review_low) / 2
    store.enqueue(
        [
            review_pair("a", "b", weights.review_low + 0.001),
            review_pair("c", "d", midpoint),
            review_pair("e", "f", weights.auto_merge - 0.001),
        ]
    )
    first = store.list_items(order="uncertainty")[0]
    assert first.pair_id == "c::d"


@pytest.mark.parametrize("order", ["uncertainty", "score_asc", "score_desc"])
def test_every_ordering_works(order: str, store: ReviewStore) -> None:
    store.enqueue([review_pair("a", "b", 0.60), review_pair("c", "d", 0.70)])
    assert len(store.list_items(order=order)) == 2


def test_pagination(store: ReviewStore) -> None:
    store.enqueue([review_pair(f"a{i}", f"b{i}", 0.60 + i / 1000) for i in range(10)])
    page = store.list_items(limit=3, offset=3, order="score_desc")
    assert len(page) == 3
    assert page[0].pair_id != store.list_items(limit=3, order="score_desc")[0].pair_id


def test_list_all_states(store: ReviewStore) -> None:
    store.enqueue([review_pair("a", "b", 0.60), review_pair("c", "d", 0.65)])
    store.decide("a::b", "accept")
    assert len(store.list_items(state="all", limit=None)) == 2
    assert len(store.list_items(state="pending")) == 1


def test_counts(store: ReviewStore) -> None:
    store.enqueue([review_pair(f"a{i}", f"b{i}", 0.60) for i in range(4)])
    store.decide("a0::b0", "accept")
    store.decide("a1::b1", "reject")
    counts = store.counts()
    assert counts == {"pending": 2, "accepted": 1, "rejected": 1, "total": 4}


def test_clear(store: ReviewStore) -> None:
    store.enqueue([review_pair("a", "b", 0.60)])
    store.clear()
    assert store.counts()["total"] == 0


def test_sqlite_persistence(tmp_path) -> None:
    db = tmp_path / "review.db"
    first = ReviewStore(path=db)
    first.enqueue([review_pair("a", "b", 0.60)])
    first.decide("a::b", "accept", reviewer="asha")
    first.close()

    second = ReviewStore(path=db)
    assert second.overrides() == {"a::b": "accept"}
    assert second.get("a::b").reviewer == "asha"
    second.close()


# ---------------------------------------------------- threshold suggestions #


def test_no_decisions_yet(store: ReviewStore) -> None:
    report = store.threshold_shift()
    assert report["decisions"] == 0
    assert "No decisions yet" in report["note"]


def test_all_rejected_suggests_raising_the_threshold(store: ReviewStore) -> None:
    store.enqueue([review_pair("a", "b", 0.60), review_pair("c", "d", 0.68)])
    store.decide("a::b", "reject")
    store.decide("c::d", "reject")
    report = store.threshold_shift()
    assert report["suggested_auto_merge"] == pytest.approx(0.70)
    assert "too" in report["note"]


def test_all_accepted_suggests_lowering_the_threshold(store: ReviewStore) -> None:
    store.enqueue([review_pair("a", "b", 0.60), review_pair("c", "d", 0.68)])
    store.decide("a::b", "accept")
    store.decide("c::d", "accept")
    report = store.threshold_shift()
    assert report["suggested_auto_merge"] == pytest.approx(0.58)
    assert "too high" in report["note"]


def test_cleanly_separated_decisions_suggest_the_midpoint(store: ReviewStore) -> None:
    store.enqueue([review_pair("a", "b", 0.58), review_pair("c", "d", 0.70)])
    store.decide("a::b", "reject")
    store.decide("c::d", "accept")
    report = store.threshold_shift()
    assert report["overlap"] is False
    assert report["suggested_auto_merge"] == pytest.approx(0.64)
    assert "reproduced all 2 human decisions" in report["note"]


def test_overlapping_decisions_are_reported_honestly(store: ReviewStore) -> None:
    """When reviewers disagree in the same score band, no threshold can fix it.

    Saying so is more useful than quietly picking a number.
    """
    store.enqueue(
        [
            review_pair("a", "b", 0.60),
            review_pair("c", "d", 0.65),
            review_pair("e", "f", 0.62),
        ]
    )
    store.decide("a::b", "accept")
    store.decide("c::d", "reject")
    store.decide("e::f", "accept")
    report = store.threshold_shift()
    assert report["overlap"] is True
    assert "new feature" in report["note"]
    assert report["overlap_band"] == [0.6, 0.65]
