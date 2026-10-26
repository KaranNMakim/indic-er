"""`make eval` — measure the pipeline against the baseline and write a report.

    py -m app.evaluation.run_eval [--sweep] [--diagnose 15] [--no-report]

Design note: the expensive step is computing features for ~250,000 candidate
pairs (~15s). Feature *values* do not depend on the weights, so they are
computed once and cached; every threshold in the sweep and every ablation is
then a re-combination of the cached features, which is nearly free. Without
that, a sweep of 25 thresholds plus four ablations would take half an hour.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.baseline import run_baseline
from app.core.blocking import candidate_pairs, describe_strategies
from app.core.cluster import cluster_records
from app.core.config import DATA_DIR, DOCS_DIR, WeightsConfig, weights_config
from app.core.features import combine, compute_features, decide, evidence_gate
from app.core.models import FeatureValue, InputRecord, NormalizedRecord, PairScore
from app.core.normalize import normalize_records
from app.evaluation.metrics import (
    ClusteringReport,
    blocking_recall,
    clusters_to_labels,
    evaluate_clustering,
    slice_report,
)

RECORDS_CSV = DATA_DIR / "records.csv"
TRUTH_CSV = DATA_DIR / "ground_truth.csv"
REPORT_PATH = DOCS_DIR / "eval-report.md"


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Dataset:
    records: list[InputRecord]
    normalized: dict[str, NormalizedRecord]
    truth: dict[str, str]
    tags: dict[str, set[str]]
    pairs: list[PairScore]
    blocking_stats: dict[str, Any]
    blocking: dict[str, Any] = field(default_factory=dict)


def load_records() -> list[InputRecord]:
    if not RECORDS_CSV.exists():
        raise SystemExit(
            f"{RECORDS_CSV} not found — run `py data/generate.py` (or `make data`) first"
        )
    with RECORDS_CSV.open(encoding="utf-8") as handle:
        return [
            InputRecord(
                id=row["id"],
                name=row["name"],
                kind=row["kind"] or "person",
                address=row["address"] or None,
                phone=row["phone"] or None,
                identifier_scheme=row["identifier_scheme"] or None,
                identifier_masked=row["identifier_masked"] or None,
                source=row["source"] or "input",
            )
            for row in csv.DictReader(handle)
        ]


def load_truth() -> tuple[dict[str, str], dict[str, set[str]]]:
    with TRUTH_CSV.open(encoding="utf-8") as handle:
        truth: dict[str, str] = {}
        tags: dict[str, set[str]] = defaultdict(set)
        for row in csv.DictReader(handle):
            truth[row["record_id"]] = row["entity_id"]
            for tag in filter(None, (row.get("tags") or "").split("|")):
                tags[tag].add(row["record_id"])
    return truth, dict(tags)


def build_dataset(weights: WeightsConfig) -> Dataset:
    records = load_records()
    truth, tags = load_truth()
    print(f"  loaded {len(records)} records, {len(set(truth.values()))} true entities")

    started = time.perf_counter()
    normalized = normalize_records(records)
    print(f"  normalized in {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    blocking = candidate_pairs(normalized, weights)
    print(
        f"  blocked in {time.perf_counter() - started:.1f}s "
        f"-> {blocking.pair_count:,} candidate pairs"
    )

    started = time.perf_counter()
    pairs = [
        PairScore(
            left_id=left,
            right_id=right,
            score=0.0,
            decision="reject",
            features=compute_features(normalized[left], normalized[right], weights),
            blocked_by=describe_strategies(strategies),
        )
        for (left, right), strategies in blocking.pairs.items()
    ]
    print(f"  scored features in {time.perf_counter() - started:.1f}s")

    return Dataset(
        records=records,
        normalized=normalized,
        truth=truth,
        tags=tags,
        pairs=pairs,
        blocking_stats=blocking.stats(len(normalized)),
        blocking=blocking.pairs,
    )


# --------------------------------------------------------------------------- #
# scoring variants from cached features
# --------------------------------------------------------------------------- #


def rescore(pairs: list[PairScore], weights: WeightsConfig) -> list[PairScore]:
    """Re-decide cached pairs under different weights or thresholds."""
    out: list[PairScore] = []
    for pair in pairs:
        features = {
            name: FeatureValue(
                name=name,
                value=pair.features[name].value,
                weight=spec.weight,
                detail=pair.features[name].detail,
                matched_tokens=pair.features[name].matched_tokens,
            )
            for name, spec in weights.features.items()
            if name in pair.features
        }
        score = combine(features)
        gate = evidence_gate(features, weights) if score >= weights.auto_merge else None
        out.append(
            PairScore(
                left_id=pair.left_id,
                right_id=pair.right_id,
                score=score,
                decision=decide(score, weights, features),
                features=features,
                blocked_by=pair.blocked_by,
                gate=gate,
            )
        )
    return out


def run_variant(
    dataset: Dataset, weights: WeightsConfig, label: str
) -> tuple[ClusteringReport, dict[str, str], list[PairScore]]:
    pairs = rescore(dataset.pairs, weights)
    outcome = cluster_records(dataset.normalized, pairs, weights)
    predicted = clusters_to_labels([entity.member_ids for entity in outcome.entities])
    merged = sum(1 for p in pairs if p.decision == "merge")
    review = sum(1 for p in pairs if p.decision == "review")
    gated = sum(1 for p in pairs if p.gate)
    report = evaluate_clustering(
        predicted,
        dataset.truth,
        label=label,
        extra={
            "merge_pairs": merged,
            "review_pairs": review,
            "gated_pairs": gated,
            "blocked_merges": len(outcome.blocked_merges),
            "auto_merge": weights.auto_merge,
            "review_low": weights.review_low,
        },
    )
    return report, predicted, pairs


# --------------------------------------------------------------------------- #
# diagnosis
# --------------------------------------------------------------------------- #


def diagnose(
    dataset: Dataset, predicted: dict[str, str], pairs: list[PairScore], limit: int
) -> dict[str, list[dict[str, Any]]]:
    """Sample the pairs we got wrong, with the evidence that misled us."""
    by_id = {p.pair_id: p for p in pairs}
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []

    for pair in sorted(pairs, key=lambda p: -p.score):
        same_predicted = predicted.get(pair.left_id) == predicted.get(pair.right_id)
        same_true = dataset.truth.get(pair.left_id) == dataset.truth.get(pair.right_id)
        if same_predicted and not same_true and len(false_positives) < limit:
            false_positives.append(_describe_pair(dataset, pair))
        elif not same_predicted and same_true and len(false_negatives) < limit:
            false_negatives.append(_describe_pair(dataset, pair))
        if len(false_positives) >= limit and len(false_negatives) >= limit:
            break

    # True pairs that blocking never proposed are the other half of recall.
    unproposed: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = defaultdict(list)
    for record_id, entity in dataset.truth.items():
        groups[entity].append(record_id)
    for members in groups.values():
        members = sorted(members)
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                if f"{left}::{right}" in by_id:
                    continue
                if len(unproposed) < limit:
                    unproposed.append(
                        {
                            "left": dataset.normalized[left].raw_name,
                            "right": dataset.normalized[right].raw_name,
                            "left_address": (dataset.records and _address_of(dataset, left)),
                            "right_address": _address_of(dataset, right),
                        }
                    )
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "never_proposed": unproposed,
    }


def _address_of(dataset: Dataset, record_id: str) -> str:
    record = dataset.normalized[record_id]
    return record.address.raw if record.address else ""


def _describe_pair(dataset: Dataset, pair: PairScore) -> dict[str, Any]:
    left = dataset.normalized[pair.left_id]
    right = dataset.normalized[pair.right_id]
    return {
        "pair_id": pair.pair_id,
        "score": round(pair.score, 3),
        "decision": pair.decision,
        "gate": pair.gate,
        "left": left.raw_name,
        "right": right.raw_name,
        "left_address": left.address.raw if left.address else "",
        "right_address": right.address.raw if right.address else "",
        "left_phone": left.phone_last10 or "",
        "right_phone": right.phone_last10 or "",
        "features": {
            name: (round(f.value, 3) if f.value is not None else None)
            for name, f in pair.features.items()
        },
    }


# --------------------------------------------------------------------------- #
# sweep and ablations
# --------------------------------------------------------------------------- #

SWEEP_POINTS = [round(0.50 + 0.02 * i, 2) for i in range(23)]

ABLATIONS = {
    "no_phonetics": {"feature_weights": {"name_phonetic": 0.0}},
    "no_address": {"feature_weights": {"address_core": 0.0, "pincode": 0.0, "landmark": 0.0}},
    "no_rarity": {"feature_weights": {"name_rarity": 0.0}},
    "no_phone": {"feature_weights": {"phone": 0.0}},
    "no_evidence_gate": {"gate_enabled": False},
    "no_conflict_rules": {"conflicts_enabled": False},
}


def sweep(dataset: Dataset, weights: WeightsConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in SWEEP_POINTS:
        variant = weights.with_overrides(
            auto_merge=threshold, review_low=min(weights.review_low, threshold)
        )
        report, _, _ = run_variant(dataset, variant, f"auto_merge={threshold}")
        rows.append(
            {
                "auto_merge": threshold,
                "pairwise_precision": report.pairwise.precision,
                "pairwise_recall": report.pairwise.recall,
                "pairwise_f1": report.pairwise.f1,
                "bcubed_f1": report.bcubed.f1,
                "entities": report.predicted_clusters,
                "largest": report.largest_predicted,
            }
        )
        print(
            f"    auto_merge={threshold:.2f}  P={report.pairwise.precision:.3f} "
            f"R={report.pairwise.recall:.3f} F1={report.pairwise.f1:.3f} "
            f"B3F1={report.bcubed.f1:.3f} entities={report.predicted_clusters}"
        )
    return rows


BLOCKING_ABLATIONS = {
    "all_strategies": None,
    "without_phonetics": ("pincode", "phone_last10"),
    "without_address_or_phone": (
        "phonetic_sorted",
        "phonetic_first_token",
        "phonetic_token_any",
        "phonetic_initial_sorted",
    ),
    "exact_name_key_only": ("phonetic_sorted",),
}


def ablate_blocking(dataset: Dataset, weights: WeightsConfig) -> list[dict[str, Any]]:
    """What each blocking family contributes to the recall ceiling.

    The scoring ablations show the phonetic key barely moving pairwise F1,
    which is true and easy to misread. Its job is not to score — it is to make
    a true pair a *candidate* at all, and a pair blocking never proposes is
    lost whatever the scorer does. This is where that contribution shows up.
    """
    rows: list[dict[str, Any]] = []
    for label, strategies in BLOCKING_ABLATIONS.items():
        variant = weights
        if strategies is not None:
            variant = weights.with_overrides()
            object.__setattr__(variant, "blocking_strategies", strategies)
        blocking = candidate_pairs(dataset.normalized, variant)
        recall = blocking_recall(list(blocking.pairs), dataset.truth)
        rows.append(
            {
                "strategies": label,
                "candidate_pairs": blocking.pair_count,
                **recall,
            }
        )
        print(
            f"    {label:26} recall={recall['blocking_recall']:.4f} "
            f"missed={recall['missed_true_pairs']:5} "
            f"candidates={blocking.pair_count:,}"
        )
    return rows


def ablate(dataset: Dataset, weights: WeightsConfig) -> list[ClusteringReport]:
    reports: list[ClusteringReport] = []
    for label, overrides in ABLATIONS.items():
        variant = weights.with_overrides(**overrides)
        report, _, _ = run_variant(dataset, variant, label)
        reports.append(report)
        print(
            f"    {label:20} P={report.pairwise.precision:.3f} "
            f"R={report.pairwise.recall:.3f} F1={report.pairwise.f1:.3f} "
            f"B3F1={report.bcubed.f1:.3f}"
        )
    return reports


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", type=int, default=0, help="print N wrong pairs")
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="also dump raw results")
    args = parser.parse_args(argv)

    weights = weights_config()
    print("\n== dataset ==")
    dataset = build_dataset(weights)

    print("\n== blocking ==")
    block = blocking_recall([(p.left_id, p.right_id) for p in dataset.pairs], dataset.truth)
    for key, value in block.items():
        print(f"    {key:24} {value}")

    print("\n== pipeline ==")
    pipeline_report, predicted, pairs = run_variant(dataset, weights, "pipeline")
    _print_report(pipeline_report)

    print("\n== baseline ==")
    baseline_reports: list[ClusteringReport] = []
    for mode in ("exact", "fuzzy"):
        result = run_baseline(dataset.records, weights, mode=mode)
        labels = clusters_to_labels(result.clusters)
        report = evaluate_clustering(
            labels, dataset.truth, label=f"baseline_{mode}", extra=result.stats()
        )
        baseline_reports.append(report)
        _print_report(report)

    slices = slice_report(predicted, dataset.truth, dataset.tags)
    print("\n== slices (recall on true pairs touching each slice) ==")
    for row in slices:
        recall = f"{row['recall']:.3f}" if row["recall"] is not None else "   n/a"
        print(f"    {row['slice']:20} recall={recall}  pairs={row['true_pairs']:6}")

    sweep_rows: list[dict[str, Any]] = []
    if not args.no_sweep:
        print("\n== threshold sweep ==")
        sweep_rows = sweep(dataset, weights)

    ablation_reports: list[ClusteringReport] = []
    blocking_ablations: list[dict[str, Any]] = []
    if not args.no_ablations:
        print("\n== scoring ablations ==")
        ablation_reports = ablate(dataset, weights)
        print("\n== blocking ablations (the recall ceiling) ==")
        blocking_ablations = ablate_blocking(dataset, weights)

    diagnosis: dict[str, list[dict[str, Any]]] = {}
    if args.diagnose:
        diagnosis = diagnose(dataset, predicted, pairs, args.diagnose)
        print("\n== false positives (merged but different entities) ==")
        for row in diagnosis["false_positives"]:
            print(f"    {row['score']:.3f} {row['decision']:6} {row['left']!r} | {row['right']!r}")
            print(f"          {row['left_address'][:70]!r}")
            print(f"          {row['right_address'][:70]!r}")
            print(f"          {row['features']}")
        print("\n== false negatives (same entity, not merged) ==")
        for row in diagnosis["false_negatives"]:
            print(f"    {row['score']:.3f} {row['decision']:6} {row['left']!r} | {row['right']!r}")
            print(f"          gate={row['gate']}")
            print(f"          {row['features']}")
        print("\n== true pairs blocking never proposed ==")
        for row in diagnosis["never_proposed"]:
            print(f"    {row['left']!r} | {row['right']!r}")

    payload = {
        "blocking": {**dataset.blocking_stats, **block},
        "pipeline": pipeline_report.to_dict(),
        "baselines": [r.to_dict() for r in baseline_reports],
        "slices": slices,
        "sweep": sweep_rows,
        "ablations": [r.to_dict() for r in ablation_reports],
        "blocking_ablations": blocking_ablations,
        "diagnosis": diagnosis,
    }
    # Always cached, so `/metrics` and the web app can show ground-truth
    # numbers without re-running the evaluation inside a request.
    cache = DATA_DIR / "eval.json"
    cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nresults cached for /metrics at {cache}")

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"raw results also written to {args.json}")

    if not args.no_report:
        from app.evaluation.report import write_report

        write_report(REPORT_PATH, payload, dataset)
        print(f"\nreport written to {REPORT_PATH}")
    return 0


def _print_report(report: ClusteringReport) -> None:
    pairwise = report.pairwise
    bcubed = report.bcubed
    print(
        f"    {report.label:16} "
        f"pairwise P={pairwise.precision:.3f} R={pairwise.recall:.3f} F1={pairwise.f1:.3f} | "
        f"B3 P={bcubed.precision:.3f} R={bcubed.recall:.3f} F1={bcubed.f1:.3f}"
    )
    print(
        f"    {'':16} entities={report.predicted_clusters} (true {report.true_clusters}) "
        f"largest={report.largest_predicted} (true {report.largest_true}) "
        f"TP={pairwise.true_positives} FP={pairwise.false_positives} "
        f"FN={pairwise.false_negatives}"
    )


if __name__ == "__main__":
    sys.exit(main())
