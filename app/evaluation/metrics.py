"""Evaluation metrics.

Two families, because they answer different questions and a system can look
good on one while failing on the other.

**Pairwise P/R/F1** asks: of the record pairs we said were the same, how many
really are? It is the metric that maps onto the business cost — a false
positive is two customers merged into one account.

**B-cubed** asks the same question per *record* rather than per pair, which
stops one enormous wrong cluster from dominating the score. A cluster of 200
contributes 19,900 pairs to the pairwise metric and only 200 records to
B-cubed, so reporting both makes over-merging visible and honest. B-cubed is
the standard choice in the coreference literature for exactly this reason.

**Blocking recall** is reported separately because it is a ceiling: a true
pair that never became a candidate cannot be recovered by any threshold.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


@dataclass(frozen=True, slots=True)
class PairwiseMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass(frozen=True, slots=True)
class BCubedMetrics:
    precision: float
    recall: float

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass(slots=True)
class ClusteringReport:
    pairwise: PairwiseMetrics
    bcubed: BCubedMetrics
    predicted_clusters: int
    true_clusters: int
    largest_predicted: int
    largest_true: int
    records: int
    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "records": self.records,
            "pairwise": self.pairwise.to_dict(),
            "bcubed": self.bcubed.to_dict(),
            "predicted_clusters": self.predicted_clusters,
            "true_clusters": self.true_clusters,
            "largest_predicted": self.largest_predicted,
            "largest_true": self.largest_true,
            **self.extra,
        }


def clusters_to_labels(clusters: Iterable[Sequence[str]]) -> dict[str, str]:
    """``[[a, b], [c]]`` -> ``{a: "c0", b: "c0", c: "c1"}``."""
    labels: dict[str, str] = {}
    for index, members in enumerate(clusters):
        for member in members:
            labels[member] = f"c{index}"
    return labels


def labels_to_clusters(labels: dict[str, str]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record_id, label in labels.items():
        groups[label].append(record_id)
    return [sorted(members) for members in groups.values()]


def pairwise_metrics(predicted: dict[str, str], truth: dict[str, str]) -> PairwiseMetrics:
    """Pairwise TP/FP/FN without enumerating pairs.

    Counting pairs directly would mean 12.5M comparisons on 5,000 records. The
    identity used instead: inside one predicted cluster, the pairs that are
    truly together are exactly the pairs within each true-label group, so
    ``sum C(count, 2)`` over the cluster's true-label histogram gives TP and
    the rest of ``C(n, 2)`` is FP.
    """
    shared = [record_id for record_id in predicted if record_id in truth]

    predicted_groups: dict[str, list[str]] = defaultdict(list)
    for record_id in shared:
        predicted_groups[predicted[record_id]].append(record_id)

    true_positives = 0
    false_positives = 0
    for members in predicted_groups.values():
        histogram = Counter(truth[member] for member in members)
        same = sum(_pairs(count) for count in histogram.values())
        true_positives += same
        false_positives += _pairs(len(members)) - same

    true_groups = Counter(truth[record_id] for record_id in shared)
    total_true_pairs = sum(_pairs(count) for count in true_groups.values())
    return PairwiseMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=total_true_pairs - true_positives,
    )


def bcubed_metrics(predicted: dict[str, str], truth: dict[str, str]) -> BCubedMetrics:
    """B-cubed precision and recall, averaged over records."""
    shared = [record_id for record_id in predicted if record_id in truth]
    if not shared:
        return BCubedMetrics(precision=1.0, recall=1.0)

    predicted_members: dict[str, list[str]] = defaultdict(list)
    true_members: dict[str, list[str]] = defaultdict(list)
    for record_id in shared:
        predicted_members[predicted[record_id]].append(record_id)
        true_members[truth[record_id]].append(record_id)

    precision_total = 0.0
    recall_total = 0.0
    for record_id in shared:
        cluster = predicted_members[predicted[record_id]]
        entity = true_members[truth[record_id]]
        correct = sum(1 for member in cluster if truth.get(member) == truth[record_id])
        precision_total += correct / len(cluster)
        recall_total += correct / len(entity)

    count = len(shared)
    return BCubedMetrics(precision=precision_total / count, recall=recall_total / count)


def evaluate_clustering(
    predicted: dict[str, str],
    truth: dict[str, str],
    label: str = "",
    extra: dict[str, Any] | None = None,
) -> ClusteringReport:
    shared = [record_id for record_id in predicted if record_id in truth]
    predicted_sizes = Counter(predicted[record_id] for record_id in shared)
    true_sizes = Counter(truth[record_id] for record_id in shared)
    return ClusteringReport(
        pairwise=pairwise_metrics(predicted, truth),
        bcubed=bcubed_metrics(predicted, truth),
        predicted_clusters=len(predicted_sizes),
        true_clusters=len(true_sizes),
        largest_predicted=max(predicted_sizes.values(), default=0),
        largest_true=max(true_sizes.values(), default=0),
        records=len(shared),
        label=label,
        extra=extra or {},
    )


def blocking_recall(
    candidate_pairs: Iterable[tuple[str, str]], truth: dict[str, str]
) -> dict[str, Any]:
    """Share of true same-entity pairs that blocking proposed at all.

    A ceiling, not a score: a true pair missing from the candidates is lost
    whatever the threshold does. Reported on its own so a recall problem can
    be attributed to the right stage.
    """
    true_groups = Counter(truth.values())
    total_true_pairs = sum(_pairs(count) for count in true_groups.values())
    found = 0
    for left, right in candidate_pairs:
        if left in truth and right in truth and truth[left] == truth[right]:
            found += 1
    return {
        "true_pairs": total_true_pairs,
        "true_pairs_proposed": found,
        "blocking_recall": round(found / total_true_pairs, 4) if total_true_pairs else 1.0,
        "missed_true_pairs": total_true_pairs - found,
    }


def slice_report(
    predicted: dict[str, str],
    truth: dict[str, str],
    tags: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Per-slice recall, so one average cannot hide a broken failure mode.

    A slice is a set of records (those carrying a tag). Recall is measured
    over the true pairs in which *at least one* side carries the tag, which is
    the question that matters: "if a record arrives in Gujarati script, do we
    still find its duplicates?"
    """
    true_groups: dict[str, list[str]] = defaultdict(list)
    for record_id, entity in truth.items():
        true_groups[entity].append(record_id)

    rows: list[dict[str, Any]] = []
    for tag, members in sorted(tags.items()):
        relevant = 0
        found = 0
        for group in true_groups.values():
            for i, left in enumerate(group):
                for right in group[i + 1 :]:
                    if left not in members and right not in members:
                        continue
                    relevant += 1
                    if predicted.get(left) and predicted.get(left) == predicted.get(right):
                        found += 1
        rows.append(
            {
                "slice": tag,
                "records": len(members),
                "true_pairs": relevant,
                "recall": round(found / relevant, 4) if relevant else None,
            }
        )
    return rows
