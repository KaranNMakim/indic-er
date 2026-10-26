"""Human-in-the-loop review queue.

Borderline pairs are where an entity-resolution system earns or loses trust,
and they are not a bug to be tuned away — on messy Indian data there will
always be a band where the evidence genuinely does not decide. The useful
question is whether a reviewer's work *compounds*:

* a decision is stored, with who made it and when
* on the next run it is applied as a hard override, before clustering, so the
  same pair is never asked about twice
* the accumulated decisions are used to report where the thresholds *should*
  be, which is how a few hundred human judgements turn into a better model
  rather than into a pile of one-off fixes

Stored in SQLite so the decisions survive a restart.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import WeightsConfig, weights_config
from app.core.models import PairScore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_pairs (
    pair_id     TEXT PRIMARY KEY,
    left_id     TEXT NOT NULL,
    right_id    TEXT NOT NULL,
    score       REAL NOT NULL,
    features    TEXT NOT NULL,
    explanation TEXT NOT NULL DEFAULT '',
    state       TEXT NOT NULL DEFAULT 'pending',
    decision    TEXT,
    reviewer    TEXT,
    decided_at  TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_state ON review_pairs(state);
CREATE INDEX IF NOT EXISTS idx_review_score ON review_pairs(score);
"""

VALID_DECISIONS = frozenset({"accept", "reject"})


@dataclass(frozen=True, slots=True)
class ReviewItem:
    pair_id: str
    left_id: str
    right_id: str
    score: float
    features: dict[str, Any]
    explanation: str
    state: str
    decision: str | None
    reviewer: str | None
    decided_at: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "score": round(self.score, 4),
            "features": self.features,
            "explanation": self.explanation,
            "state": self.state,
            "decision": self.decision,
            "reviewer": self.reviewer,
            "decided_at": self.decided_at,
            "created_at": self.created_at,
        }


class ReviewStore:
    """SQLite-backed queue. ``path=None`` keeps it in memory."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(target), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------------------------------------------------------------- write #

    def enqueue(self, pairs: list[PairScore]) -> int:
        """Add review-band pairs. Already-decided pairs are left alone.

        A pair a human has ruled on must never return to the queue, even if a
        re-run scores it differently — that is the difference between review
        work compounding and review work being wasted.
        """
        now = datetime.now(UTC).isoformat()
        added = 0
        for pair in pairs:
            if pair.decision != "review":
                continue
            existing = self._conn.execute(
                "SELECT state FROM review_pairs WHERE pair_id = ?", (pair.pair_id,)
            ).fetchone()
            if existing is not None and existing["state"] != "pending":
                continue
            self._conn.execute(
                """
                INSERT INTO review_pairs
                    (pair_id, left_id, right_id, score, features, explanation,
                     state, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                ON CONFLICT(pair_id) DO UPDATE SET
                    score = excluded.score,
                    features = excluded.features,
                    explanation = excluded.explanation
                """,
                (
                    pair.pair_id,
                    pair.left_id,
                    pair.right_id,
                    pair.score,
                    json.dumps(
                        {name: fv.to_dict() for name, fv in pair.features.items()},
                        ensure_ascii=False,
                    ),
                    pair.explanation,
                    now,
                ),
            )
            added += 1
        self._conn.commit()
        return added

    def decide(self, pair_id: str, decision: str, reviewer: str = "demo") -> ReviewItem:
        """Record a human decision."""
        if decision not in VALID_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}, got {decision!r}")
        row = self._conn.execute(
            "SELECT pair_id FROM review_pairs WHERE pair_id = ?", (pair_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"pair not in review queue: {pair_id}")
        self._conn.execute(
            """
            UPDATE review_pairs
               SET state = 'decided', decision = ?, reviewer = ?, decided_at = ?
             WHERE pair_id = ?
            """,
            (decision, reviewer, datetime.now(UTC).isoformat(), pair_id),
        )
        self._conn.commit()
        item = self.get(pair_id)
        assert item is not None
        return item

    def undo(self, pair_id: str) -> None:
        """Put a decided pair back in the queue (demo convenience)."""
        self._conn.execute(
            """
            UPDATE review_pairs
               SET state = 'pending', decision = NULL, reviewer = NULL, decided_at = NULL
             WHERE pair_id = ?
            """,
            (pair_id,),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM review_pairs")
        self._conn.commit()

    # ----------------------------------------------------------------- read #

    def get(self, pair_id: str) -> ReviewItem | None:
        row = self._conn.execute(
            "SELECT * FROM review_pairs WHERE pair_id = ?", (pair_id,)
        ).fetchone()
        return self._to_item(row) if row else None

    def list_items(
        self,
        state: str = "pending",
        limit: int | None = None,
        offset: int = 0,
        order: str | None = None,
        weights: WeightsConfig | None = None,
    ) -> list[ReviewItem]:
        """List queue items.

        ``order="uncertainty"`` puts the pairs closest to the decision
        boundary first — the ones where a human's five seconds are worth the
        most. ``state="all"`` returns everything.
        """
        weights = weights or weights_config()
        order = order or weights.review_order
        limit = limit if limit is not None else weights.review_page_size

        where, params = ("", ())
        if state != "all":
            where, params = ("WHERE state = ?", (state,))
        rows = self._conn.execute(f"SELECT * FROM review_pairs {where}", params).fetchall()
        items = [self._to_item(row) for row in rows]

        midpoint = (weights.auto_merge + weights.review_low) / 2
        if order == "uncertainty":
            items.sort(key=lambda i: (abs(i.score - midpoint), i.pair_id))
        elif order == "score_asc":
            items.sort(key=lambda i: (i.score, i.pair_id))
        else:
            items.sort(key=lambda i: (-i.score, i.pair_id))
        return items[offset : offset + limit] if limit else items[offset:]

    def overrides(self) -> dict[str, str]:
        """Decided pairs, as ``{pair_id: "accept" | "reject"}``."""
        rows = self._conn.execute(
            "SELECT pair_id, decision FROM review_pairs "
            "WHERE state = 'decided' AND decision IS NOT NULL"
        ).fetchall()
        return {row["pair_id"]: row["decision"] for row in rows}

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT state, decision, COUNT(*) AS n FROM review_pairs GROUP BY state, decision"
        ).fetchall()
        out = {"pending": 0, "accepted": 0, "rejected": 0, "total": 0}
        for row in rows:
            out["total"] += row["n"]
            if row["state"] == "pending":
                out["pending"] += row["n"]
            elif row["decision"] == "accept":
                out["accepted"] += row["n"]
            elif row["decision"] == "reject":
                out["rejected"] += row["n"]
        return out

    # ------------------------------------------------------------ analytics #

    def threshold_shift(self, weights: WeightsConfig | None = None) -> dict[str, Any]:
        """What the decisions so far say about where the thresholds should be.

        The honest version of "the model learns from review": no retraining,
        just the observation that if reviewers keep accepting pairs at 0.58,
        the auto-merge threshold is too high, and if they keep rejecting pairs
        at 0.70, it is too low. The suggestion is reported, never applied
        automatically — moving a threshold is a decision with a cost, and it
        belongs to whoever owns the false-merge risk.
        """
        weights = weights or weights_config()
        rows = self._conn.execute(
            "SELECT score, decision FROM review_pairs WHERE state = 'decided'"
        ).fetchall()
        accepted = sorted(row["score"] for row in rows if row["decision"] == "accept")
        rejected = sorted(row["score"] for row in rows if row["decision"] == "reject")

        report: dict[str, Any] = {
            "decisions": len(rows),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "current_auto_merge": weights.auto_merge,
            "current_review_low": weights.review_low,
            "suggested_auto_merge": weights.auto_merge,
            "overlap": False,
            "note": "",
        }
        if not accepted and not rejected:
            report["note"] = (
                "No decisions yet. Work the review queue and this will show where "
                "the auto-merge threshold should move."
            )
            return report
        if not accepted:
            report["suggested_auto_merge"] = round(min(1.0, max(rejected) + 0.02), 3)
            report["note"] = (
                f"{_count(len(rejected))} rejected so far, the highest scoring "
                f"{max(rejected):.2f}. The auto-merge threshold looks too low — "
                "raising it would cut false merges without losing a confirmed match."
            )
            return report
        if not rejected:
            report["suggested_auto_merge"] = round(max(0.0, min(accepted) - 0.02), 3)
            report["note"] = (
                f"{_count(len(accepted))} accepted so far, the lowest scoring "
                f"{min(accepted):.2f}. The auto-merge threshold looks too high — "
                "lowering it to that point would have merged them automatically."
            )
            return report

        lowest_accept, highest_reject = min(accepted), max(rejected)
        if lowest_accept > highest_reject:
            suggested = round((lowest_accept + highest_reject) / 2, 3)
            report["suggested_auto_merge"] = suggested
            report["note"] = (
                f"The decisions separate cleanly: nothing above {highest_reject:.2f} was "
                f"rejected and nothing below {lowest_accept:.2f} was accepted. "
                f"{suggested:.2f} sits between them and would have reproduced all "
                f"{len(rows)} human decisions automatically."
            )
            return report

        overlap_low, overlap_high = highest_reject, lowest_accept
        report["overlap"] = True
        report["overlap_band"] = [round(overlap_high, 3), round(overlap_low, 3)]
        report["suggested_auto_merge"] = round(_best_split(accepted, rejected), 3)
        report["note"] = (
            f"The decisions overlap between {overlap_high:.2f} and {overlap_low:.2f}: "
            "pairs in that band were sometimes accepted and sometimes rejected, so no "
            "single threshold can reproduce them. That band is genuinely ambiguous on "
            "score alone and is where a new feature — not a new threshold — is what "
            f"would help. {report['suggested_auto_merge']:.2f} is the best compromise."
        )
        return report

    # --------------------------------------------------------------- helper #

    def _to_item(self, row: sqlite3.Row) -> ReviewItem:
        return ReviewItem(
            pair_id=row["pair_id"],
            left_id=row["left_id"],
            right_id=row["right_id"],
            score=row["score"],
            features=json.loads(row["features"]),
            explanation=row["explanation"] or "",
            state=row["state"],
            decision=row["decision"],
            reviewer=row["reviewer"],
            decided_at=row["decided_at"],
            created_at=row["created_at"],
        )

    def close(self) -> None:
        self._conn.close()


def _count(n: int) -> str:
    """ "1 pair" / "7 pairs" — the note is read by a human, not a log parser."""
    return f"{n} pair{'' if n == 1 else 's'}"


def _best_split(accepted: list[float], rejected: list[float]) -> float:
    """The threshold that reproduces the most human decisions."""
    candidates = sorted({*accepted, *rejected})
    best_threshold, best_correct = candidates[0], -1
    for candidate in candidates:
        correct = sum(1 for s in accepted if s >= candidate) + sum(
            1 for s in rejected if s < candidate
        )
        if correct > best_correct:
            best_threshold, best_correct = candidate, correct
    return best_threshold
