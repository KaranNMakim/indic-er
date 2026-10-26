"""Process-wide application state.

One instance, built on first use and reused: the graph store, the review
queue, the LLM client, and the seeded demo dataset with its resolution.

The seeded dataset is deliberately smaller than the full 5,000-record
evaluation set. `make demo` has to boot in a few seconds to be worth
demonstrating, and the full set takes ~20s to score. `/metrics` serves the
full evaluation numbers from `data/eval.json` instead, so the demo is
interactive *and* the headline figures are the real ones. Set
``INDIC_ER_SEED_RECORDS=5000`` to seed everything.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from app.core.config import DATA_DIR, WeightsConfig, settings, weights_config
from app.core.models import CanonicalEntity, InputRecord, ResolutionResult
from app.core.pipeline import resolve
from app.core.review import ReviewStore
from app.kg.graph_store import NetworkXGraphStore
from app.kg.llm import LLMClient, build_llm_client
from app.kg.ontology import load_ontology
from app.kg.provenance import Provenance

log = logging.getLogger(__name__)

DEFAULT_SEED_RECORDS = 1500


class AppState:
    """Everything the API needs that outlives a request."""

    def __init__(self) -> None:
        self.settings = settings()
        self.weights: WeightsConfig = weights_config()
        self.ontology = load_ontology()
        self.llm: LLMClient = build_llm_client(self.settings)
        self.graph = NetworkXGraphStore(db_path=self.settings.db_path, ontology=self.ontology)
        self.review = ReviewStore(path=self.settings.db_path.with_name("review.db"))
        self.records: dict[str, InputRecord] = {}
        self.result: ResolutionResult | None = None
        self.entities: dict[str, CanonicalEntity] = {}
        self.entity_of_record: dict[str, str] = {}
        self.seed_source = "none"
        self._lock = threading.Lock()
        self.load_seed()

    # ----------------------------------------------------------------- seed #

    def load_seed(self) -> None:
        """Load the seeded dataset from disk, preferring the generated set."""
        limit = int(os.environ.get("INDIC_ER_SEED_RECORDS", DEFAULT_SEED_RECORDS))
        records_csv = DATA_DIR / "records.csv"
        showcase = DATA_DIR / "showcase.json"

        if records_csv.exists():
            self.records = {r.record_id: r for r in _read_csv(records_csv, limit)}
            self.seed_source = f"{records_csv.name} (first {len(self.records)})"
        elif showcase.exists():
            self.records = {r.record_id: r for r in _read_showcase(showcase)}
            self.seed_source = showcase.name
        else:
            self.records = {}
            self.seed_source = "none — run `make data`"
        log.info("seed_loaded", extra={"records": len(self.records), "source": self.seed_source})

    def showcase_records(self) -> list[InputRecord]:
        """The curated 14-record set the playground loads by default."""
        path = DATA_DIR / "showcase.json"
        return _read_showcase(path) if path.exists() else []

    # ------------------------------------------------------------- resolve #

    def resolve_seed(self, *, write_graph: bool = True, enqueue: bool = True) -> ResolutionResult:
        """Resolve the seeded dataset and cache the result.

        Serialised behind a lock: two concurrent requests triggering a 1,500
        record resolution would double the work and interleave the graph
        writes.
        """
        with self._lock:
            if not self.records:
                self.result = resolve([], self.weights)
                return self.result

            result = resolve(
                list(self.records.values()),
                self.weights,
                overrides=self.review.overrides(),
            )
            self.result = result
            self.entities = {entity.entity_id: entity for entity in result.entities}
            self.entity_of_record = {
                member: entity.entity_id
                for entity in result.entities
                for member in entity.member_ids
            }
            if enqueue:
                added = self.review.enqueue(result.review_pairs)
                log.info("review_enqueued", extra={"added": added})
            if write_graph:
                self.write_to_graph(result)
            return result

    def ensure_resolved(self) -> ResolutionResult:
        if self.result is None:
            return self.resolve_seed()
        return self.result

    # --------------------------------------------------------------- graph #

    def write_to_graph(self, result: ResolutionResult) -> None:
        """Project a resolution into the knowledge graph, with provenance.

        Every node and edge carries where it came from, when, how sure we are
        and by what method — the ontology refuses the write otherwise.
        """
        with self.graph.bulk():
            self._write_all(result)
        log.info("graph_written", extra=self.graph.stats())

    def _write_all(self, result: ResolutionResult) -> None:
        self.graph.clear()
        for record_id, record in result.normalized.items():
            self.graph.add_node(
                record_id,
                "Business" if record.kind == "business" else "Person",
                _node_props(record),
                Provenance.from_rule(record.source, 1.0, "ingest", record_id=record_id),
            )

        for entity in result.entities:
            self.graph.add_node(
                entity.entity_id,
                "CanonicalEntity",
                {
                    "entity_id": entity.entity_id,
                    "kind": entity.kind,
                    "canonical_name": entity.canonical_name,
                    "member_ids": list(entity.member_ids),
                    "cluster_score": float(entity.cluster_score),
                    "review_state": entity.review_state,
                },
                Provenance.from_rule(
                    "pipeline",
                    float(entity.cluster_score),
                    "cluster",
                    members=len(entity.member_ids),
                ),
            )
            for member in entity.member_ids:
                self.graph.add_edge(
                    member,
                    entity.entity_id,
                    "same_as",
                    {
                        "score": float(entity.cluster_score),
                        "evidence": {"cluster": entity.entity_id},
                        "decision": "merge",
                    },
                    Provenance.from_rule("pipeline", float(entity.cluster_score), "canonical_link"),
                )

        for pair in result.pairs:
            if pair.decision != "merge":
                continue
            method = (
                Provenance.from_human(
                    pair.override.replace("human_", ""), "accept", pair_id=pair.pair_id
                )
                if pair.override
                else Provenance.from_rule(
                    "pipeline", pair.score, "pairwise_score", pair_id=pair.pair_id
                )
            )
            self.graph.add_edge(
                pair.left_id,
                pair.right_id,
                "same_as",
                {
                    "score": float(pair.score),
                    "evidence": {
                        name: feature.value
                        for name, feature in pair.features.items()
                        if feature.value is not None
                    },
                    "decision": pair.decision,
                },
                method,
            )

    # -------------------------------------------------------------- lookup #

    def entity_for_record(self, record_id: str) -> CanonicalEntity | None:
        entity_id = self.entity_of_record.get(record_id)
        return self.entities.get(entity_id) if entity_id else None

    def find_entity(self, entity_id: str) -> CanonicalEntity | None:
        """By entity id, or by the id of any record inside it."""
        entity = self.entities.get(entity_id)
        if entity is not None:
            return entity
        return self.entity_for_record(entity_id)

    # ----------------------------------------------------------- evaluation #

    def evaluation(self) -> dict[str, Any] | None:
        """The full-dataset evaluation results, if `make eval` has been run."""
        path = DATA_DIR / "eval.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover
            log.warning("eval_json_unreadable", extra={"path": str(path)})
            return None


def _node_props(record) -> dict[str, Any]:
    props: dict[str, Any] = {
        "name": record.raw_name,
        "name_script": record.name.script,
        "name_latin": record.name.display,
        "phonetic_key": record.name.phonetic_key,
        "source": record.source,
    }
    if record.kind == "business" and record.name.legal_suffix:
        props["legal_suffix"] = record.name.legal_suffix
    return props


def _read_csv(path: Path, limit: int) -> list[InputRecord]:
    records: list[InputRecord] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if len(records) >= limit:
                break
            records.append(
                InputRecord(
                    id=row["id"],
                    name=row["name"],
                    kind=row.get("kind") or "person",
                    address=row.get("address") or None,
                    phone=row.get("phone") or None,
                    identifier_scheme=row.get("identifier_scheme") or None,
                    identifier_masked=row.get("identifier_masked") or None,
                    source=row.get("source") or "input",
                )
            )
    return records


def _read_showcase(path: Path) -> list[InputRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        InputRecord(
            id=row["id"],
            name=row["name"],
            kind=row.get("kind") or "person",
            address=row.get("address") or None,
            phone=row.get("phone") or None,
            source=row.get("source") or "input",
        )
        for row in payload.get("records", [])
    ]


_STATE: AppState | None = None
_STATE_LOCK = threading.Lock()


def get_state() -> AppState:
    global _STATE
    if _STATE is None:
        with _STATE_LOCK:
            if _STATE is None:
                _STATE = AppState()
    return _STATE


def reset_state() -> None:
    """Drop the singleton. Used by tests."""
    global _STATE
    with _STATE_LOCK:
        if _STATE is not None:
            _STATE.graph.close()
            _STATE.review.close()
        _STATE = None
