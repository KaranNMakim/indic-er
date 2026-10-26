"""GraphStore: the interface, and a NetworkX + SQLite implementation.

Why an interface at all in a POV: the matching code should never know where
the graph lives. NetworkX gives free traversal and SQLite gives durability,
which is the right trade for a demo on one machine. Swapping in Neo4j or Kuzu
means writing one more subclass — no change to ``app/core``.

Every write goes through ontology validation and requires a
:class:`~app.kg.provenance.Provenance`, so the store cannot hold a fact that
nobody can account for.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from app.kg.ontology import Ontology, load_ontology
from app.kg.provenance import Provenance


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    class_name: str
    props: dict[str, Any]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class Edge:
    src: str
    dst: str
    relation: str
    attributes: dict[str, Any]
    provenance: Provenance


class GraphStore(ABC):
    """Minimum surface the rest of the system relies on."""

    @abstractmethod
    def add_node(
        self, node_id: str, class_name: str, props: dict[str, Any], provenance: Provenance
    ) -> None: ...

    @abstractmethod
    def add_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        attributes: dict[str, Any],
        provenance: Provenance,
    ) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> Node | None: ...

    @abstractmethod
    def nodes_by_class(self, class_name: str) -> Iterator[Node]: ...

    @abstractmethod
    def edges(self, relation: str | None = None) -> Iterator[Edge]: ...

    @abstractmethod
    def neighbors(self, node_id: str, relation: str | None = None) -> Iterator[Node]: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def stats(self) -> dict[str, Any]: ...

    def add_nodes(self, nodes: Iterable[tuple[str, str, dict[str, Any], Provenance]]) -> None:
        for node_id, class_name, props, prov in nodes:
            self.add_node(node_id, class_name, props, prov)


# --------------------------------------------------------------------------- #
# NetworkX + SQLite
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    class_name  TEXT NOT NULL,
    props       TEXT NOT NULL,
    provenance  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    src         TEXT NOT NULL,
    dst         TEXT NOT NULL,
    relation    TEXT NOT NULL,
    attributes  TEXT NOT NULL,
    provenance  TEXT NOT NULL,
    PRIMARY KEY (src, dst, relation)
);
CREATE INDEX IF NOT EXISTS idx_nodes_class ON nodes(class_name);
CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(relation);
"""


class NetworkXGraphStore(GraphStore):
    """In-memory NetworkX graph, optionally persisted to SQLite.

    Pass ``db_path=None`` for a pure in-memory store (what the tests and the
    stateless ``/resolve`` call use).
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        ontology: Ontology | None = None,
        *,
        validate: bool = True,
    ) -> None:
        self.ontology = ontology or load_ontology()
        self.validate = validate
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._conn: sqlite3.Connection | None = None
        self._autocommit = True
        if db_path is not None:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            self._load_from_db()

    @contextmanager
    def bulk(self):
        """Defer SQLite commits until the block exits.

        Committing per row made a 4,700-fact write take 41 seconds. One
        transaction takes well under one. Projecting a whole resolution into
        the graph is the only write path that matters for latency, and it is
        naturally a single unit of work.
        """
        if self._conn is None:
            yield self
            return
        previous = self._autocommit
        self._autocommit = False
        try:
            yield self
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._autocommit = previous

    def _maybe_commit(self) -> None:
        if self._conn is not None and self._autocommit:
            self._conn.commit()

    # ----------------------------------------------------------------- load #

    def _load_from_db(self) -> None:
        assert self._conn is not None
        for node_id, class_name, props, prov in self._conn.execute(
            "SELECT id, class_name, props, provenance FROM nodes"
        ):
            self.graph.add_node(
                node_id,
                class_name=class_name,
                props=json.loads(props),
                provenance=Provenance.model_validate_json(prov),
            )
        for src, dst, relation, attrs, prov in self._conn.execute(
            "SELECT src, dst, relation, attributes, provenance FROM edges"
        ):
            self.graph.add_edge(
                src,
                dst,
                key=relation,
                relation=relation,
                attributes=json.loads(attrs),
                provenance=Provenance.model_validate_json(prov),
            )

    # ---------------------------------------------------------------- write #

    def add_node(
        self, node_id: str, class_name: str, props: dict[str, Any], provenance: Provenance
    ) -> None:
        clean = {k: v for k, v in props.items() if v is not None}
        if self.validate:
            self.ontology.validate_node(class_name, clean)
        self.graph.add_node(node_id, class_name=class_name, props=clean, provenance=provenance)
        if self._conn is not None:
            self._conn.execute(
                "INSERT OR REPLACE INTO nodes (id, class_name, props, provenance) "
                "VALUES (?, ?, ?, ?)",
                (
                    node_id,
                    class_name,
                    json.dumps(clean, ensure_ascii=False, default=str),
                    provenance.model_dump_json(),
                ),
            )
            self._maybe_commit()

    def add_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        attributes: dict[str, Any],
        provenance: Provenance,
    ) -> None:
        if src not in self.graph:
            raise KeyError(f"source node not in graph: {src}")
        if dst not in self.graph:
            raise KeyError(f"target node not in graph: {dst}")
        if self.validate:
            self.ontology.validate_edge(
                relation,
                self.graph.nodes[src]["class_name"],
                self.graph.nodes[dst]["class_name"],
                attributes,
            )
        self.graph.add_edge(
            src,
            dst,
            key=relation,
            relation=relation,
            attributes=attributes,
            provenance=provenance,
        )
        symmetric = self.ontology.relations[relation].symmetric
        if symmetric and not self.graph.has_edge(dst, src, key=relation):
            self.graph.add_edge(
                dst,
                src,
                key=relation,
                relation=relation,
                attributes=attributes,
                provenance=provenance,
            )
        if self._conn is not None:
            self._conn.execute(
                "INSERT OR REPLACE INTO edges (src, dst, relation, attributes, provenance) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    src,
                    dst,
                    relation,
                    json.dumps(attributes, ensure_ascii=False, default=str),
                    provenance.model_dump_json(),
                ),
            )
            self._maybe_commit()

    def clear(self) -> None:
        self.graph.clear()
        if self._conn is not None:
            self._conn.execute("DELETE FROM nodes")
            self._conn.execute("DELETE FROM edges")
            self._maybe_commit()

    # ----------------------------------------------------------------- read #

    def get_node(self, node_id: str) -> Node | None:
        data = self.graph.nodes.get(node_id)
        if not data:
            return None
        return Node(
            id=node_id,
            class_name=data["class_name"],
            props=data["props"],
            provenance=data["provenance"],
        )

    def nodes_by_class(self, class_name: str) -> Iterator[Node]:
        for node_id, data in self.graph.nodes(data=True):
            if data.get("class_name") == class_name:
                yield Node(
                    id=node_id,
                    class_name=class_name,
                    props=data["props"],
                    provenance=data["provenance"],
                )

    def edges(self, relation: str | None = None) -> Iterator[Edge]:
        seen: set[tuple[str, str, str]] = set()
        for src, dst, key, data in self.graph.edges(keys=True, data=True):
            if relation is not None and key != relation:
                continue
            if self.ontology.relations[key].symmetric:
                canonical = (min(src, dst), max(src, dst), key)
                if canonical in seen:
                    continue
                seen.add(canonical)
            yield Edge(
                src=src,
                dst=dst,
                relation=key,
                attributes=data["attributes"],
                provenance=data["provenance"],
            )

    def neighbors(self, node_id: str, relation: str | None = None) -> Iterator[Node]:
        if node_id not in self.graph:
            return
        for _, dst, key in self.graph.out_edges(node_id, keys=True):
            if relation is not None and key != relation:
                continue
            node = self.get_node(dst)
            if node is not None:
                yield node

    def stats(self) -> dict[str, Any]:
        by_class: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            by_class[data["class_name"]] = by_class.get(data["class_name"], 0) + 1
        by_relation: dict[str, int] = {}
        for edge in self.edges():
            by_relation[edge.relation] = by_relation.get(edge.relation, 0) + 1
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": sum(by_relation.values()),
            "nodes_by_class": dict(sorted(by_class.items())),
            "edges_by_relation": dict(sorted(by_relation.items())),
            "backend": "networkx+sqlite" if self._conn is not None else "networkx",
            "persistent": self._conn is not None,
        }

    # --------------------------------------------------------- presentation #

    def cytoscape(self, node_ids: Iterable[str] | None = None) -> dict[str, Any]:
        """Export (a subgraph as) Cytoscape.js elements."""
        wanted = set(node_ids) if node_ids is not None else set(self.graph.nodes)
        elements: list[dict[str, Any]] = []
        for node_id in sorted(wanted):
            data = self.graph.nodes.get(node_id)
            if not data:
                continue
            props = data["props"]
            elements.append(
                {
                    "group": "nodes",
                    "data": {
                        "id": node_id,
                        "label": props.get("name")
                        or props.get("canonical_name")
                        or props.get("raw")
                        or node_id,
                        "kind": data["class_name"],
                        "props": props,
                        "provenance": data["provenance"].as_row(),
                    },
                }
            )
        for edge in self.edges():
            if edge.src in wanted and edge.dst in wanted:
                elements.append(
                    {
                        "group": "edges",
                        "data": {
                            "id": f"{edge.src}|{edge.relation}|{edge.dst}",
                            "source": edge.src,
                            "target": edge.dst,
                            "relation": edge.relation,
                            "score": edge.attributes.get("score"),
                            "decision": edge.attributes.get("decision"),
                            "provenance": edge.provenance.as_row(),
                        },
                    }
                )
        return {"elements": elements}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
