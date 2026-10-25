"""Ontology: YAML in, validated model out, JSON-LD / Turtle on request.

The ontology is loaded and validated at startup. A relation pointing at an
undeclared class, a property with an unknown type, or a missing provenance
field fails the load rather than producing a graph nobody can trust.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import ConfigError, load_yaml

PropertyType = Literal["string", "int", "float", "bool", "json", "datetime"]

ALLOWED_TYPES: frozenset[str] = frozenset(["string", "int", "float", "bool", "json", "datetime"])

_PY_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "int": (int,),
    "float": (float, int),
    "bool": (bool,),
    "json": (dict, list, str),
    "datetime": (str,),
}


class OntologyError(ConfigError):
    """The ontology failed validation."""


class PropertySpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: PropertyType = "string"
    required: bool = False
    description: str = ""
    pattern: str | None = None

    def check(self, name: str, value: Any) -> None:
        """Raise if ``value`` is not acceptable for this property."""
        if value is None:
            if self.required:
                raise OntologyError(f"property {name!r} is required")
            return
        expected = _PY_TYPES[self.type]
        if self.type != "bool" and isinstance(value, bool):
            raise OntologyError(f"property {name!r} expects {self.type}, got bool")
        if not isinstance(value, expected):
            raise OntologyError(
                f"property {name!r} expects {self.type}, got {type(value).__name__}"
            )
        if self.pattern and isinstance(value, str) and not re.match(self.pattern, value):
            raise OntologyError(f"property {name!r} value {value!r} does not match {self.pattern}")


class ClassSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = ""
    description: str = ""
    properties: dict[str, PropertySpec] = Field(default_factory=dict)


class RelationSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = ""
    description: str = ""
    domain: tuple[str, ...] = ()
    range: tuple[str, ...] = ()
    symmetric: bool = False
    attributes: dict[str, PropertySpec] = Field(default_factory=dict)


class ProvenanceSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    required: tuple[str, ...] = ("source", "timestamp", "confidence", "method")
    methods: tuple[str, ...] = ("rule", "model", "human")


class Ontology(BaseModel):
    """The validated ontology."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    namespace: str
    prefixes: dict[str, str] = Field(default_factory=dict)
    classes: dict[str, ClassSpec]
    relations: dict[str, RelationSpec]
    provenance: ProvenanceSpec = Field(default_factory=ProvenanceSpec)

    # ---------------------------------------------------------------- checks #

    @model_validator(mode="after")
    def _validate_references(self) -> Ontology:
        if not self.classes:
            raise OntologyError("ontology declares no classes")
        if not self.relations:
            raise OntologyError("ontology declares no relations")
        if not self.namespace.endswith(("#", "/")):
            raise OntologyError("namespace must end with '#' or '/'")

        known = set(self.classes)
        for rel_name, rel in self.relations.items():
            if not rel.domain or not rel.range:
                raise OntologyError(f"relation {rel_name!r} needs a domain and a range")
            for side, members in (("domain", rel.domain), ("range", rel.range)):
                unknown = sorted(set(members) - known)
                if unknown:
                    raise OntologyError(
                        f"relation {rel_name!r} {side} references undeclared "
                        f"class(es): {', '.join(unknown)}"
                    )

        required = set(self.provenance.required)
        missing = {"source", "timestamp", "confidence", "method"} - required
        if missing:
            raise OntologyError(
                "provenance.required must include "
                f"{', '.join(sorted(missing))} — every fact must be auditable"
            )
        return self

    # ------------------------------------------------------------ public API #

    def class_names(self) -> list[str]:
        return sorted(self.classes)

    def relation_names(self) -> list[str]:
        return sorted(self.relations)

    def validate_node(self, class_name: str, props: dict[str, Any]) -> None:
        """Check node properties against the class definition."""
        spec = self.classes.get(class_name)
        if spec is None:
            raise OntologyError(
                f"unknown class {class_name!r}; known: {', '.join(self.class_names())}"
            )
        for prop_name, prop_spec in spec.properties.items():
            prop_spec.check(f"{class_name}.{prop_name}", props.get(prop_name))
        unknown = sorted(set(props) - set(spec.properties))
        if unknown:
            raise OntologyError(
                f"{class_name} has no propert{'y' if len(unknown) == 1 else 'ies'}: "
                f"{', '.join(unknown)}"
            )

    def validate_edge(
        self,
        relation: str,
        src_class: str,
        dst_class: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Check a relation's endpoints and attributes."""
        spec = self.relations.get(relation)
        if spec is None:
            raise OntologyError(
                f"unknown relation {relation!r}; known: {', '.join(self.relation_names())}"
            )
        if src_class not in spec.domain:
            raise OntologyError(
                f"{relation}: {src_class} is not in the domain ({', '.join(spec.domain)})"
            )
        if dst_class not in spec.range:
            raise OntologyError(
                f"{relation}: {dst_class} is not in the range ({', '.join(spec.range)})"
            )
        attributes = attributes or {}
        for attr_name, attr_spec in spec.attributes.items():
            attr_spec.check(f"{relation}.{attr_name}", attributes.get(attr_name))

    # --------------------------------------------------------------- export #

    def to_rdf_graph(self):  # noqa: ANN201 - rdflib Graph, imported lazily
        """Build an rdflib graph of the ontology."""
        from rdflib import RDF, RDFS, Graph, Namespace, URIRef
        from rdflib import Literal as RdfLiteral
        from rdflib.namespace import OWL, XSD

        ns = Namespace(self.namespace)
        graph = Graph()
        for prefix, uri in self.prefixes.items():
            graph.bind(prefix, Namespace(uri))
        graph.bind("owl", OWL)

        onto_uri = URIRef(self.namespace.rstrip("#/"))
        graph.add((onto_uri, RDF.type, OWL.Ontology))
        graph.add((onto_uri, RDFS.label, RdfLiteral(self.name)))
        graph.add((onto_uri, OWL.versionInfo, RdfLiteral(self.version)))

        xsd_for = {
            "string": XSD.string,
            "int": XSD.integer,
            "float": XSD.double,
            "bool": XSD.boolean,
            "datetime": XSD.dateTime,
            "json": XSD.string,
        }

        for cls_name, cls in self.classes.items():
            cls_uri = ns[cls_name]
            graph.add((cls_uri, RDF.type, OWL.Class))
            graph.add((cls_uri, RDFS.label, RdfLiteral(cls.label or cls_name)))
            if cls.description:
                graph.add((cls_uri, RDFS.comment, RdfLiteral(cls.description.strip())))
            for prop_name, prop in cls.properties.items():
                prop_uri = ns[f"{cls_name}.{prop_name}"]
                graph.add((prop_uri, RDF.type, OWL.DatatypeProperty))
                graph.add((prop_uri, RDFS.label, RdfLiteral(prop_name)))
                graph.add((prop_uri, RDFS.domain, cls_uri))
                graph.add((prop_uri, RDFS.range, xsd_for[prop.type]))
                if prop.description:
                    graph.add((prop_uri, RDFS.comment, RdfLiteral(prop.description.strip())))

        for rel_name, rel in self.relations.items():
            rel_uri = ns[rel_name]
            graph.add((rel_uri, RDF.type, OWL.ObjectProperty))
            graph.add((rel_uri, RDFS.label, RdfLiteral(rel.label or rel_name)))
            if rel.description:
                graph.add((rel_uri, RDFS.comment, RdfLiteral(rel.description.strip())))
            if rel.symmetric:
                graph.add((rel_uri, RDF.type, OWL.SymmetricProperty))
            for cls_name in rel.domain:
                graph.add((rel_uri, RDFS.domain, ns[cls_name]))
            for cls_name in rel.range:
                graph.add((rel_uri, RDFS.range, ns[cls_name]))

        return graph

    def to_turtle(self) -> str:
        return self.to_rdf_graph().serialize(format="turtle")

    def to_jsonld(self) -> str:
        return self.to_rdf_graph().serialize(format="json-ld", indent=2)

    def to_summary(self) -> dict[str, Any]:
        """Compact JSON for the ``/ontology`` endpoint and the web app."""
        return {
            "name": self.name,
            "version": self.version,
            "namespace": self.namespace,
            "classes": {
                name: {
                    "description": cls.description.strip(),
                    "properties": {
                        p: {
                            "type": spec.type,
                            "required": spec.required,
                            "description": spec.description.strip(),
                        }
                        for p, spec in cls.properties.items()
                    },
                }
                for name, cls in self.classes.items()
            },
            "relations": {
                name: {
                    "description": rel.description.strip(),
                    "domain": list(rel.domain),
                    "range": list(rel.range),
                    "symmetric": rel.symmetric,
                    "attributes": {
                        a: {"type": spec.type, "required": spec.required}
                        for a, spec in rel.attributes.items()
                    },
                }
                for name, rel in self.relations.items()
            },
            "provenance": {
                "required": list(self.provenance.required),
                "methods": list(self.provenance.methods),
            },
        }


def parse_ontology(raw: dict[str, Any]) -> Ontology:
    """Validate a raw mapping into an :class:`Ontology`."""
    try:
        return Ontology.model_validate(raw)
    except OntologyError:
        raise
    except Exception as exc:  # pydantic ValidationError and friends
        raise OntologyError(f"ontology failed validation: {exc}") from exc


@lru_cache(maxsize=1)
def load_ontology() -> Ontology:
    """Load and validate ``config/ontology.yaml`` (cached)."""
    return parse_ontology(load_yaml("ontology"))


def ontology_json() -> str:
    return json.dumps(load_ontology().to_summary(), indent=2, ensure_ascii=False)
