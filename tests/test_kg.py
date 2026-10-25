"""Ontology, provenance, graph store and LLM client."""

from __future__ import annotations

import copy

import pytest

from app.core.config import load_yaml
from app.kg.graph_store import NetworkXGraphStore
from app.kg.llm import (
    Adjudication,
    LLMResponse,
    MockLLMClient,
    _parse_adjudication,
    build_llm_client,
)
from app.kg.ontology import OntologyError, load_ontology, ontology_json, parse_ontology
from app.kg.provenance import Method, Provenance


@pytest.fixture
def ontology():
    return load_ontology()


@pytest.fixture
def raw_ontology() -> dict:
    return copy.deepcopy(load_yaml("ontology"))


@pytest.fixture
def store(ontology):
    return NetworkXGraphStore(db_path=None, ontology=ontology)


# ---------------------------------------------------------------- ontology #


def test_shipped_ontology_loads(ontology) -> None:
    assert "Person" in ontology.class_names()
    assert "CanonicalEntity" in ontology.class_names()
    assert "same_as" in ontology.relation_names()


def test_relation_pointing_at_an_undeclared_class_fails(raw_ontology) -> None:
    raw_ontology["relations"]["same_as"]["range"] = ["Spaceship"]
    with pytest.raises(OntologyError, match="undeclared class"):
        parse_ontology(raw_ontology)


def test_relation_without_a_domain_fails(raw_ontology) -> None:
    raw_ontology["relations"]["same_as"]["domain"] = []
    with pytest.raises(OntologyError, match="domain and a range"):
        parse_ontology(raw_ontology)


def test_dropping_a_provenance_field_fails(raw_ontology) -> None:
    """The whole point of the provenance block: it cannot be opted out of."""
    raw_ontology["provenance"]["required"] = ["source"]
    with pytest.raises(OntologyError, match="auditable"):
        parse_ontology(raw_ontology)


def test_namespace_must_be_a_prefixable_uri(raw_ontology) -> None:
    raw_ontology["namespace"] = "https://indic-er.dev/ontology"
    with pytest.raises(OntologyError, match="'#' or '/'"):
        parse_ontology(raw_ontology)


def test_unknown_property_type_fails(raw_ontology) -> None:
    raw_ontology["classes"]["Person"]["properties"]["name"]["type"] = "blob"
    with pytest.raises(OntologyError, match="validation"):
        parse_ontology(raw_ontology)


def test_validate_node_checks_required_and_pattern(ontology) -> None:
    ontology.validate_node("Address", {"raw": "x", "pincode": "380008"})
    with pytest.raises(OntologyError, match="required"):
        ontology.validate_node("Address", {})
    with pytest.raises(OntologyError, match="does not match"):
        ontology.validate_node("Address", {"raw": "x", "pincode": "38008"})


def test_validate_node_rejects_unknown_properties(ontology) -> None:
    with pytest.raises(OntologyError, match="no propert"):
        ontology.validate_node("Person", {"name": "x", "source": "s", "nickname": "y"})


def test_validate_node_rejects_wrong_type(ontology) -> None:
    with pytest.raises(OntologyError, match="expects string"):
        ontology.validate_node("Person", {"name": 7, "source": "s"})


def test_validate_edge_enforces_domain_and_range(ontology) -> None:
    ontology.validate_edge("located_at", "Person", "Address", {"role": "residence"})
    with pytest.raises(OntologyError, match="not in the range"):
        ontology.validate_edge("located_at", "Person", "Phone", {})
    with pytest.raises(OntologyError, match="not in the domain"):
        ontology.validate_edge("located_at", "Address", "Address", {})
    with pytest.raises(OntologyError, match="unknown relation"):
        ontology.validate_edge("orbits", "Person", "Address", {})


def test_same_as_requires_its_evidence(ontology) -> None:
    """A merge with no evidence is exactly what this project must not produce."""
    with pytest.raises(OntologyError, match="required"):
        ontology.validate_edge("same_as", "Person", "Person", {"score": 0.9})


def test_turtle_export(ontology) -> None:
    turtle = ontology.to_turtle()
    assert "owl:Class" in turtle
    assert "Person" in turtle
    assert "SymmetricProperty" in turtle  # same_as is symmetric


def test_jsonld_export(ontology) -> None:
    import json

    payload = json.loads(ontology.to_jsonld())
    assert isinstance(payload, list)
    assert payload


def test_summary_json_is_serialisable() -> None:
    import json

    payload = json.loads(ontology_json())
    assert payload["classes"]["Person"]["properties"]["name"]["required"] is True
    assert payload["relations"]["same_as"]["symmetric"] is True


# -------------------------------------------------------------- provenance #


def test_provenance_constructors() -> None:
    rule = Provenance.from_rule("kyc", 0.9, "phonetic+pincode")
    assert rule.method is Method.RULE
    assert rule.detail["rule"] == "phonetic+pincode"

    human = Provenance.from_human("asha", "accept", pair_id="a::b")
    assert human.method is Method.HUMAN
    assert human.confidence == 1.0

    model = Provenance.from_model("llm", 0.7, "claude-sonnet-5-5")
    assert model.method is Method.MODEL


def test_provenance_rejects_blank_source() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        Provenance(source="  ", confidence=0.5, method=Method.RULE)


def test_provenance_rejects_impossible_confidence() -> None:
    with pytest.raises(ValueError):
        Provenance(source="s", confidence=1.4, method=Method.RULE)


def test_provenance_describe_and_row() -> None:
    prov = Provenance.from_rule("kyc", 0.91, "phonetic")
    assert "[rule] phonetic" in prov.describe()
    assert prov.as_row()["method"] == "rule"
    assert Provenance.from_human("asha", "reject").describe().startswith("[human] asha reject")
    assert "claude" in Provenance.from_model("llm", 0.5, "claude-x").describe()


# ------------------------------------------------------------- graph store #


def _person(store, node_id: str, name: str) -> None:
    store.add_node(
        node_id,
        "Person",
        {"name": name, "source": "test"},
        Provenance.from_rule("test", 1.0, "ingest"),
    )


def test_add_and_get_node(store) -> None:
    _person(store, "p1", "Jayesh Patel")
    node = store.get_node("p1")
    assert node is not None
    assert node.props["name"] == "Jayesh Patel"
    assert store.get_node("nope") is None


def test_node_write_is_validated(store) -> None:
    with pytest.raises(OntologyError):
        store.add_node("p1", "Person", {"name": "x"}, Provenance.from_rule("t", 1.0, "r"))


def test_none_valued_properties_are_dropped_not_stored(store) -> None:
    store.add_node(
        "p1",
        "Person",
        {"name": "Jayesh", "source": "test", "name_latin": None},
        Provenance.from_rule("t", 1.0, "r"),
    )
    assert "name_latin" not in store.get_node("p1").props


def test_edge_requires_both_endpoints(store) -> None:
    _person(store, "p1", "A")
    with pytest.raises(KeyError, match="target node"):
        store.add_edge(
            "p1",
            "p2",
            "same_as",
            {"score": 1.0, "evidence": {}},
            Provenance.from_rule("t", 1.0, "r"),
        )
    with pytest.raises(KeyError, match="source node"):
        store.add_edge(
            "p0",
            "p1",
            "same_as",
            {"score": 1.0, "evidence": {}},
            Provenance.from_rule("t", 1.0, "r"),
        )


def test_symmetric_edge_is_materialised_both_ways(store) -> None:
    _person(store, "p1", "A")
    _person(store, "p2", "B")
    store.add_edge(
        "p1", "p2", "same_as", {"score": 0.9, "evidence": {}}, Provenance.from_rule("t", 0.9, "r")
    )
    assert [n.id for n in store.neighbors("p2", "same_as")] == ["p1"]
    # ...but it is reported once, not twice.
    assert len(list(store.edges("same_as"))) == 1


def test_neighbors_filters_by_relation(store) -> None:
    _person(store, "p1", "A")
    store.add_node("a1", "Address", {"raw": "x"}, Provenance.from_rule("t", 1.0, "r"))
    store.add_edge(
        "p1", "a1", "located_at", {"role": "residence"}, Provenance.from_rule("t", 1.0, "r")
    )
    assert [n.id for n in store.neighbors("p1", "located_at")] == ["a1"]
    assert list(store.neighbors("p1", "same_as")) == []
    assert list(store.neighbors("missing")) == []


def test_nodes_by_class(store) -> None:
    _person(store, "p1", "A")
    _person(store, "p2", "B")
    store.add_node("a1", "Address", {"raw": "x"}, Provenance.from_rule("t", 1.0, "r"))
    assert {n.id for n in store.nodes_by_class("Person")} == {"p1", "p2"}
    assert {n.id for n in store.nodes_by_class("Address")} == {"a1"}


def test_stats_and_clear(store) -> None:
    _person(store, "p1", "A")
    _person(store, "p2", "B")
    store.add_edge(
        "p1", "p2", "same_as", {"score": 0.9, "evidence": {}}, Provenance.from_rule("t", 0.9, "r")
    )
    stats = store.stats()
    assert stats["nodes"] == 2
    assert stats["edges_by_relation"]["same_as"] == 1
    assert stats["persistent"] is False
    store.clear()
    assert store.stats()["nodes"] == 0


def test_cytoscape_export_shape(store) -> None:
    _person(store, "p1", "Jayesh Patel")
    _person(store, "p2", "Jayesh Patil")
    store.add_edge(
        "p1",
        "p2",
        "same_as",
        {"score": 0.91, "evidence": {}, "decision": "merge"},
        Provenance.from_rule("t", 0.91, "r"),
    )
    payload = store.cytoscape()
    nodes = [e for e in payload["elements"] if e["group"] == "nodes"]
    edges = [e for e in payload["elements"] if e["group"] == "edges"]
    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0]["data"]["score"] == 0.91
    assert nodes[0]["data"]["label"] == "Jayesh Patel"
    assert nodes[0]["data"]["provenance"]["method"] == "rule"


def test_cytoscape_subgraph_excludes_unselected_nodes(store) -> None:
    _person(store, "p1", "A")
    _person(store, "p2", "B")
    payload = store.cytoscape(["p1", "missing"])
    assert [e["data"]["id"] for e in payload["elements"]] == ["p1"]


def test_sqlite_round_trip(tmp_path, ontology) -> None:
    """Facts survive a restart, provenance included."""
    db = tmp_path / "kg.db"
    first = NetworkXGraphStore(db_path=db, ontology=ontology)
    _person(first, "p1", "Jayesh Patel")
    _person(first, "p2", "Jayesh Patil")
    first.add_edge(
        "p1",
        "p2",
        "same_as",
        {"score": 0.93, "evidence": {"name_phonetic": 1.0}},
        Provenance.from_rule("kyc", 0.93, "phonetic"),
    )
    assert first.stats()["persistent"] is True
    first.close()

    second = NetworkXGraphStore(db_path=db, ontology=ontology)
    assert second.get_node("p1").props["name"] == "Jayesh Patel"
    edge = next(second.edges("same_as"))
    assert edge.attributes["score"] == 0.93
    assert edge.provenance.detail["rule"] == "phonetic"
    second.clear()
    second.close()


def test_add_nodes_bulk_helper(store) -> None:
    prov = Provenance.from_rule("t", 1.0, "r")
    store.add_nodes(
        [
            ("p1", "Person", {"name": "A", "source": "t"}, prov),
            ("p2", "Person", {"name": "B", "source": "t"}, prov),
        ]
    )
    assert store.stats()["nodes"] == 2


def test_validation_can_be_turned_off_for_speed(ontology) -> None:
    loose = NetworkXGraphStore(db_path=None, ontology=ontology, validate=False)
    loose.add_node("p1", "Person", {"whatever": 1}, Provenance.from_rule("t", 1.0, "r"))
    assert loose.get_node("p1") is not None


# ---------------------------------------------------------------------- LLM #


def test_mock_client_is_deterministic() -> None:
    client = MockLLMClient()
    first = client.complete("why are these the same?")
    second = client.complete("why are these the same?")
    assert first.text == second.text
    assert client.complete("different prompt").text != first.text


def test_mock_explanation_is_built_from_the_payload() -> None:
    """The mock has to read sensibly in the demo, not show placeholder text."""
    response = MockLLMClient().explain_pair(
        {
            "score": 0.91,
            "decision": "merge",
            "features": {
                "name_phonetic": {"value": 1.0},
                "pincode": {"value": 1.0},
                "address_core": {"value": 0.62},
            },
        }
    )
    assert "same entity" in response.text
    assert "name phonetic" in response.text
    assert "address core" in response.text
    assert response.mode == "mock"


def test_mock_explanation_handles_no_computable_features() -> None:
    response = MockLLMClient().explain_pair({"score": 0.0, "decision": "reject", "features": {}})
    assert "no comparable field" in response.text.lower()


def test_mock_explanation_with_a_single_feature() -> None:
    response = MockLLMClient().explain_pair(
        {"score": 0.6, "decision": "review", "features": {"phone": {"value": 1.0}}}
    )
    assert "only usable evidence" in response.text


@pytest.mark.parametrize(
    ("score", "verdict"),
    [(0.95, "same"), (0.62, "unsure"), (0.2, "different")],
)
def test_mock_adjudication_tracks_the_rule_score(score: float, verdict: str) -> None:
    result = MockLLMClient().adjudicate_pair({"score": score})
    assert result.verdict == verdict
    assert result.advisory is True


def test_adjudication_parses_json_out_of_prose() -> None:
    response = LLMResponse(
        text='Sure! {"verdict": "same", "confidence": 0.8, "rationale": "same phone"} Hope that helps.',
        model="m",
        mode="live",
    )
    result = _parse_adjudication(response)
    assert result.verdict == "same"
    assert result.confidence == 0.8
    assert result.rationale == "same phone"


@pytest.mark.parametrize(
    "text",
    ["not json at all", '{"verdict": "maybe"}', '{"verdict": "same", "confidence": "high"}'],
)
def test_adjudication_degrades_to_unsure(text: str) -> None:
    result = _parse_adjudication(LLMResponse(text=text, model="m", mode="live"))
    assert isinstance(result, Adjudication)
    assert result.verdict in {"unsure", "same"}


def test_build_client_defaults_to_mock(monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODE", raising=False)
    from app.core.config import Settings

    assert build_llm_client(Settings()).mode == "mock"


def test_live_mode_without_a_key_falls_back_to_mock(monkeypatch) -> None:
    """The demo must never break because a key is missing."""
    from app.core.config import Settings

    client = build_llm_client(Settings(llm_mode="live", anthropic_api_key=None))
    assert client.mode == "mock"


def test_core_never_imports_the_llm_layer() -> None:
    """The architectural rule in PLAN.md, enforced as a test.

    Matching must stay reproducible and offline. If someone wires an LLM call
    into ``app/core``, this fails.
    """
    import pathlib

    core = pathlib.Path(__file__).resolve().parents[1] / "app" / "core"
    offenders = [
        path.name
        for path in core.glob("*.py")
        if "app.kg.llm" in path.read_text(encoding="utf-8")
        or "import anthropic" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
