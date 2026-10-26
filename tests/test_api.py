"""API contract tests.

The seeded dataset is cut to a few dozen records here (via
``INDIC_ER_SEED_RECORDS``) and the stores are redirected to a temp directory,
so the suite exercises the real wiring without paying for a 1,500-record
resolution per test session.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

SHOWCASE_PAIR = {
    "left": {
        "id": "a",
        "name": "Jayeshbhai Patel",
        "address": "12, Shreeji Soc., Nr. Swaminarayan Mandir, Maninagar, Ahmedabad 380008",
        "phone": "+91 98250 12345",
    },
    "right": {
        "id": "b",
        "name": "જયેશ પટેલ",
        "address": "12 Shriji Society, opp Swaminarayan Temple, Maninagar, Ahmedabad",
        "phone": "9825012345",
    },
}


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> Iterator[TestClient]:
    import os

    tmp = tmp_path_factory.mktemp("api")
    os.environ["INDIC_ER_SEED_RECORDS"] = "60"
    os.environ["INDIC_ER_DB"] = str(tmp / "indic_er.db")

    from app.api import state as state_module
    from app.core.config import reset_caches

    reset_caches()
    state_module.reset_state()

    from app.api.main import app

    with TestClient(app) as test_client:
        yield test_client

    state_module.reset_state()
    os.environ.pop("INDIC_ER_SEED_RECORDS", None)
    os.environ.pop("INDIC_ER_DB", None)
    reset_caches()


# --------------------------------------------------------------------- meta #


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["llm_mode"] == "mock"
    assert payload["thresholds"]["auto_merge"] > payload["thresholds"]["review_low"]
    assert "same_as" not in payload  # health is a summary, not a dump
    assert payload["graph"]["backend"].startswith("networkx")


def test_health_sets_a_request_id(client: TestClient) -> None:
    assert client.get("/health").headers["x-request-id"]


def test_request_id_is_echoed_back(client: TestClient) -> None:
    response = client.get("/health", headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"


def test_ontology_json(client: TestClient) -> None:
    payload = client.get("/ontology").json()
    assert payload["classes"]["Person"]["properties"]["name"]["required"] is True
    assert payload["relations"]["same_as"]["symmetric"] is True
    assert set(payload["provenance"]["required"]) == {
        "source",
        "timestamp",
        "confidence",
        "method",
    }


def test_ontology_turtle(client: TestClient) -> None:
    response = client.get("/ontology", params={"format": "turtle"})
    assert response.headers["content-type"].startswith("text/turtle")
    assert "owl:Class" in response.text


def test_ontology_jsonld(client: TestClient) -> None:
    response = client.get("/ontology", params={"format": "json-ld"})
    assert response.headers["content-type"].startswith("application/ld+json")
    assert isinstance(json.loads(response.text), list)


def test_openapi_documents_every_endpoint(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for required in (
        "/resolve",
        "/match",
        "/entities/{entity_id}",
        "/review-queue",
        "/review/{pair_id}",
        "/metrics",
        "/health",
        "/ontology",
    ):
        assert required in paths, f"{required} missing from the OpenAPI schema"


# ------------------------------------------------------------------ resolve #


def test_resolve_merges_the_motivating_example(client: TestClient) -> None:
    response = client.post(
        "/resolve",
        json={
            "records": [SHOWCASE_PAIR["left"], SHOWCASE_PAIR["right"]],
            "apply_reviews": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["entities"]) == 1
    assert payload["entities"][0]["size"] == 2
    assert payload["entities"][0]["canonical_name"] == "Jayesh Patel"
    assert payload["stats"]["pairs_merged"] == 1
    assert payload["normalized"]["b"]["name"]["script"] == "gujarati"


def test_resolve_can_skip_the_normalization_trace(client: TestClient) -> None:
    payload = client.post(
        "/resolve",
        json={
            "records": [SHOWCASE_PAIR["left"], SHOWCASE_PAIR["right"]],
            "include_normalized": False,
        },
    ).json()
    assert payload["normalized"] is None


def test_resolve_accepts_a_threshold_override(client: TestClient) -> None:
    """The sweep in the eval report is only useful if it is reachable."""
    records = [
        {
            "id": "a",
            "name": "Jayesh Patel",
            "address": "12 Shreeji Soc, Maninagar, Ahmedabad 380008",
        },
        {
            "id": "b",
            "name": "Jayesh Patil",
            "address": "12 Shriji Society, Maninagar, Ahmedabad 380008",
        },
    ]
    strict = client.post(
        "/resolve", json={"records": records, "auto_merge": 0.99, "apply_reviews": False}
    ).json()
    loose = client.post(
        "/resolve", json={"records": records, "auto_merge": 0.50, "apply_reviews": False}
    ).json()
    assert len(strict["entities"]) >= len(loose["entities"])


def test_resolve_rejects_duplicate_ids(client: TestClient) -> None:
    response = client.post(
        "/resolve",
        json={"records": [{"id": "a", "name": "One"}, {"id": "a", "name": "Two"}]},
    )
    assert response.status_code == 422
    assert "duplicate" in response.text.lower()


def test_resolve_rejects_an_empty_batch(client: TestClient) -> None:
    assert client.post("/resolve", json={"records": []}).status_code == 422


def test_resolve_rejects_an_unmasked_identifier(client: TestClient) -> None:
    """indic-er refuses to accept a full identifier over the wire."""
    response = client.post(
        "/resolve",
        json={"records": [{"id": "a", "name": "Jayesh Patel", "identifier_masked": "ABCDE1234F"}]},
    )
    assert response.status_code == 422
    assert "masked" in response.text.lower()


def test_resolve_rejects_unknown_fields(client: TestClient) -> None:
    response = client.post(
        "/resolve", json={"records": [{"id": "a", "name": "X"}], "nonsense": True}
    )
    assert response.status_code == 422


# -------------------------------------------------------------------- match #


def test_match_returns_score_decision_and_explanation(client: TestClient) -> None:
    payload = client.post("/match", json=SHOWCASE_PAIR).json()
    assert payload["pair"]["decision"] == "merge"
    assert payload["pair"]["score"] > 0.9
    assert "Same entity" in payload["explanation"]
    assert payload["thresholds"]["auto_merge"] > 0


def test_match_feature_table_is_ordered_and_complete(client: TestClient) -> None:
    features = client.post("/match", json=SHOWCASE_PAIR).json()["features"]
    names = {row["name"] for row in features}
    assert {"name_string", "name_phonetic", "address_core", "pincode", "phone"} <= names
    computed = [row["contribution"] for row in features if row["computed"]]
    assert computed == sorted(computed, reverse=True)


def test_match_highlights_the_matching_tokens(client: TestClient) -> None:
    highlights = client.post("/match", json=SHOWCASE_PAIR).json()["highlights"]
    assert "jayesh" in highlights["left"]
    assert "patel" in highlights["right"]


def test_match_works_on_a_pair_blocking_would_never_propose(client: TestClient) -> None:
    payload = client.post(
        "/match",
        json={
            "left": {"id": "a", "name": "Jayesh Patel", "phone": "9825012345"},
            "right": {"id": "b", "name": "Murugan Pillai", "address": "T Nagar, Chennai 600017"},
        },
    ).json()
    assert payload["pair"]["decision"] == "reject"
    assert payload["pair"]["blocked_by"] == ["direct"]


def test_match_rejects_the_same_id_twice(client: TestClient) -> None:
    response = client.post(
        "/match",
        json={
            "left": {"id": "a", "name": "Jayesh Patel"},
            "right": {"id": "a", "name": "Jayesh Patil"},
        },
    )
    assert response.status_code == 422


def test_match_llm_explanation_is_marked_advisory(client: TestClient) -> None:
    """The LLM must never look like it decided anything."""
    payload = client.post(
        "/match", json={**SHOWCASE_PAIR, "explain_with_llm": True, "adjudicate": True}
    ).json()
    assert payload["llm_explanation"]["advisory"] is True
    assert payload["llm_explanation"]["mode"] == "mock"
    assert payload["adjudication"]["advisory"] is True
    assert payload["adjudication"]["verdict"] in {"same", "different", "unsure"}
    # The decision still comes from the rules.
    assert payload["pair"]["decision"] == "merge"


def test_match_is_deterministic_in_mock_mode(client: TestClient) -> None:
    body = {**SHOWCASE_PAIR, "explain_with_llm": True}
    first = client.post("/match", json=body).json()
    second = client.post("/match", json=body).json()
    assert first["llm_explanation"]["text"] == second["llm_explanation"]["text"]
    assert first["pair"]["score"] == second["pair"]["score"]


# ----------------------------------------------------------------- entities #


def test_list_entities(client: TestClient) -> None:
    payload = client.get("/entities", params={"limit": 5}).json()
    assert payload["total"] >= 1
    assert len(payload["entities"]) <= 5
    assert "canonical_name" in payload["entities"][0]


def test_list_entities_filters_by_size_and_kind(client: TestClient) -> None:
    merged = client.get("/entities", params={"min_size": 2, "limit": 100}).json()
    assert all(e["size"] >= 2 for e in merged["entities"])
    people = client.get("/entities", params={"kind": "person", "limit": 100}).json()
    assert all(e["kind"] == "person" for e in people["entities"])


def test_list_entities_search(client: TestClient) -> None:
    payload = client.get("/entities", params={"q": "patel", "limit": 100}).json()
    for entity in payload["entities"]:
        haystack = (entity["canonical_name"] + " ".join(entity["variants"])).lower()
        assert "patel" in haystack


def test_get_entity_by_entity_id(client: TestClient) -> None:
    listing = client.get("/entities", params={"min_size": 2, "limit": 1}).json()
    entity_id = listing["entities"][0]["entity_id"]
    payload = client.get(f"/entities/{entity_id}").json()
    assert payload["entity"]["entity_id"] == entity_id
    assert len(payload["members"]) == payload["entity"]["size"]
    assert payload["graph"]["elements"]
    assert payload["provenance"]
    assert all(
        {"source", "timestamp", "confidence", "method"} <= set(row) for row in payload["provenance"]
    )


def test_get_entity_by_member_record_id(client: TestClient) -> None:
    """A caller who has a record id should not have to look up the entity."""
    listing = client.get("/entities", params={"min_size": 2, "limit": 1}).json()
    entity = listing["entities"][0]
    payload = client.get(f"/entities/{entity['member_ids'][0]}").json()
    assert payload["entity"]["entity_id"] == entity["entity_id"]


def test_get_unknown_entity_is_404(client: TestClient) -> None:
    assert client.get("/entities/ent_nope").status_code == 404


def test_entity_graph_endpoint(client: TestClient) -> None:
    listing = client.get("/entities", params={"min_size": 2, "limit": 1}).json()
    entity_id = listing["entities"][0]["entity_id"]
    payload = client.get(f"/graph/{entity_id}").json()
    kinds = {e["group"] for e in payload["elements"]}
    assert "nodes" in kinds
    assert client.get("/graph/nope").status_code == 404


# ------------------------------------------------------------------- review #


def test_review_queue_shape(client: TestClient) -> None:
    payload = client.get("/review-queue", params={"limit": 5}).json()
    assert "items" in payload
    assert set(payload["counts"]) == {"pending", "accepted", "rejected", "total"}
    assert "note" in payload["threshold_shift"]


def test_review_queue_items_carry_both_records(client: TestClient) -> None:
    payload = client.get("/review-queue", params={"limit": 3}).json()
    if not payload["items"]:
        pytest.skip("no borderline pairs in this slice of the seeded data")
    item = payload["items"][0]
    assert item["left"] and item["right"]
    assert item["features"]


def test_review_decision_round_trip(client: TestClient) -> None:
    queue = client.get("/review-queue", params={"limit": 1}).json()
    if not queue["items"]:
        pytest.skip("no borderline pairs in this slice of the seeded data")
    pair_id = queue["items"][0]["pair_id"]

    response = client.post(f"/review/{pair_id}", json={"decision": "accept", "reviewer": "asha"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["decided"]["decision"] == "accept"
    assert payload["decided"]["reviewer"] == "asha"
    assert payload["counts"]["accepted"] >= 1
    assert payload["overrides_applied"] >= 1

    undo = client.post(f"/review/{pair_id}/undo")
    assert undo.status_code == 200
    assert undo.json()["counts"]["accepted"] == payload["counts"]["accepted"] - 1


def test_review_rejects_an_unknown_pair(client: TestClient) -> None:
    response = client.post("/review/a::b", json={"decision": "accept"})
    assert response.status_code == 404


def test_review_rejects_an_invalid_decision(client: TestClient) -> None:
    response = client.post("/review/a::b", json={"decision": "maybe"})
    assert response.status_code == 422


def test_review_undo_on_an_unknown_pair_is_404(client: TestClient) -> None:
    assert client.post("/review/x::y/undo").status_code == 404


# ------------------------------------------------------------------ metrics #


def test_metrics_separates_seeded_from_evaluation(client: TestClient) -> None:
    """Live demo numbers and ground-truth numbers must not be conflated."""
    payload = client.get("/metrics").json()
    assert payload["seeded"]["records"] > 0
    assert "clustering" in payload["seeded"]
    assert payload["thresholds"]["auto_merge"] > 0
    assert payload["features"]
    assert isinstance(payload["evaluation_available"], bool)
    if not payload["evaluation_available"]:
        assert "make eval" in payload["note"]


def test_metrics_lists_every_feature_with_its_weight(client: TestClient) -> None:
    features = client.get("/metrics").json()["features"]
    assert {f["name"] for f in features} >= {
        "name_string",
        "name_phonetic",
        "address_core",
        "house_number",
        "pincode",
        "phone",
    }
    assert all(f["weight"] > 0 for f in features)
    assert all(f["description"] for f in features)


# --------------------------------------------------------------- demo data #


def test_showcase_is_served(client: TestClient) -> None:
    payload = client.get("/showcase").json()
    assert payload["count"] >= 10
    names = [r["name"] for r in payload["records"]]
    assert any("જયેશ" in name for name in names), "expected a Gujarati-script record"
    assert any("Pvt" in name for name in names), "expected a business record"


def test_seed_info(client: TestClient) -> None:
    payload = client.get("/seed").json()
    assert payload["records"] > 0
    assert payload["entities"] > 0
    assert payload["graph"]["nodes"] > 0


def test_reseed(client: TestClient) -> None:
    payload = client.post("/seed/resolve").json()
    assert payload["entities"] > 0


# ------------------------------------------------------------------- web app #


def test_the_web_app_is_served_from_the_same_origin(client: TestClient) -> None:
    """One origin means the demo needs no CORS setup and no second process."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "indic-er" in response.text.lower()
