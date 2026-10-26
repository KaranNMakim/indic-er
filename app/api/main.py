"""FastAPI application.

Serves the API and the web app from one origin, so the demo needs no CORS
configuration, no second process and no proxy.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.logging_conf import configure_logging
from app.core.config import WEB_DIR, settings
from app.kg.ontology import load_ontology

log = logging.getLogger("indic_er.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = settings()
    configure_logging(config.log_level, config.log_json)
    # Fail fast: a bad ontology or an unparseable config file should stop the
    # process at boot, not surface as a 500 on the first request.
    ontology = load_ontology()
    from app.core.config import (
        address_config,
        names_config,
        phonetic_config,
        translit_config,
        weights_config,
    )

    weights = weights_config()
    names_config()
    phonetic_config()
    address_config()
    translit_config()

    from app.api.state import get_state

    state = get_state()
    log.info(
        "startup",
        extra={
            "version": __version__,
            "ontology": f"{ontology.name} {ontology.version}",
            "classes": len(ontology.classes),
            "relations": len(ontology.relations),
            "llm_mode": config.llm_mode,
            "auto_merge": weights.auto_merge,
            "review_low": weights.review_low,
            "seeded_records": len(state.records),
        },
    )
    yield
    log.info("shutdown")


app = FastAPI(
    title="indic-er",
    version=__version__,
    summary="Entity resolution for messy Indian names, businesses and addresses",
    description=(
        "A proof of value. Resolve batches of records into canonical entities, "
        "score a single pair with a per-feature explanation, work a review "
        "queue, and compare the pipeline against a plain fuzzy-match baseline."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_log(request: Request, call_next) -> Response:
    """One structured line per request, with a correlation id."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    response.headers["x-request-id"] = request_id
    # API responses are never cacheable. `/metrics` in particular is read from
    # data/eval.json on every request, so a browser serving a stale copy shows
    # numbers from a previous evaluation run — which is exactly the kind of
    # quietly-wrong output a demo must not produce.
    if not request.url.path.startswith("/static"):
        response.headers.setdefault("Cache-Control", "no-store")
    if request.url.path not in {"/health"} and not request.url.path.startswith("/static"):
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": elapsed_ms,
            },
        )
    return response


@app.get("/health", tags=["meta"], summary="Liveness and configuration snapshot")
def health() -> dict[str, object]:
    """Enough to tell whether the service is usable, not just alive."""
    from app.api.state import get_state
    from app.core.config import weights_config

    config = settings()
    weights = weights_config()
    state = get_state()
    ontology = load_ontology()
    return {
        "status": "ok",
        "version": __version__,
        "ontology": {"name": ontology.name, "version": ontology.version},
        "llm_mode": config.llm_mode,
        "llm_model": state.llm.model,
        "thresholds": {"auto_merge": weights.auto_merge, "review_low": weights.review_low},
        "blocking": list(weights.blocking_strategies),
        "features": sorted(weights.features),
        "seeded_records": len(state.records),
        "graph": state.graph.stats(),
        "review_queue": state.review.counts(),
    }


@app.get("/ontology", tags=["meta"], summary="The ontology, as JSON, JSON-LD or Turtle")
def ontology_endpoint(format: str = "json") -> Response:
    """``format=json`` (default), ``json-ld`` or ``turtle``."""
    ontology = load_ontology()
    fmt = format.lower().replace("_", "-")
    if fmt in {"turtle", "ttl"}:
        return PlainTextResponse(ontology.to_turtle(), media_type="text/turtle")
    if fmt in {"json-ld", "jsonld"}:
        return Response(ontology.to_jsonld(), media_type="application/ld+json")
    return JSONResponse(ontology.to_summary())


from app.api.routes import router  # noqa: E402  (router needs `app` settings loaded)

app.include_router(router)


# --------------------------------------------------------------------------- #
# Web app
# --------------------------------------------------------------------------- #

class NoCacheStaticFiles(StaticFiles):
    """Static files that always revalidate.

    Starlette's StaticFiles sends an ETag and a Last-Modified but no
    Cache-Control, so browsers apply *heuristic* freshness and happily serve a
    stale asset without asking. On a demo that is a trap: edit a file, reload,
    and see the old build. `no-cache` still allows a 304 against the ETag, so
    this costs a conditional request, not a re-download.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response


if WEB_DIR.exists():
    app.mount("/static", NoCacheStaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
