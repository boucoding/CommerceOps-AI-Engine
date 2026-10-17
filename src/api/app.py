"""
FastAPI service exposing the same pipeline as the batch CLI.

Endpoints
    GET  /healthz                      liveness
    GET  /readyz                       readiness (database reachable)
    POST /v1/emails                    process one email (idempotent on email_id)
    POST /v1/emails/batch              process up to API_MAX_BATCH emails
    GET  /v1/emails/{email_id}         stored outcome
    GET  /v1/products/{product_id}     catalog item with live stock
    GET  /v1/products?query=...        hybrid product search (same as the agent's search_catalog tool)

When API_KEY is configured every /v1 route requires the ``X-API-Key`` header.
"""

from __future__ import annotations

import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator, Awaitable, Callable, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse

from src import __version__
from src.api.schemas import BatchIn, BatchOut, EmailIn, OutcomeOut, ProductOut, SearchHit, SearchOut
from src.config import Settings, get_settings
from src.engine import Engine, build_engine
from src.models.domain import CustomerEmail
from src.utils.logger import configure_logging, correlation, get_logger
from src.utils.reporter import summarize

logger = get_logger("api")


def get_engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


EngineDep = Annotated[Engine, Depends(get_engine)]


def _to_email(item: EmailIn) -> CustomerEmail:
    return CustomerEmail(
        email_id=item.email_id or f"API-{uuid.uuid4().hex[:16]}",
        subject=item.subject.strip(),
        body=item.body.strip(),
    )


def create_app(settings: Optional[Settings] = None, engine: Optional[Engine] = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)
        app.state.engine = engine or build_engine(settings)
        logger.info("API ready (llm=%s, model=%s)", settings.LLM_MODE.value, app.state.engine.llm.model)
        try:
            yield
        finally:
            if engine is None:
                app.state.engine.close()

    app = FastAPI(
        title=settings.APP_NAME,
        version=__version__,
        description="Classifies customer emails, reserves stock for orders and drafts grounded replies.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        with correlation(rid):
            response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    def require_api_key(request: Request) -> None:
        if settings.API_KEY is None:
            return
        supplied = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(supplied, settings.API_KEY.get_secret_value()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")

    secured = [Depends(require_api_key)]

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    def readyz(engine: EngineDep) -> JSONResponse:
        ok = engine.store.ping()
        body = {
            "status": "ready" if ok else "unavailable",
            "llm_mode": engine.settings.LLM_MODE.value,
            "model": engine.llm.model,
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.post("/v1/emails", response_model=OutcomeOut, tags=["emails"], dependencies=secured)
    def process_email(item: EmailIn, engine: EngineDep) -> OutcomeOut:
        outcome = engine.process([_to_email(item)])[0]
        return OutcomeOut.from_outcome(outcome)

    @app.post("/v1/emails/batch", response_model=BatchOut, tags=["emails"], dependencies=secured)
    def process_batch(batch: BatchIn, engine: EngineDep) -> BatchOut:
        if len(batch.emails) > settings.API_MAX_BATCH:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"At most {settings.API_MAX_BATCH} emails")
        emails = [_to_email(item) for item in batch.emails]
        if len({e.email_id for e in emails}) != len(emails):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "email_id values must be unique")
        outcomes = engine.process(emails)
        return BatchOut(
            outcomes=[OutcomeOut.from_outcome(o) for o in outcomes],
            summary=summarize(outcomes, settings.LLM_PRICE_INPUT_PER_1M, settings.LLM_PRICE_OUTPUT_PER_1M),
        )

    @app.get("/v1/emails/{email_id}", response_model=OutcomeOut, tags=["emails"], dependencies=secured)
    def get_email(email_id: str, engine: EngineDep) -> OutcomeOut:
        outcome = engine.store.get_outcome(email_id)
        if outcome is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No processed email with this id")
        return OutcomeOut.from_outcome(outcome)

    @app.get("/v1/products/{product_id}", response_model=ProductOut, tags=["catalog"], dependencies=secured)
    def get_product(product_id: str, engine: EngineDep) -> ProductOut:
        found = engine.store.get_products([product_id.strip().upper()])
        if not found:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown product")
        return ProductOut.from_product(next(iter(found.values())))

    @app.get("/v1/products", response_model=SearchOut, tags=["catalog"], dependencies=secured)
    def search_products(
        engine: EngineDep,
        query: Annotated[str, Query(min_length=2, max_length=500)],
        limit: Annotated[int, Query(ge=1, le=20)] = 5,
    ) -> SearchOut:
        hits = engine.store.search(query, engine.agent.toolbox.embed(query), limit)
        return SearchOut(results=[SearchHit(**p.to_dict(), match=m) for p, m in hits])

    return app
