from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from casecrawler.api.routes import ingest, search, sources
from casecrawler.config import load_config

# Import sources for registry discovery
import casecrawler.sources.pubmed  # noqa: F401
import casecrawler.sources.openfda  # noqa: F401
import casecrawler.sources.dailymed  # noqa: F401
import casecrawler.sources.rxnorm  # noqa: F401
import casecrawler.sources.medrxiv  # noqa: F401
import casecrawler.sources.clinicaltrials  # noqa: F401
import casecrawler.sources.glass  # noqa: F401
import casecrawler.sources.annas_archive  # noqa: F401
import casecrawler.sources.firecrawl  # noqa: F401


def create_app() -> FastAPI:
    load_config()

    app = FastAPI(
        title="CaseCrawler",
        description="Medical knowledge ingestion engine",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ingest.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(sources.router, prefix="/api")

    return app


app = create_app()
