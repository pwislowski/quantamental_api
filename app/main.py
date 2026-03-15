from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app.api.v1.analytics import router as analytics_router
from app.api.v1.backtests import router as backtests_router
from app.api.v1.factors import router as factors_router
from app.api.v1.instruments import router as instruments_router
from app.api.v1.portfolios import router as portfolio_router
from app.core.config import config
from app.core.exceptions import setup_exception_handlers
from app.core.logger import setup_logging
from app.core.security import setup_security_middleware

setup_logging()

log = structlog.get_logger()


def _seed_instruments_if_empty() -> None:
    """Populate instruments table on first run. Skips if already populated."""
    from sqlmodel import select

    from app.core.db import get_sessionmaker
    from app.models.instrument import Instrument
    from app.services.instrument_sync import InstrumentSyncService

    session = get_sessionmaker()()
    try:
        existing = session.execute(select(Instrument).limit(1)).first()
        if existing is None:
            log.info("instruments_table_empty_seeding")
            InstrumentSyncService(session).sync()
    except Exception:
        log.warning("startup_instrument_seed_failed", exc_info=True)
    finally:
        session.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(_seed_instruments_if_empty)
    yield


app = FastAPI(
    title="Quantamental API",
    version=config.VERSION,
    docs_url="/docs" if config.SHOW_DOCS else None,
    redoc_url="/redoc" if config.SHOW_DOCS else None,
    lifespan=lifespan,
)

setup_exception_handlers(app)
setup_security_middleware(app)

app.include_router(analytics_router)
app.include_router(backtests_router)
app.include_router(factors_router)
app.include_router(instruments_router)
app.include_router(portfolio_router)


@app.get("/")
async def root() -> dict:
    return {"message": "hello"}


@app.get("/health")
async def healthcheck() -> dict:
    return {"status": "ok"}
