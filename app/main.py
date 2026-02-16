from fastapi import FastAPI

from app.api.v1.factors import router as factors_router
from app.api.v1.portfolios import router as portfolio_router
from app.core.config import config
from app.core.exceptions import setup_exception_handlers
from app.core.logger import setup_logging
from app.core.security import setup_security_middleware

setup_logging()

app = FastAPI(
    title="Quantamental API",
    version=config.VERSION,
    docs_url="/docs" if config.SHOW_DOCS else None,
    redoc_url="/redoc" if config.SHOW_DOCS else None,
)

setup_exception_handlers(app)
setup_security_middleware(app)

app.include_router(factors_router)
app.include_router(portfolio_router)


@app.get("/")
async def root() -> dict:
    return {"message": "hello"}


@app.get("/health")
async def healthcheck() -> dict:
    return {"status": "ok"}
