import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pytest_mock import MockerFixture
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

################################################################################
# mock rate limiter #
################################################################################

# mock rate limiter to bypass rate limiting in tests #
mock_limiter = MagicMock()
mock_limiter.limit.return_value = lambda f: f  # Pass-through decorator
sys.modules["app.core.rate_limiter"] = MagicMock(limiter=mock_limiter)


################################################################################
# mock pgqueuer #
################################################################################

# mock pgqueuer components
mock_sync_driver = MagicMock()
mock_sync_queries = MagicMock()
mock_sync_queries.enqueue = MagicMock(return_value="mock-job-id")
mock_sync_queries.dequeue = MagicMock(return_value=None)
sys.modules["pgqueuer.db"] = MagicMock(SyncPsycopgDriver=MagicMock(return_value=mock_sync_driver))
sys.modules["pgqueuer.queries"] = MagicMock(SyncQueries=MagicMock(return_value=mock_sync_queries))


# create mock lifespan that doesn't connect to external databases #
@asynccontextmanager
async def mock_lifespan(app: FastAPI):
    mock_queries = MagicMock()
    mock_queries.enqueue = MagicMock(return_value="mock-job-id")  # sync method
    mock_queries.dequeue = MagicMock(return_value=None)  # sync method

    app.extra["pgq_queries"] = mock_queries
    yield


################################################################################
from app.main import app  # noqa: E402

app.router.lifespan_context = mock_lifespan

from app.core.db import Base, get_db  # noqa: E402

TEST_JWT_SECRET_KEY = "test-jwt-secret-key"
TEST_JWT_ALGORITHM = "HS256"
MOCK_USER_ID = "4ef80032-9b19-4948-8f1f-69758f35a70a"


def _make_sqlite_engine():
    """Create a shared in-memory SQLite engine safe for cross-thread use."""
    engine = create_engine(
        "sqlite:///file:test.db?mode=memory&cache=shared&uri=true",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    # Enable WAL mode and foreign keys for SQLite
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


@pytest.fixture(scope="function")
def test_engine():
    engine = _make_sqlite_engine()
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def test_session(test_engine) -> Generator[Session, None, None]:
    SessionLocal = sessionmaker(
        bind=test_engine,
        expire_on_commit=False,
        autoflush=False,
    )

    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest_asyncio.fixture(scope="function")
async def client(test_session) -> AsyncGenerator[AsyncClient, None]:
    def override_get_db():
        yield test_session

    # override the FastAPI dependency injection #
    app.dependency_overrides[get_db] = override_get_db

    host, port = "127.0.0.1", "9000"
    async with AsyncClient(
        transport=ASGITransport(app=app, client=(host, port)), base_url="http://test"
    ) as test_client:
        yield test_client

    # clean up dependency overrides after test
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def unauthorized_client(test_session, mock_sync_queries) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client without auth override - tests unauthorized access."""

    def override_get_db():
        yield test_session

    # only override database, not auth
    app.dependency_overrides[get_db] = override_get_db

    host, port = "127.0.0.1", "9000"
    async with AsyncClient(
        transport=ASGITransport(app=app, client=(host, port)), base_url="http://test"
    ) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def mock_sync_queries():
    """provide mock sync queries for tests"""
    mock_queries = MagicMock()
    mock_queries.enqueue = MagicMock(return_value="mock-job-id")
    mock_queries.dequeue = MagicMock(return_value=None)
    return mock_queries


@pytest.fixture(scope="function", autouse=True)
def mock_jwt_settings(mocker: MockerFixture):
    """mock JWT settings globally for all tests"""
    try:
        mocker.patch("app.utils.jwt.JWT_SECRET_KEY", TEST_JWT_SECRET_KEY)
        mocker.patch("app.utils.jwt.JWT_ALGORITHM", TEST_JWT_ALGORITHM)
    except (AttributeError, ModuleNotFoundError):
        pass  # app.utils.jwt not available (auth removed)
