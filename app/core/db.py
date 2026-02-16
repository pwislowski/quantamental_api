from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlmodel import SQLModel

from app.core.config import database_config

# libSQL/SQLite-friendly naming convention for indexes #
INDEXES_NAMING_CONVENTION = {
    "ix": "%(column_0_label)s_idx",
    "uq": "%(table_name)s_%(column_0_name)s_key",
    "ck": "%(table_name)s_%(constraint_name)s_check",
    "fk": "%(table_name)s_%(column_0_name)s_fkey",
    "pk": "%(table_name)s_pkey",
}

# Update the existing SQLModel.metadata in-place instead of replacing it,
# so model imports work regardless of order.
SQLModel.metadata.naming_convention = INDEXES_NAMING_CONVENTION
metadata = SQLModel.metadata


def _build_libsql_url(raw_url: str) -> str:
    url = raw_url.strip()

    if url.startswith("sqlite+libsql://"):
        base_url = url
    elif url.startswith("libsql://"):
        base_url = "sqlite+libsql://" + url[len("libsql://") :]
    elif url.startswith("https://"):
        base_url = "sqlite+libsql://" + url[len("https://") :]
    elif url.startswith("http://"):
        base_url = "sqlite+libsql://" + url[len("http://") :]
    else:
        base_url = f"sqlite+libsql://{url}"

    if "secure=" in base_url:
        return base_url

    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}secure=true"


_engine = None
_sessionmaker = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            _build_libsql_url(database_config.DATABASE_URL),
            echo=False,
            connect_args={
                "auth_token": database_config.DATABASE_TOKEN,
            },
        )
    return _engine


def get_sessionmaker():
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = sessionmaker(
            autoflush=False,
            expire_on_commit=False,
            bind=get_engine(),
        )
    return _sessionmaker


# Alias for Alembic / external use
Base = SQLModel


def get_db() -> Generator[Session, None, None]:
    factory = get_sessionmaker()
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
