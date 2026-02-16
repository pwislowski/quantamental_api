import os
from logging.config import fileConfig

from sqlalchemy import create_engine
from sqlmodel import SQLModel

# Import model modules so their tables register on SQLModel.metadata for autogenerate
import app.models.factor  # noqa: F401
import app.models.portfolio  # noqa: F401
from alembic import context
from app.core.config import database_config
from app.core.db import _build_libsql_url

config = context.config

if config.config_file_name and os.path.exists(config.config_file_name):
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def get_engine():
    return create_engine(
        _build_libsql_url(database_config.DATABASE_URL),
        connect_args={"auth_token": database_config.DATABASE_TOKEN},
    )


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    url = _build_libsql_url(database_config.DATABASE_URL)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = get_engine()
    with engine.connect() as connection:
        do_run_migrations(connection)
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
