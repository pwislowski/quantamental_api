set dotenv-load := true

default:
    @just --list

compose:
    @docker compose up

app:
    uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

ruff:
    uv run ruff format

autogen_migrate MSG:
    uv run alembic -c pyproject.toml revision --autogenerate -m "{{ MSG }}"

migrate:
    uv run alembic -c pyproject.toml upgrade head
