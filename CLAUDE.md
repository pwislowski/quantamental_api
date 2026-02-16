# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Development Commands

### Environment Setup
```bash
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv sync

# Install dev dependencies
uv sync --dev
```

### Running the Application
```bash
# Run FastAPI app with auto-reload
just app
# OR
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Run background workers
just workers
# OR
uv run pgq run app.workers.main:main --restart-on-failure

# Run with Docker Compose (includes postgres)
just compose
# OR
docker compose up
```

### Testing
```bash
# run all tests
uv run pytest

# run tests with verbose output
uv run pytest -v

# run specific test file
uv run pytest tests/test_specific.py
```

### Code Quality
```bash
# Format code with ruff
just ruff
# OR
uv run ruff format

# Lint code
uv run ruff check

# Fix linting issues automatically
uv run ruff check --fix
```

## Architecture Overview

### Core Structure
This is a modern FastAPI application for a finance dashboard with factor-optimized portfolios, portfolio optimization, and backtesting.

- **Modular API Design**: API routes organized in `app/api/v1/` with versioning
- **Service Layer**: Business logic separated in `app/services/`
- **Repository/Domain Layer**: Domain models and database access in `app/models/`
- **Dependency Injection**: FastAPI dependencies for auth, database, and shared logic
- **Background Jobs**: pgQueuer integration for async task processing (e.g., long-running backtests)

### Key Components

**Configuration Management (`app/core/config.py`)**:
- Environment-specific config loading (development, staging, production)
- Pydantic settings validation

**Database Layer (SQLModel)**:
- SQLModel models and SQLAlchemy-based engine/session
- Alembic for database migrations
- Organized model structure with separate schemas and functions per domain

**Analytics/Finance Domain**:
- Factor models, portfolio constraints, and optimization inputs
- Backtest definitions, runs, and results stored as structured tables
- Heavy analytics should run in workers, not in request threads

**Monitoring & Observability**:
- Structured logging with structlog
- Sentry integration for error tracking
- Request correlation IDs via asgi-correlation-id

### Directory Structure
```
app/
├── api/           # API routes and endpoints
│   ├── v1/        # Versioned API routes
│   └── webhooks/  # Webhook handlers
├── core/          # Core functionality (config, db, logging, etc.)
├── models/        # SQLModel models, schemas, and domain logic
│   ├── user/      # User domain (models, schemas, dependencies)
│   │       ├── dependencies.py # Dependency injection functions
│   │       ├── functions.py    # Reusable domain functions
│   │       ├── models.py       # SQLModel models
│   │       └── schemas.py      # Pydantic schemas and dataclasses
│   └── portfolio/ # Portfolio/backtest domains
├── services/      # Business logic services
├── utils/         # Utility functions and helpers
└── integrations/  # External service integrations
```

### Testing Strategy
- Comprehensive test setup with database fixtures using @fixtures in /fixtures folder
- Mock implementations for external services (pgQueuer, rate limiter)
- Mocks are done this way: `mocker.patch("app.utils.communications.sms.get_twilio_client")` (no decorators)
- Separate test database with automatic cleanup
- Dependency override patterns for isolating tests

### Environment Configuration
- Development: Uses `.env` file for local configuration
- Staging/Production: Environment variables or secret managers
- Environment-specific domains and settings via `_add_non_secret_variables()`

## Important Notes

### Database
- PostgreSQL is required for this application
- SQLModel is the ORM (built on SQLAlchemy)
- Uses pgQueuer for background job processing (requires PostgreSQL extensions)
- Test database runs on port 5433 (configured in docker-compose.yml)

### Development Workflow
- Use `justfile` commands for common development tasks
- `justfile` is like Makefile - a place to automate certain commands
- Code formatting enforced with ruff (line length: 120)
- Tests use pytest with asyncio support
- Background workers run separately from the main API process

---

# FastAPI best practices to follow

## Async Routes
FastAPI is an async framework. Prefer non-blocking I/O.

Always prefer to use non blocking packages like `httpx.AsyncClient` over `requests`.

Same rules apply for endpoints and dependencies:
`def` -> for blocking operations
`async` -> for light or non-blocking logic

### I/O Intensive Tasks
- Sync routes run in the threadpool.
- Async routes should only await non-blocking I/O.

Avoid blocking calls inside `async` routes.

### CPU Intensive Tasks
Finance analytics (optimization, backtesting) are CPU-heavy.
- Do not run CPU-heavy work inside request handlers.
- Offload to workers (pgQueuer) and return job IDs/statuses.

## SQLModel (ORM) Guidance

### Model Design
- Use SQLModel for ORM objects; avoid direct SQLAlchemy model declarations.
- Keep SQLModel `table=True` models in `models.py`.
- Keep Pydantic request/response schemas in `schemas.py`.

### Sessions and Engines
- Use a single configured engine and session dependency.
- Always use the session from the dependency; do not create ad-hoc sessions.

### Migrations
- Alembic is still used. Ensure metadata is from SQLModel.
- Migrations must be static and revertable.

### Query Patterns
- Prefer SQLModel select queries (SQLAlchemy Core under the hood).
- Keep queries close to the domain (models/services), not in routes.

## Finance Dashboard Domain Rules

### Data Integrity
- Treat financial time series as immutable; prefer append-only with versioning.
- Ensure timestamps are timezone-aware (UTC).
- Normalize identifiers (tickers, instruments) consistently.

### Optimization & Backtesting
- Validate all inputs (constraints, objective functions, rebalance frequency).
- Use deterministic seeds where randomness is involved.
- Persist backtest inputs and outputs for reproducibility.

### Performance
- Heavy analytics and factor computations must run in background workers.
- API endpoints should return summaries or job statuses.

### Observability
- Log job IDs and parameters for backtests and optimizations.
- Surface domain errors with clear messages (e.g., infeasible constraints).

## Dependencies

### Prefer `async` dependencies
FastAPI supports both `sync` and `async` dependencies. Use `async` unless you are calling blocking code.

### Dependency reuse
Chain and reuse dependencies to avoid duplication. Dependencies are cached per request.

## Miscellaneous

### FastAPI response serialization
Do not return Pydantic model instances from endpoints; return dicts and let `response_model` handle validation/serialization.

### If you must use sync SDK, then run it in a thread pool
Use `run_in_threadpool` for sync-only SDKs.

### Docs
- Hide docs in non-local environments by default.
- Provide `response_model`, `status_code`, and descriptions.

### DB naming conventions
- lower_case_snake
- plural table names (e.g., `portfolios`, `backtests`)
- `_at` for datetime, `_date` for date
- consistent foreign keys (e.g., `portfolio_id`, `backtest_id`)

### Connection pool
Always use the connection pool and session dependency.

### Use lifespan
Use `lifespan` to manage resources on startup/shutdown.
