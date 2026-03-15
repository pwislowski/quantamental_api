"""replace backtest tickers json with backtest_instruments join table

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-15 16:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel

from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "backtest_instruments",
        sa.Column("backtest_id", sa.Integer(), nullable=False),
        sa.Column("ticker", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(["backtest_id"], ["backtests.id"]),
        sa.ForeignKeyConstraint(["ticker"], ["instruments.ticker"]),
        sa.PrimaryKeyConstraint("backtest_id", "ticker"),
    )
    op.drop_column("backtests", "tickers")


def downgrade() -> None:
    op.add_column("backtests", sa.Column("tickers", sa.JSON(), nullable=True))
    op.drop_table("backtest_instruments")
