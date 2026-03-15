"""add instruments table

Revision ID: a1b2c3d4e5f6
Revises: 6e8dbc4efa4c
Create Date: 2026-03-15 15:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "6e8dbc4efa4c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "instruments",
        sa.Column("ticker", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("company_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("gics_sector", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("gics_sub_industry", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("ticker", name="instruments_pkey"),
    )


def downgrade() -> None:
    op.drop_table("instruments")
