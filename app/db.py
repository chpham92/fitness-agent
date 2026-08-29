"""
Database layer — SQLAlchemy Core (not the ORM) over aiosqlite.

Why Core and not the ORM: the schema is two tables with no relationships
worth mapping to objects. Core gives real async engine/connection semantics
(the thing production reviewers actually check for) without ORM ceremony
that would just be there to look impressive.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    select,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

DB_PATH = os.environ.get("FITNESS_AGENT_DB", "fitness_agent.db")
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

metadata = MetaData()

exercises = Table(
    "exercises",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False, unique=True),
    Column("muscle_group", String, nullable=False),
    Column("equipment", String, nullable=False),
    Column("description", String, nullable=False),
)

sets = Table(
    "sets",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String, nullable=False, index=True),
    Column("exercise_name", String, nullable=False),
    Column("weight", Float, nullable=False),
    Column("reps", Integer, nullable=False),
    Column("logged_at", DateTime, nullable=False),
    Column("created_at", DateTime, nullable=False),
)


def get_engine(database_url: str = DATABASE_URL) -> AsyncEngine:
    return create_async_engine(database_url, echo=False, future=True)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


async def exercise_exists(engine: AsyncEngine, name: str) -> bool:
    async with engine.connect() as conn:
        result = await conn.execute(
            select(exercises.c.id).where(exercises.c.name == name)
        )
        return result.first() is not None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
