"""
Idempotent seed script: creates tables if missing, inserts exercises
that aren't already present by name. Safe to run repeatedly (e.g. on
every container start) without duplicating rows.

Usage: python -m app.seed
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from app.db import DATABASE_URL, exercises, get_engine, init_db

DATA_FILE = Path(__file__).parent.parent / "data" / "exercises.json"


async def seed(database_url: str = DATABASE_URL) -> int:
    engine = get_engine(database_url)
    await init_db(engine)

    with open(DATA_FILE) as f:
        rows = json.load(f)

    inserted = 0
    async with engine.begin() as conn:
        existing = await conn.execute(select(exercises.c.name))
        existing_names = {row[0] for row in existing}

        to_insert = [r for r in rows if r["name"] not in existing_names]
        if to_insert:
            await conn.execute(exercises.insert(), to_insert)
            inserted = len(to_insert)

    await engine.dispose()
    return inserted


if __name__ == "__main__":
    count = asyncio.run(seed())
    print(f"Seeded {count} new exercise(s).")
