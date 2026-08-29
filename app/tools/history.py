from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db import sets
from app.models import GetRecentHistoryArgs, HistoryResult, SetRecord

ANTHROPIC_TOOL_SCHEMA = {
    "name": "get_recent_history",
    "description": (
        "Get a user's recent logged sets, optionally filtered to one "
        "exercise. Use this before writing a plan so recommendations "
        "account for what the user has actually been doing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "exercise": {
                "type": "string",
                "description": "Optional. Filter to one exercise.",
            },
            "limit": {"type": "integer", "description": "Default 10, max 100."},
        },
        "required": ["user_id"],
    },
}


async def get_recent_history(
    engine: AsyncEngine, args: GetRecentHistoryArgs
) -> HistoryResult:
    query = (
        select(sets)
        .where(sets.c.user_id == args.user_id)
        .order_by(sets.c.logged_at.desc(), sets.c.created_at.desc())
        .limit(args.limit)
    )
    if args.exercise:
        query = query.where(sets.c.exercise_name.ilike(f"%{args.exercise}%"))

    async with engine.connect() as conn:
        result = await conn.execute(query)
        rows = result.mappings().all()

    records = [
        SetRecord(
            exercise=row["exercise_name"],
            weight=row["weight"],
            reps=row["reps"],
            logged_at=row["logged_at"],
        )
        for row in rows
    ]
    return HistoryResult(user_id=args.user_id, sets=records, count=len(records))
