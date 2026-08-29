from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db import exercises
from app.models import ExerciseLookupResult, ExerciseRecord, LookUpExerciseArgs

ANTHROPIC_TOOL_SCHEMA = {
    "name": "look_up_exercise",
    "description": (
        "Look up exercises by name or muscle group from the exercise "
        "library. Use this before recommending an exercise you're not "
        "certain exists in the library."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Exercise name or muscle group, e.g. 'squat' or 'hamstrings'.",
            }
        },
        "required": ["query"],
    },
}


async def look_up_exercise(
    engine: AsyncEngine, args: LookUpExerciseArgs
) -> ExerciseLookupResult:
    """Case-insensitive partial match against name or muscle_group."""
    pattern = f"%{args.query.lower()}%"
    async with engine.connect() as conn:
        result = await conn.execute(
            select(exercises).where(
                or_(
                    exercises.c.name.ilike(pattern),
                    exercises.c.muscle_group.ilike(pattern),
                )
            )
        )
        rows = result.mappings().all()

    matches = [
        ExerciseRecord(
            name=row["name"],
            muscle_group=row["muscle_group"],
            equipment=row["equipment"],
            description=row["description"],
        )
        for row in rows
    ]
    return ExerciseLookupResult(matches=matches, count=len(matches))
