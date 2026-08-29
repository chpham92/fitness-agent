from __future__ import annotations

from datetime import date

from sqlalchemy.ext.asyncio import AsyncEngine

from app.db import exercise_exists, sets, utcnow
from app.models import LogSetArgs, LogSetResult

ANTHROPIC_TOOL_SCHEMA = {
    "name": "log_set",
    "description": (
        "Log a completed set for a user: exercise, weight, and reps. "
        "Only call this when the user has actually reported doing a set, "
        "not when just planning one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "exercise": {"type": "string"},
            "weight": {"type": "number", "description": "Weight in lbs."},
            "reps": {"type": "integer"},
            "logged_at": {
                "type": "string",
                "description": "ISO date (YYYY-MM-DD). Omit to default to today.",
            },
        },
        "required": ["user_id", "exercise", "weight", "reps"],
    },
}


class UnknownExerciseError(ValueError):
    """Raised when log_set is called for an exercise not in the library.

    Deliberately a distinct exception (not a generic ValueError) so the
    agent loop can catch it and feed a useful correction back to the LLM
    instead of a raw traceback.
    """


async def log_set(engine: AsyncEngine, args: LogSetArgs) -> LogSetResult:
    if not await exercise_exists(engine, args.exercise):
        raise UnknownExerciseError(
            f"'{args.exercise}' is not in the exercise library. "
            "Call look_up_exercise first to find the correct name."
        )

    logged_at = args.logged_at or date.today()
    now = utcnow()

    async with engine.begin() as conn:
        result = await conn.execute(
            sets.insert().values(
                user_id=args.user_id,
                exercise_name=args.exercise,
                weight=args.weight,
                reps=args.reps,
                logged_at=logged_at,
                created_at=now,
            )
        )
        set_id = result.inserted_primary_key[0]

    return LogSetResult(
        success=True,
        set_id=set_id,
        exercise=args.exercise,
        weight=args.weight,
        reps=args.reps,
        logged_at=logged_at,
    )
