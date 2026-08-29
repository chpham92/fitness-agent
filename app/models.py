"""
Pydantic models.

Split into three intentional groups, because they get validated at
different points in the request lifecycle and mixing them up is a
common tell in reviews:

1. Tool ARGUMENT models — validate what the LLM sends us before we ever
   touch the database. This is the boundary that has to be paranoid.
2. Tool RESULT models — what we hand back to the LLM as a tool_result.
   Not strictly required to be Pydantic, but keeping them typed catches
   bugs where a tool silently changes shape.
3. Structured OUTPUT model — the final answer, forced via tool_choice on
   a terminal "emit_plan" tool so the LLM can't free-text its way around
   the schema.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Tool argument models (LLM -> us)
# ---------------------------------------------------------------------------


class LookUpExerciseArgs(BaseModel):
    """Args for the look_up_exercise tool. Accepts a name OR a muscle group,
    not both empty — that's the case worth rejecting before hitting the DB."""

    query: str = Field(
        ...,
        min_length=2,
        max_length=64,
        description="Exercise name or muscle group to search for, e.g. 'squat' or 'hamstrings'.",
    )

    @field_validator("query")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query cannot be empty or whitespace")
        return v


class LogSetArgs(BaseModel):
    """Args for the log_set tool. Weight/reps bounds are deliberately loose
    but non-negative — this is where a naive agent build skips validation
    and lets the LLM write garbage into the DB."""

    user_id: str = Field(..., min_length=1, max_length=64)
    exercise: str = Field(..., min_length=2, max_length=64)
    weight: float = Field(..., ge=0, le=2000, description="Weight in lbs.")
    reps: int = Field(..., ge=1, le=200)
    logged_at: Optional[date] = Field(
        default=None, description="Defaults to today if omitted."
    )


class GetRecentHistoryArgs(BaseModel):
    """Args for the get_recent_history tool."""

    user_id: str = Field(..., min_length=1, max_length=64)
    exercise: Optional[str] = Field(
        default=None, description="Filter to one exercise, or omit for all."
    )
    limit: int = Field(default=10, ge=1, le=100)


# ---------------------------------------------------------------------------
# Tool result models (us -> LLM)
# ---------------------------------------------------------------------------


class ExerciseRecord(BaseModel):
    name: str
    muscle_group: str
    equipment: str
    description: str


class ExerciseLookupResult(BaseModel):
    matches: list[ExerciseRecord]
    count: int


class LogSetResult(BaseModel):
    success: bool
    set_id: int
    exercise: str
    weight: float
    reps: int
    logged_at: date


class SetRecord(BaseModel):
    exercise: str
    weight: float
    reps: int
    logged_at: date


class HistoryResult(BaseModel):
    user_id: str
    sets: list[SetRecord]
    count: int


# ---------------------------------------------------------------------------
# Structured final output (forced via tool_choice on the terminal tool)
# ---------------------------------------------------------------------------


class Intensity(str, Enum):
    light = "light"
    moderate = "moderate"
    heavy = "heavy"


class PlannedExercise(BaseModel):
    exercise: str
    sets: int = Field(..., ge=1, le=10)
    reps: int = Field(..., ge=1, le=50)
    intensity: Intensity
    notes: Optional[str] = Field(default=None, max_length=280)


class WorkoutPlanResponse(BaseModel):
    """The schema every /chat turn ultimately has to resolve to. Forcing
    this via tool_choice (rather than parsing free text) is the actual
    'structured output' requirement — not just 'the model tends to
    return JSON.'"""

    summary: str = Field(..., max_length=500)
    plan: list[PlannedExercise]
    history_checked: bool = Field(
        ...,
        description=(
            "True iff get_recent_history was called earlier in this turn — "
            "regardless of whether it returned any sets. Does not mean "
            "history had data, or that it changed the plan; only that the "
            "check happened. (Previously named based_on_history, which "
            "implied more than that: it was true even for a brand-new user "
            "with zero logged sets, since the tool had still been called.)"
        ),
    )
    generated_at: datetime
