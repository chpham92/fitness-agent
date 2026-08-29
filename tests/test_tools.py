from datetime import date

import pytest
from pydantic import ValidationError

from app.models import GetRecentHistoryArgs, LogSetArgs, LookUpExerciseArgs
from app.tools.exercise_lookup import look_up_exercise
from app.tools.history import get_recent_history
from app.tools.log_set import UnknownExerciseError, log_set

# ---------------------------------------------------------------------------
# look_up_exercise
# ---------------------------------------------------------------------------


async def test_lookup_by_exact_name(seeded_engine):
    args = LookUpExerciseArgs(query="Back Squat")
    result = await look_up_exercise(seeded_engine, args)
    assert result.count == 1
    assert result.matches[0].name == "Back Squat"


async def test_lookup_by_muscle_group_returns_multiple(seeded_engine):
    args = LookUpExerciseArgs(query="quads")
    result = await look_up_exercise(seeded_engine, args)
    assert result.count >= 2
    assert all(m.muscle_group == "quads" for m in result.matches)


async def test_lookup_case_insensitive_partial(seeded_engine):
    args = LookUpExerciseArgs(query="squat")
    result = await look_up_exercise(seeded_engine, args)
    names = {m.name for m in result.matches}
    assert "Back Squat" in names
    assert "Goblet Squat" in names


async def test_lookup_no_match_returns_empty_not_error(seeded_engine):
    args = LookUpExerciseArgs(query="zzz_nonexistent")
    result = await look_up_exercise(seeded_engine, args)
    assert result.count == 0
    assert result.matches == []


def test_lookup_args_reject_empty_query():
    with pytest.raises(ValidationError):
        LookUpExerciseArgs(query="  ")


def test_lookup_args_reject_too_short():
    with pytest.raises(ValidationError):
        LookUpExerciseArgs(query="a")


# ---------------------------------------------------------------------------
# log_set
# ---------------------------------------------------------------------------


async def test_log_set_happy_path(seeded_engine):
    args = LogSetArgs(user_id="chris", exercise="Back Squat", weight=225, reps=5)
    result = await log_set(seeded_engine, args)
    assert result.success is True
    assert result.exercise == "Back Squat"
    assert result.logged_at == date.today()


async def test_log_set_unknown_exercise_raises(seeded_engine):
    args = LogSetArgs(user_id="chris", exercise="Not A Real Exercise", weight=100, reps=5)
    with pytest.raises(UnknownExerciseError):
        await log_set(seeded_engine, args)


def test_log_set_args_reject_negative_weight():
    with pytest.raises(ValidationError):
        LogSetArgs(user_id="chris", exercise="Back Squat", weight=-10, reps=5)


def test_log_set_args_reject_zero_reps():
    with pytest.raises(ValidationError):
        LogSetArgs(user_id="chris", exercise="Back Squat", weight=100, reps=0)


def test_log_set_args_reject_absurd_weight():
    # 5000 lbs is not a real barbell lift; this is exactly the kind of
    # malformed/hallucinated tool arg Pydantic should catch before it
    # ever reaches the database.
    with pytest.raises(ValidationError):
        LogSetArgs(user_id="chris", exercise="Back Squat", weight=5000, reps=5)


def test_log_set_args_reject_missing_required_field():
    with pytest.raises(ValidationError):
        LogSetArgs(user_id="chris", weight=100, reps=5)  # missing exercise


# ---------------------------------------------------------------------------
# get_recent_history
# ---------------------------------------------------------------------------


async def test_history_empty_for_new_user(seeded_engine):
    args = GetRecentHistoryArgs(user_id="brand_new_user")
    result = await get_recent_history(seeded_engine, args)
    assert result.count == 0


async def test_history_returns_logged_sets(seeded_engine):
    await log_set(
        seeded_engine,
        LogSetArgs(user_id="chris", exercise="Back Squat", weight=225, reps=5),
    )
    await log_set(
        seeded_engine,
        LogSetArgs(user_id="chris", exercise="Bench Press", weight=185, reps=5),
    )

    result = await get_recent_history(seeded_engine, GetRecentHistoryArgs(user_id="chris"))
    assert result.count == 2
    exercises_logged = {s.exercise for s in result.sets}
    assert exercises_logged == {"Back Squat", "Bench Press"}


async def test_history_filters_by_exercise(seeded_engine):
    await log_set(
        seeded_engine,
        LogSetArgs(user_id="chris", exercise="Back Squat", weight=225, reps=5),
    )
    await log_set(
        seeded_engine,
        LogSetArgs(user_id="chris", exercise="Bench Press", weight=185, reps=5),
    )

    result = await get_recent_history(
        seeded_engine, GetRecentHistoryArgs(user_id="chris", exercise="Back Squat")
    )
    assert result.count == 1
    assert result.sets[0].exercise == "Back Squat"


async def test_history_respects_limit(seeded_engine):
    for i in range(5):
        await log_set(
            seeded_engine,
            LogSetArgs(user_id="chris", exercise="Back Squat", weight=200 + i, reps=5),
        )

    result = await get_recent_history(
        seeded_engine, GetRecentHistoryArgs(user_id="chris", limit=2)
    )
    assert result.count == 2


async def test_history_isolated_per_user(seeded_engine):
    await log_set(
        seeded_engine,
        LogSetArgs(user_id="chris", exercise="Back Squat", weight=225, reps=5),
    )
    result = await get_recent_history(seeded_engine, GetRecentHistoryArgs(user_id="someone_else"))
    assert result.count == 0
