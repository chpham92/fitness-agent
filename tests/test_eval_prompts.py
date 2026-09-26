"""
Pure sanity checks on the eval prompt set — no API, no DB. These exist to
catch exactly the kind of silent mistake that would quietly corrupt every
downstream score: a duplicate user_id (fixture cross-contamination) or a
seed_set exercise name that doesn't match the library exactly (log_set
would raise UnknownExerciseError before the agent even runs).
"""
import json
from pathlib import Path

from eval.prompts import EVAL_PROMPTS

EXERCISE_LIBRARY = {
    e["name"]
    for e in json.loads((Path(__file__).parent.parent / "data" / "exercises.json").read_text(encoding="utf-8"))
}


def test_prompt_ids_are_unique():
    ids = [p.id for p in EVAL_PROMPTS]
    assert len(ids) == len(set(ids))


def test_prompt_user_ids_are_unique():
    # Each prompt gets its own eval_ user_id so seed_sets from one
    # scenario can never bleed into another's history within a shared
    # per-iteration DB file (see eval/db.py).
    user_ids = [p.user_id for p in EVAL_PROMPTS]
    assert len(user_ids) == len(set(user_ids))


def test_seed_set_exercises_exist_in_library():
    for prompt in EVAL_PROMPTS:
        for seed_set in prompt.seed_sets:
            assert seed_set.exercise in EXERCISE_LIBRARY, (
                f"{prompt.id}: '{seed_set.exercise}' is not an exact match in "
                f"data/exercises.json — log_set would reject this fixture."
            )


def test_every_prompt_has_a_nonblank_message_and_scenario():
    for prompt in EVAL_PROMPTS:
        assert prompt.message.strip()
        assert prompt.scenario.strip()
