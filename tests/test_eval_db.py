"""
eval/db.py's fixture-seeding, tested directly against a real (throwaway)
SQLite file — no Anthropic API involved, so this runs in every test pass
regardless of ANTHROPIC_API_KEY.
"""
from app.models import GetRecentHistoryArgs
from app.tools.history import get_recent_history
from eval.db import setup_eval_db
from eval.prompts import EVAL_PROMPTS


async def test_setup_eval_db_seeds_exercise_library(tmp_path):
    engine = await setup_eval_db(tmp_path / "eval_test.db")
    try:
        from app.tools.exercise_lookup import look_up_exercise
        from app.models import LookUpExerciseArgs

        result = await look_up_exercise(engine, LookUpExerciseArgs(query="squat"))
        assert result.count > 0
    finally:
        await engine.dispose()


async def test_setup_eval_db_inserts_each_prompts_seed_sets_under_its_own_user(tmp_path):
    engine = await setup_eval_db(tmp_path / "eval_test.db")
    try:
        for prompt in EVAL_PROMPTS:
            history = await get_recent_history(
                engine, GetRecentHistoryArgs(user_id=prompt.user_id, limit=100)
            )
            assert history.count == len(prompt.seed_sets), (
                f"{prompt.id}: expected {len(prompt.seed_sets)} seeded set(s), "
                f"found {history.count}"
            )
    finally:
        await engine.dispose()


async def test_seeded_sets_land_on_the_correct_exercise_and_weight(tmp_path):
    engine = await setup_eval_db(tmp_path / "eval_test.db")
    try:
        prompt = next(p for p in EVAL_PROMPTS if p.id == "returning_user_progression")
        history = await get_recent_history(
            engine, GetRecentHistoryArgs(user_id=prompt.user_id, limit=100)
        )
        weights = sorted(s.weight for s in history.sets)
        assert weights == sorted(s.weight for s in prompt.seed_sets)
        assert all(s.exercise == "Back Squat" for s in history.sets)
    finally:
        await engine.dispose()
