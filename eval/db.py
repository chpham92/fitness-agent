"""
Per-iteration eval database setup: a fresh SQLite file, seeded with the
exercise library, plus every prompt's fixture sets pre-inserted under its
own eval_ user_id — so each iteration starts from an identical baseline
instead of accumulating drift (or worse, letting one iteration's real
log_set calls become the next iteration's "history").
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine

from app.db import get_engine, init_db
from app.models import LogSetArgs
from app.seed import seed
from app.tools.log_set import log_set
from eval.prompts import EVAL_PROMPTS


async def setup_eval_db(db_path: Path) -> AsyncEngine:
    """Create, seed, and fixture a fresh eval database at db_path.
    Assumes db_path doesn't already exist — callers are responsible for
    using a fresh temp file per run (see eval/runner.py)."""
    database_url = f"sqlite+aiosqlite:///{db_path}"
    engine = get_engine(database_url)
    await init_db(engine)
    await seed(database_url)

    today = date.today()
    for prompt in EVAL_PROMPTS:
        for seed_set in prompt.seed_sets:
            await log_set(
                engine,
                LogSetArgs(
                    user_id=prompt.user_id,
                    exercise=seed_set.exercise,
                    weight=seed_set.weight,
                    reps=seed_set.reps,
                    logged_at=today - timedelta(days=seed_set.days_ago),
                ),
            )

    return engine
