import uuid

import pytest
import pytest_asyncio

from app.db import get_engine, init_db
from app.seed import seed


@pytest_asyncio.fixture
async def engine():
    """Each test gets its own throwaway SQLite file so tests can run in
    parallel and never see each other's state. File-based (not :memory:)
    because :memory: databases don't survive across separate connections
    in aiosqlite, and our tools open a fresh connection per call."""
    db_name = f"test_{uuid.uuid4().hex}.db"
    url = f"sqlite+aiosqlite:///{db_name}"
    eng = get_engine(url)
    await init_db(eng)
    yield eng
    await eng.dispose()
    import os

    if os.path.exists(db_name):
        os.remove(db_name)


@pytest_asyncio.fixture
async def seeded_engine(engine):
    url = str(engine.url)
    await seed(url)
    return engine
