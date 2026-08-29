"""
Integration test: a real call through the agent loop against a live
Claude model. Requires ANTHROPIC_API_KEY; skipped otherwise so the rest
of the suite still runs in CI/without credentials.
"""
import os

import anthropic
import pytest

from app.agent import run_agent_loop
from app.models import WorkoutPlanResponse

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live-API integration test",
)


async def test_agent_makes_sequential_tool_calls_before_emit_plan(seeded_engine):
    client = anthropic.AsyncAnthropic()
    trace: list[str] = []

    plan = await run_agent_loop(
        engine=seeded_engine,
        user_id="chris",
        message=(
            "I've been working on my legs lately. Check my recent history, "
            "then look up a couple of good quad exercises I haven't tried, "
            "and give me a leg day plan for today."
        ),
        client=client,
        trace=trace,
    )

    assert isinstance(plan, WorkoutPlanResponse)
    assert len(plan.plan) > 0
    assert len(trace) >= 2
