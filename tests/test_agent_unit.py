"""
Unit tests for app/agent.py that don't need ANTHROPIC_API_KEY — they either
exercise pure dispatch/validation logic directly, or inject a fake client
that fails deterministically. Kept separate from test_agent.py so they
still run without a live API key (unlike that file's integration test).
"""
import json

import anthropic
import pytest

from app.agent import AgentError, _execute_client_tool, run_agent_loop, stream_agent_events
from app.models import GetRecentHistoryArgs, LogSetArgs, WorkoutPlanResponse
from app.tools.history import get_recent_history
from app.tools.log_set import log_set
from tests.fakes import (
    EndTurnThenForcedClient,
    FailingClient,
    make_rate_limit_error,
    make_timeout_error,
)

VALID_PLAN_INPUT = {
    "summary": "Quick test plan",
    "plan": [{"exercise": "Back Squat", "sets": 3, "reps": 5, "intensity": "moderate"}],
    "history_checked": False,
    "generated_at": "2026-01-01T00:00:00Z",
}

# ---------------------------------------------------------------------------
# user_id can't be spoofed via tool_use input
# ---------------------------------------------------------------------------
#
# Pydantic validates *shape* (types, bounds), not *meaning* — a tool_use
# input with user_id="someone_else" is perfectly valid LogSetArgs on its
# own. The loop closes that gap by overwriting user_id with the value from
# the actual /chat request before validation ever sees the model's input
# (see _execute_client_tool). These tests lock that behavior in as a
# regression test rather than treating it as new work — it's been true
# since Day 2.


async def test_log_set_ignores_spoofed_user_id(seeded_engine):
    content, is_error = await _execute_client_tool(
        seeded_engine,
        user_id="real_user",
        name="log_set",
        raw_input={"user_id": "attacker", "exercise": "Back Squat", "weight": 135, "reps": 5},
    )
    assert is_error is False

    real = await get_recent_history(seeded_engine, GetRecentHistoryArgs(user_id="real_user"))
    attacker = await get_recent_history(seeded_engine, GetRecentHistoryArgs(user_id="attacker"))
    assert real.count == 1
    assert attacker.count == 0


async def test_get_recent_history_ignores_spoofed_user_id(seeded_engine):
    await log_set(
        seeded_engine, LogSetArgs(user_id="real_user", exercise="Back Squat", weight=135, reps=5)
    )

    content, is_error = await _execute_client_tool(
        seeded_engine,
        user_id="real_user",
        name="get_recent_history",
        raw_input={"user_id": "attacker", "limit": 10},
    )
    assert is_error is False
    data = json.loads(content)
    assert data["user_id"] == "real_user"
    assert data["count"] == 1


# ---------------------------------------------------------------------------
# Anthropic API errors propagate as themselves — never swallowed
# ---------------------------------------------------------------------------


async def test_run_agent_loop_propagates_rate_limit_error(seeded_engine):
    client = FailingClient(make_rate_limit_error())
    with pytest.raises(anthropic.RateLimitError):
        await run_agent_loop(engine=seeded_engine, user_id="chris", message="hi", client=client)


async def test_stream_agent_events_propagates_rate_limit_error(seeded_engine):
    client = FailingClient(make_rate_limit_error())
    with pytest.raises(anthropic.RateLimitError):
        async for _ in stream_agent_events(
            engine=seeded_engine, user_id="chris", message="hi", client=client
        ):
            pass


async def test_run_agent_loop_propagates_timeout_error(seeded_engine):
    client = FailingClient(make_timeout_error())
    with pytest.raises(anthropic.APITimeoutError):
        await run_agent_loop(engine=seeded_engine, user_id="chris", message="hi", client=client)


# ---------------------------------------------------------------------------
# Claude ending its turn without calling emit_plan is already handled — not
# just on the last iteration, but the moment it happens. These tests prove
# it end-to-end for both loops rather than trusting a reading of the code:
# a fake client that always ends its turn with plain text when asked with
# tool_choice "auto" still has to produce a validated plan.
# ---------------------------------------------------------------------------


async def test_end_turn_without_emit_plan_still_forces_valid_plan(seeded_engine):
    client = EndTurnThenForcedClient(VALID_PLAN_INPUT)

    plan = await run_agent_loop(
        engine=seeded_engine, user_id="chris", message="just chatting, no plan needed", client=client
    )

    assert isinstance(plan, WorkoutPlanResponse)
    assert plan.summary == "Quick test plan"
    # First call went out with tool_choice "auto" (Claude's free choice);
    # only after it ended the turn with no tool call did a forced call
    # with tool_choice "tool" happen — confirms the safety net actually
    # fired rather than the fake client just always returning a plan.
    assert client.calls[0] == ("create", {"type": "auto"})
    assert client.calls[1] == ("create", {"type": "tool", "name": "emit_plan"})


async def test_stream_end_turn_without_emit_plan_still_yields_final(seeded_engine):
    client = EndTurnThenForcedClient(VALID_PLAN_INPUT)

    events = [
        evt
        async for evt in stream_agent_events(
            engine=seeded_engine, user_id="chris", message="just chatting, no plan needed", client=client
        )
    ]

    assert events[-1]["event"] == "final"
    assert events[-1]["data"]["summary"] == "Quick test plan"
    assert client.calls[0] == ("stream", {"type": "auto"})
    assert client.calls[1] == ("create", {"type": "tool", "name": "emit_plan"})


# ---------------------------------------------------------------------------
# Logging the identical set twice in a row is allowed, deliberately — see
# README's design-decisions log. Two identical-looking sets are real,
# common training data (e.g. two straight sets at the same top weight),
# and the tool has no reliable way to distinguish "accidental double
# submission" from "the user actually did that twice." Guessing wrong
# either way is worse than just recording what was reported, so there's
# no deduplication logic to test for correctness — this test instead locks
# in that logging the same set twice produces two rows, not one silently
# merged or rejected one.
# ---------------------------------------------------------------------------


async def test_logging_the_same_set_twice_creates_two_rows(seeded_engine):
    args = LogSetArgs(user_id="chris", exercise="Back Squat", weight=225, reps=5)

    first = await log_set(seeded_engine, args)
    second = await log_set(seeded_engine, args)

    assert first.set_id != second.set_id
    history = await get_recent_history(seeded_engine, GetRecentHistoryArgs(user_id="chris"))
    assert history.count == 2
