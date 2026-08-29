"""
The raw Claude tool-use loop.

No framework — see the README's design-decisions log for why. The shape
is: send messages + 4 tool schemas (3 real tools + the terminal
`emit_plan`), execute whatever Claude asks for, feed results back, repeat.

`emit_plan` is treated exactly like the other three tools: its input is
Pydantic-validated, and a validation failure becomes an `is_error`
tool_result that Claude gets a chance to correct — it is not special-cased
into its own error path. It's only special in that a *successful*
validation ends the loop instead of continuing it.

Two safety nets keep every call resolving to a validated plan without an
unbounded loop:
- If Claude ends its turn (`stop_reason == "end_turn"`) without having
  called `emit_plan`, we force it via `tool_choice`.
- If the loop exhausts `max_iterations` tool-calling rounds, we do the same.
A forced call gets one retry on validation failure before we give up and
raise `AgentError`.
"""
from __future__ import annotations

import os
from typing import AsyncIterator

import anthropic
from pydantic import ValidationError

from app.db import utcnow
from app.models import (
    GetRecentHistoryArgs,
    LogSetArgs,
    LookUpExerciseArgs,
    WorkoutPlanResponse,
)
from app.tools.exercise_lookup import (
    ANTHROPIC_TOOL_SCHEMA as LOOKUP_SCHEMA,
    look_up_exercise,
)
from app.tools.history import (
    ANTHROPIC_TOOL_SCHEMA as HISTORY_SCHEMA,
    get_recent_history,
)
from app.tools.log_set import (
    ANTHROPIC_TOOL_SCHEMA as LOG_SET_SCHEMA,
    UnknownExerciseError,
    log_set,
)

DEFAULT_MODEL = os.environ.get("FITNESS_AGENT_MODEL", "claude-opus-5")
MAX_TOKENS = 8192
DEFAULT_MAX_ITERATIONS = 8
MAX_FORCE_RETRIES = 1

EMIT_PLAN_SCHEMA = {
    "name": "emit_plan",
    "description": (
        "Deliver the final workout plan for this turn. Call this exactly "
        "once, as your last action, once you have everything you need to "
        "answer the user. This is the only way to deliver your final "
        "answer — do not describe the plan in free text instead."
    ),
    # Generated straight from the response model so the tool schema and
    # the validator can never drift apart.
    "input_schema": WorkoutPlanResponse.model_json_schema(),
}

TOOLS = [LOOKUP_SCHEMA, LOG_SET_SCHEMA, HISTORY_SCHEMA, EMIT_PLAN_SCHEMA]


class AgentError(RuntimeError):
    """Raised when the loop can't reach a validated plan within its retry budget."""


def _build_system_prompt(user_id: str) -> str:
    return (
        "You are a fitness coaching assistant with tools for looking up "
        "exercises in the exercise library, logging completed sets, and "
        "retrieving a user's recent workout history.\n\n"
        f'The current user\'s id is "{user_id}". Always use this exact '
        "value when calling log_set or get_recent_history — never ask the "
        "user for it or invent one.\n\n"
        "Guidelines:\n"
        "- Before recommending or logging an exercise you're not certain "
        "exists in the library, call look_up_exercise to confirm the exact "
        "name.\n"
        "- Before writing a workout plan, call get_recent_history so the "
        "plan accounts for what the user has actually been doing.\n"
        "- Only call log_set when the user reports having actually "
        "completed a set, not when they're just planning one.\n"
        "- When you're ready to answer, call emit_plan exactly once with "
        "the complete structured plan. It is the only way to deliver your "
        "final answer — do not write the plan as free text."
    )


def _tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict:
    result = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        result["is_error"] = True
    return result


async def _execute_client_tool(
    engine, user_id: str, name: str, raw_input: dict
) -> tuple[str, bool]:
    """Validate args against the tool's Pydantic model, then run it.

    user_id is injected server-side (not trusted from the model's tool
    call) so nothing Claude produces can log or read another user's data —
    the same validation-boundary principle as Day 1, applied to identity.
    Returns (content_for_tool_result, is_error).
    """
    try:
        if name == "look_up_exercise":
            args = LookUpExerciseArgs(**raw_input)
            result = await look_up_exercise(engine, args)
        elif name == "log_set":
            args = LogSetArgs(**{**raw_input, "user_id": user_id})
            result = await log_set(engine, args)
        elif name == "get_recent_history":
            args = GetRecentHistoryArgs(**{**raw_input, "user_id": user_id})
            result = await get_recent_history(engine, args)
        else:
            return f"Unknown tool: {name}", True
    except ValidationError as e:
        return f"Invalid arguments for {name}: {e}", True
    except UnknownExerciseError as e:
        return str(e), True

    return result.model_dump_json(), False


def _validate_plan(raw_input: dict) -> tuple[WorkoutPlanResponse | None, str | None]:
    try:
        plan = WorkoutPlanResponse.model_validate(raw_input)
    except ValidationError as e:
        return None, f"Invalid plan: {e}"
    # Stamp server time rather than trust the model's notion of "now".
    plan.generated_at = utcnow()
    return plan, None


async def _force_finalize(
    client: anthropic.AsyncAnthropic, system_prompt: str, messages: list[dict]
) -> WorkoutPlanResponse:
    """Force emit_plan via tool_choice. Used when Claude ends the turn (or
    the iteration budget runs out) without having called it naturally."""
    for _ in range(MAX_FORCE_RETRIES + 1):
        response = await client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=TOOLS,
            tool_choice={"type": "tool", "name": "emit_plan"},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        emit_block = next(
            (b for b in response.content if b.type == "tool_use" and b.name == "emit_plan"),
            None,
        )
        if emit_block is None:
            raise AgentError(
                f"Forced emit_plan call produced no tool_use block "
                f"(stop_reason={response.stop_reason!r})."
            )

        plan, error = _validate_plan(emit_block.input)
        if plan is not None:
            return plan

        messages.append({"role": "user", "content": [_tool_result(emit_block.id, error, is_error=True)]})

    raise AgentError("Claude could not produce a valid workout plan after forced retries.")


async def run_agent_loop(
    engine,
    user_id: str,
    message: str,
    client: anthropic.AsyncAnthropic | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    trace: list[str] | None = None,
) -> WorkoutPlanResponse:
    """Run one /chat turn to completion and return the validated plan.

    `trace`, if given, collects the name of every non-emit_plan tool call
    Claude makes, in order — mainly so tests can assert on tool-calling
    behavior without re-parsing message history.
    """
    client = client or anthropic.AsyncAnthropic()
    system_prompt = _build_system_prompt(user_id)
    messages: list[dict] = [{"role": "user", "content": message}]

    for _ in range(max_iterations):
        response = await client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=TOOLS,
            tool_choice={"type": "auto"},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return await _force_finalize(client, system_prompt, messages)

        if response.stop_reason != "tool_use":
            raise AgentError(f"Unexpected stop_reason from Claude: {response.stop_reason!r}")

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = []
        finalized: WorkoutPlanResponse | None = None

        for block in tool_use_blocks:
            if block.name == "emit_plan":
                plan, error = _validate_plan(block.input)
                if plan is not None:
                    finalized = plan
                    break
                tool_results.append(_tool_result(block.id, error, is_error=True))
                continue

            if trace is not None:
                trace.append(block.name)
            content, is_error = await _execute_client_tool(engine, user_id, block.name, block.input)
            tool_results.append(_tool_result(block.id, content, is_error=is_error))

        if finalized is not None:
            return finalized

        messages.append({"role": "user", "content": tool_results})

    return await _force_finalize(client, system_prompt, messages)


async def stream_agent_events(
    engine,
    user_id: str,
    message: str,
    client: anthropic.AsyncAnthropic | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AsyncIterator[dict]:
    """Stream one /chat/stream turn as a sequence of typed events.

    Yields `{"event": <type>, "data": {...}}` dicts — see the README's SSE
    event schema table for the full contract. This is the streaming sibling
    of `run_agent_loop`: same tool-dispatch (`_execute_client_tool`), same
    validation (`_validate_plan`), same forced-finalize safety net
    (`_force_finalize`) for `end_turn` / exhausted `max_iterations`. The
    only real difference is the transport — a live token stream instead of
    one blocking call — and that progress is reported via yielded events
    instead of a single return value, so the tool-calling round itself
    (assembling blocks, branching on emit_plan) is necessarily duplicated
    in miniature between the two loops rather than shared line-for-line.
    """
    client = client or anthropic.AsyncAnthropic()
    system_prompt = _build_system_prompt(user_id)
    messages: list[dict] = [{"role": "user", "content": message}]

    for _ in range(max_iterations):
        async with client.messages.stream(
            model=DEFAULT_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=TOOLS,
            tool_choice={"type": "auto"},
            messages=messages,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield {"event": "text", "data": {"delta": event.delta.text}}
            response = await stream.get_final_message()

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            plan = await _force_finalize(client, system_prompt, messages)
            yield {"event": "final", "data": plan.model_dump(mode="json")}
            return

        if response.stop_reason != "tool_use":
            raise AgentError(f"Unexpected stop_reason from Claude: {response.stop_reason!r}")

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = []
        finalized: WorkoutPlanResponse | None = None

        for block in tool_use_blocks:
            yield {"event": "tool_call", "data": {"id": block.id, "name": block.name, "input": block.input}}

            if block.name == "emit_plan":
                plan, error = _validate_plan(block.input)
                if plan is not None:
                    # No redundant success tool_result — "final" already
                    # carries the validated output.
                    finalized = plan
                    break
                yield {"event": "tool_result", "data": {"id": block.id, "name": block.name, "content": error, "is_error": True}}
                tool_results.append(_tool_result(block.id, error, is_error=True))
                continue

            content, is_error = await _execute_client_tool(engine, user_id, block.name, block.input)
            yield {
                "event": "tool_result",
                "data": {"id": block.id, "name": block.name, "content": content, "is_error": is_error},
            }
            tool_results.append(_tool_result(block.id, content, is_error=is_error))

        if finalized is not None:
            yield {"event": "final", "data": finalized.model_dump(mode="json")}
            return

        messages.append({"role": "user", "content": tool_results})

    plan = await _force_finalize(client, system_prompt, messages)
    yield {"event": "final", "data": plan.model_dump(mode="json")}
