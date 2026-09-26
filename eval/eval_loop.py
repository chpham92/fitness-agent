"""
The eval harness's own tool-calling loop.

Deliberately not a call to app.agent.run_agent_loop — that function's
`trace` param only records tool *names* (fine for its own tests, not
enough for a judge that needs to see what was actually called and
returned), and it always builds its own system prompt internally, with
no way to inject lessons without editing app/agent.py.

Instead this reuses app.agent's actual validation/dispatch logic
(_execute_client_tool, _validate_plan, _force_finalize, TOOLS) — the
part that has to stay correct — and wraps it in a loop that (a) captures
full ToolCallRecords, and (b) takes system_prompt as an explicit
parameter. This is the same "share validation/dispatch, not control
flow" pattern app/agent.py already uses between run_agent_loop and
stream_agent_events (see README's Day 3 design log) — a third sibling
loop for a third transport (eval capture), not a fork of an existing one.
app/agent.py itself is never imported for its side effects and never
modified.

Known gap: token counts here don't include tokens spent inside
_force_finalize's forced-retry calls, since that helper doesn't expose
its own usage. An undercount on the rare turns that hit it — not worth
changing app/agent.py's return type to fix for what this number is used
for (a rough per-iteration cost figure, not a billing record).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anthropic

from app import agent as _agent
from app.agent import (
    DEFAULT_MAX_ITERATIONS,
    MAX_TOKENS,
    TOOLS,
    AgentError,
    _execute_client_tool,
    _force_finalize,
    _tool_result,
    _validate_plan,
)
from app.models import WorkoutPlanResponse


@dataclass
class ToolCallRecord:
    name: str
    input: dict[str, Any]
    result_content: str
    is_error: bool


@dataclass
class EvalTurnResult:
    # plan is None iff the agent never produced a valid plan (see `failure`).
    # That is a legitimate failure of the system under test and is scored as
    # one (1 on every dimension + a rule violation), not an infrastructure
    # error: crashing would discard the whole iteration's other results, and
    # skipping or re-drawing it would quietly flatter the score.
    plan: WorkoutPlanResponse | None
    trace: list[ToolCallRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    failure: str | None = None


def _describe_failure(error: AgentError, messages: list[dict]) -> str:
    # _force_finalize appends its validation errors to `messages` as
    # tool_result blocks, so the last one says WHY the plan was rejected.
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                    return f"{error} Last validation error: {str(block['content'])[:600]}"
    return str(error)


async def _finalize_or_fail(client, system_prompt, messages, trace, input_tokens, output_tokens) -> EvalTurnResult:
    try:
        plan = await _force_finalize(client, system_prompt, messages)
    except AgentError as e:  # typed and bounded: only the agent's own failure to emit a valid plan
        return EvalTurnResult(None, trace, input_tokens, output_tokens, failure=_describe_failure(e, messages))
    return EvalTurnResult(plan, trace, input_tokens, output_tokens)


async def run_eval_turn(
    engine,
    user_id: str,
    message: str,
    system_prompt: str,
    client: anthropic.AsyncAnthropic | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> EvalTurnResult:
    """Run one turn to completion, capturing the full tool-call trace a
    judge needs — name, input, result content, and whether it errored —
    not just which tools fired."""
    client = client or anthropic.AsyncAnthropic()
    messages: list[dict] = [{"role": "user", "content": message}]
    trace: list[ToolCallRecord] = []
    input_tokens = 0
    output_tokens = 0

    for _ in range(max_iterations):
        response = await client.messages.create(
            model=_agent.DEFAULT_MODEL,  # read at call time so eval.config.pin_agent_model applies
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=TOOLS,
            tool_choice={"type": "auto"},
            messages=messages,
        )
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return await _finalize_or_fail(client, system_prompt, messages, trace, input_tokens, output_tokens)

        if response.stop_reason != "tool_use":
            failure = f"Unexpected stop_reason from Claude: {response.stop_reason!r}"
            return EvalTurnResult(None, trace, input_tokens, output_tokens, failure=failure)

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

            content, is_error = await _execute_client_tool(engine, user_id, block.name, block.input)
            trace.append(ToolCallRecord(name=block.name, input=block.input, result_content=content, is_error=is_error))
            tool_results.append(_tool_result(block.id, content, is_error=is_error))

        if finalized is not None:
            return EvalTurnResult(finalized, trace, input_tokens, output_tokens)

        messages.append({"role": "user", "content": tool_results})

    return await _finalize_or_fail(client, system_prompt, messages, trace, input_tokens, output_tokens)
