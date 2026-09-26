"""
LLM-as-judge: scores one saved eval record against the rubric.

Same structured-output pattern the agent itself uses — a forced
tool_choice on a Pydantic-validated tool — so the judge physically can't
answer in free text. Because a forced call skips extended thinking, the
schema puts written evidence before each score (see rubric.DimensionScore).

The judge sees things the agent didn't: the ground-truth history the user
actually had (independent of whether the agent looked it up), and the full
tool trace. Without ground truth it couldn't tell "correctly reported no
history" from "never checked."

The agent's operating instructions are taken from the BASE system prompt
(app.agent._build_system_prompt) — deliberately not from whatever
lessons-augmented prompt a later iteration used, so every iteration is
graded against the same fixed spec.

No temperature is set: current Claude models reject sampling parameters,
so judge variance is whatever the model's own sampling produces — which
is exactly what the consistency check (eval/consistency.py) measures.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import anthropic
from pydantic import ValidationError

from app.agent import _build_system_prompt
from eval.prompts import EVAL_PROMPTS
from eval.rubric import (
    JUDGE_TOOL_SCHEMA,
    RUBRIC_TEXT,
    JudgeScores,
    JudgeSubmission,
    submission_to_scores,
)

JUDGE_MODEL = os.environ.get("FITNESS_EVAL_JUDGE_MODEL", "claude-opus-5")
JUDGE_MAX_TOKENS = 4096
MAX_ATTEMPTS = 3
RESULT_TRUNCATE_CHARS = 500

_PROMPTS_BY_ID = {p.id: p for p in EVAL_PROMPTS}


class JudgeError(RuntimeError):
    """The judge couldn't produce a schema-valid score within its attempt budget."""


@dataclass
class JudgeResult:
    scores: JudgeScores
    input_tokens: int
    output_tokens: int
    attempts: int


def _system_prompt(user_id: str) -> str:
    return (
        "You are an evaluation judge for a fitness-coaching agent. You grade "
        "one saved response at a time.\n\n"
        "<agent_operating_instructions>\n"
        f"{_build_system_prompt(user_id)}\n"
        "</agent_operating_instructions>\n\n"
        "<rubric>\n"
        f"{RUBRIC_TEXT}"
        "</rubric>\n\n"
        "Submit your grading by calling the submit_scores tool."
    )


def _ground_truth_history(prompt_id: str) -> str:
    seeds = _PROMPTS_BY_ID[prompt_id].seed_sets
    if not seeds:
        return "The user had NO logged history before this request."
    lines = [
        f"- {s.exercise}: {s.weight:g} lbs x {s.reps} reps, {s.days_ago} day(s) ago"
        for s in seeds
    ]
    return "The user's logged history before this request (whether or not the agent looked it up):\n" + "\n".join(lines)


def _truncate(text: str, limit: int = RESULT_TRUNCATE_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [truncated, {len(text)} chars total]"


def render_trace(record: dict[str, Any]) -> str:
    lines = []
    for i, t in enumerate(record["trace"], start=1):
        flag = "  [TOOL ERROR]" if t["is_error"] else ""
        lines.append(f"{i}. {t['name']}({json.dumps(t['input'])}) -> {_truncate(t['result_content'])}{flag}")
    return "\n".join(lines) or "(the agent made no tool calls)"


def render_plan(record: dict[str, Any]) -> str:
    plan = record["plan"]
    if plan is None:
        return f"(NO VALID PLAN WAS PRODUCED) {record.get('failure') or ''}".strip()
    lines = []
    for ex in plan["plan"]:
        notes = f" — {ex['notes']}" if ex.get("notes") else ""
        lines.append(f"- {ex['exercise']}: {ex['sets']} x {ex['reps']} @ {ex['intensity']}{notes}")
    body = "\n".join(lines) or "(empty plan)"
    return f"history_checked: {plan['history_checked']}\nsummary: {plan['summary']}\nexercises:\n{body}"


def render_record(record: dict[str, Any]) -> str:
    return (
        f"## What this prompt is testing\n{record['scenario']}\n\n"
        f"## Ground truth\n{_ground_truth_history(record['prompt_id'])}\n\n"
        f"## User message\n{record['message']}\n\n"
        f"## Tool trace (in order; this is what actually happened)\n{render_trace(record)}\n\n"
        f"## Final plan (emitted via emit_plan)\n{render_plan(record)}\n"
    )


async def judge_record(
    client: anthropic.AsyncAnthropic,
    record: dict[str, Any],
    semaphore: asyncio.Semaphore | None = None,
) -> JudgeResult:
    system = _system_prompt(record["user_id"])
    user_content = render_record(record)
    input_tokens = 0
    output_tokens = 0
    last_error = "no attempt made"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if semaphore is not None:
            async with semaphore:
                response = await _call(client, system, user_content)
        else:
            response = await _call(client, system, user_content)
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        block = next((b for b in response.content if b.type == "tool_use"), None)
        if block is None:
            last_error = f"no tool_use block (stop_reason={response.stop_reason!r})"
            continue
        try:
            scores = submission_to_scores(JudgeSubmission.model_validate(block.input))
        except ValidationError as e:
            last_error = f"schema validation failed: {e}"
            continue
        return JudgeResult(scores, input_tokens, output_tokens, attempt)

    raise JudgeError(f"judge failed after {MAX_ATTEMPTS} attempts: {last_error}")


async def _call(client: anthropic.AsyncAnthropic, system: str, user_content: str):
    return await client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        system=system,
        tools=[JUDGE_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": JUDGE_TOOL_SCHEMA["name"]},
        messages=[{"role": "user", "content": user_content}],
    )
