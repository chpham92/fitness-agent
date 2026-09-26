"""
Lesson extraction: study the agent's low-scoring responses, write
generalizable lessons, and keep them in an append-only store that the
runner injects into the agent's system prompt on the next iteration.

Everything tunable here (thresholds, caps, lint rules) was fixed in the
README's Day 3 pre-registration BEFORE any lesson was extracted.

Guards against learning the test set instead of the skill:
- The extractor only ever sees TRAIN-split failures — never held-out
  prompts, and never the eval's "what this prompt tests" descriptions.
- Every candidate lesson goes through `lint_lesson`: rejected if it names
  a library exercise or reproduces a 5-word run from ANY eval prompt
  (train or held-out). Rejections are recorded, never silently dropped.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.agent import _build_system_prompt
from eval.judge import _ground_truth_history, render_plan, render_trace
from eval.prompts import EVAL_PROMPTS
from eval.rubric import DIMENSIONS
from eval.rules import violations

# ---- pre-registered (README, Day 3) ---------------------------------------
LOW_SCORE_THRESHOLD = 3.5
MAX_NEW_LESSONS_PER_ITERATION = 3
MAX_TOTAL_LESSONS = 12
LINT_NGRAM = 5
# ---------------------------------------------------------------------------

EXTRACTOR_MODEL = os.environ.get("FITNESS_EVAL_EXTRACTOR_MODEL", "claude-opus-5")
EXTRACTOR_MAX_TOKENS = 4096
MAX_ATTEMPTS = 3
WEAK_DIMENSION_BELOW = 4.0

_LIBRARY = json.loads((Path(__file__).parent.parent / "data" / "exercises.json").read_text(encoding="utf-8"))
EXERCISE_NAMES = sorted({e["name"] for e in _LIBRARY})


# A hard ceiling that only bounds prompt growth (12 lessons x 700 chars ~ 2k tokens).
# The model can't count characters (it blew through 300, then 400, even when the
# validation error was fed back), so brevity is requested in WORDS in the prompt
# and this limit is just a backstop.
MAX_LESSON_CHARS = 700


class ExtractionSubmission(BaseModel):
    # Flat scalar fields on purpose. An array-valued `lessons` field failed
    # on a forced tool call (the model returned the array as a string of
    # raw "<parameter ...>" markup) — the same failure the judge's nested
    # objects hit on Day 2. Three slots also caps the count structurally:
    # the model ignored both maxItems and maxLength in the schema, so the
    # only limits that hold are ones the shape itself enforces or code checks.
    failure_analysis: str = Field(
        ...,
        min_length=40,
        max_length=1500,
        description="What recurring failure patterns do these responses show, and why did each happen? Written BEFORE the lessons.",
    )
    lesson_1: str = Field(default="", max_length=MAX_LESSON_CHARS, description="A new, general, imperative rule. Leave empty if nothing generalizes.")
    lesson_2: str = Field(default="", max_length=MAX_LESSON_CHARS, description="A second new rule, or empty.")
    lesson_3: str = Field(default="", max_length=MAX_LESSON_CHARS, description="A third new rule, or empty.")

    @field_validator("lesson_1", "lesson_2", "lesson_3")
    @classmethod
    def _blank_or_substantial(cls, v: str) -> str:
        v = v.strip()
        if v and len(v) < 20:
            raise ValueError("a lesson must be blank or at least 20 characters")
        return v

    @property
    def lessons(self) -> list[str]:
        return [l for l in (self.lesson_1, self.lesson_2, self.lesson_3) if l]


SUBMIT_TOOL = {
    "name": "submit_lessons",
    "description": "Submit the failure analysis, then the new lessons (possibly none).",
    "input_schema": ExtractionSubmission.model_json_schema(),
}


# ---------------------------------------------------------------------------
# Leakage lint
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class Forbidden:
    exercise_names: tuple[str, ...]
    ngrams: frozenset[tuple[str, ...]]


def build_forbidden() -> Forbidden:
    ngrams: set[tuple[str, ...]] = set()
    for p in EVAL_PROMPTS:  # train AND held-out
        toks = _tokens(p.message)
        for i in range(len(toks) - LINT_NGRAM + 1):
            ngrams.add(tuple(toks[i : i + LINT_NGRAM]))
    return Forbidden(tuple(EXERCISE_NAMES), frozenset(ngrams))


def lint_lesson(lesson: str, forbidden: Forbidden) -> str | None:
    """Return a rejection reason, or None if the lesson is clean."""
    low = lesson.lower()
    for name in forbidden.exercise_names:
        if name.lower() in low:
            return f"names the library exercise '{name}'"
    toks = _tokens(lesson)
    for i in range(len(toks) - LINT_NGRAM + 1):
        if tuple(toks[i : i + LINT_NGRAM]) in forbidden.ngrams:
            return f"reproduces a {LINT_NGRAM}-word run from an eval prompt: '{' '.join(toks[i:i + LINT_NGRAM])}'"
    return None


# ---------------------------------------------------------------------------
# Choosing what to learn from
# ---------------------------------------------------------------------------


@dataclass
class FailureCase:
    prompt_id: str
    message: str
    overall: float
    record: dict[str, Any]
    rule_violations: list[str]
    weak_evidence: dict[str, tuple[float, str]] = field(default_factory=dict)  # dim -> (mean, evidence)


def select_failures(run: dict[str, Any], scores: dict[str, Any], threshold: float = LOW_SCORE_THRESHOLD) -> list[FailureCase]:
    """Train-split outputs whose mean overall score is below the threshold."""
    cases = []
    for rec in run["results"]:
        pid = rec["prompt_id"]
        if rec.get("split", "train") != "train" or pid not in scores["per_prompt"]:
            continue
        pp = scores["per_prompt"][pid]
        if pp["overall"] >= threshold:
            continue
        judgments = scores["raw_judgments"].get(pid, [])
        weak = {}
        for dim in DIMENSIONS:
            mean = pp["dimensions"][dim]
            if mean < WEAK_DIMENSION_BELOW:
                if judgments:
                    lowest = min(judgments, key=lambda j: j[dim]["score"])
                    weak[dim] = (mean, lowest[dim]["evidence"])
                else:  # a turn with no valid plan was never judged
                    weak[dim] = (mean, f"The agent never produced a valid plan. {rec.get('failure') or ''}".strip())
        cases.append(FailureCase(pid, rec["message"], pp["overall"], rec, violations(rec), weak))
    return cases


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are improving a fitness-coaching agent by studying its failures. You are
shown responses that scored poorly, and you write short lessons that will be
added to the agent's instructions for future conversations.

<agent_operating_instructions>
{base}
</agent_operating_instructions>

What makes a good lesson:
- A short imperative rule (one or two sentences, under 50 words) the agent can follow in ANY
  conversation, and that someone could check by reading a tool trace.
- GENERAL. It must not mention any specific exercise name, weight, rep count,
  or user message from the examples; write the underlying behavior, not the
  example. Lessons that name an exercise or quote a user message are rejected
  automatically.
- Aimed at the root cause. If the agent already has an instruction it failed
  to follow, you may restate it, but only by making it concrete and
  actionable (what exactly to do, in what order) — not by repeating it louder.
- NOT already covered by the current lessons.
- Prefer patterns that recur across several failures; a single failure only
  deserves a lesson if it is severe (e.g. wrong data written, or an unsafe
  recommendation).

Write your failure_analysis first, then at most {room} lessons. Returning zero
lessons is correct if nothing generalizes."""


def _render_failure(i: int, f: FailureCase) -> str:
    weak = "\n".join(
        f"- {dim} (mean {mean:.1f}): {evidence}" for dim, (mean, evidence) in f.weak_evidence.items()
    ) or "- (no single dimension below threshold)"
    return (
        f"### Response {i}  (overall {f.overall:.2f})\n"
        f"User message: {f.record['message']}\n"
        f"Ground truth: {_ground_truth_history(f.prompt_id)}\n"
        f"Tool trace:\n{render_trace(f.record)}\n"
        f"Final plan:\n{render_plan(f.record)}\n"
        f"Deterministic rule violations: {', '.join(f.rule_violations) or 'none'}\n"
        f"Judge evidence on weak dimensions:\n{weak}\n"
    )


def build_extraction_input(failures: list[FailureCase], existing: list[str]) -> str:
    current = "\n".join(f"{i}. {t}" for i, t in enumerate(existing, 1)) or "(none yet)"
    body = "\n".join(_render_failure(i, f) for i, f in enumerate(failures, 1))
    return f"## Current lessons\n{current}\n\n## Poorly-scoring responses ({len(failures)})\n{body}"


@dataclass
class ExtractionOutcome:
    lessons: list[str] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)
    failure_analysis: str = ""
    attempts: int = 0
    n_failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    note: str = ""


class ExtractionError(RuntimeError):
    pass


async def extract_lessons(
    client: anthropic.AsyncAnthropic,
    failures: list[FailureCase],
    existing: list[str],
    forbidden: Forbidden | None = None,
) -> ExtractionOutcome:
    out = ExtractionOutcome(n_failures=len(failures))
    room = min(MAX_NEW_LESSONS_PER_ITERATION, MAX_TOTAL_LESSONS - len(existing))
    if room <= 0:
        out.note = f"lesson cap ({MAX_TOTAL_LESSONS}) reached; no extraction run"
        return out
    if not failures:
        out.note = f"no train-split outputs below {LOW_SCORE_THRESHOLD}; nothing to learn from"
        return out

    forbidden = forbidden or build_forbidden()
    system = _SYSTEM.format(base=_build_system_prompt("<user_id>"), room=room)
    user = build_extraction_input(failures, existing)
    feedback = ""
    last_error = "no attempt made"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        out.attempts = attempt
        response = await client.messages.create(
            model=EXTRACTOR_MODEL,
            max_tokens=EXTRACTOR_MAX_TOKENS,
            system=system,
            tools=[SUBMIT_TOOL],
            tool_choice={"type": "tool", "name": SUBMIT_TOOL["name"]},
            messages=[{"role": "user", "content": user + feedback}],
        )
        out.input_tokens += response.usage.input_tokens
        out.output_tokens += response.usage.output_tokens
        block = next((b for b in response.content if b.type == "tool_use"), None)
        if block is None:
            last_error = f"no tool_use block (stop_reason={response.stop_reason!r})"
            continue
        try:
            sub = ExtractionSubmission.model_validate(block.input)
        except ValidationError as e:
            last_error = f"schema validation failed: {e}"
            problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:4])
            feedback = f"\n\nNOTE: your previous submission was invalid ({problems}). Resubmit, respecting the field limits."
            continue

        out.failure_analysis = sub.failure_analysis
        rejected: list[tuple[str, str]] = []
        for lesson in sub.lessons:
            reason = lint_lesson(lesson, forbidden)
            if reason is not None:
                rejected.append((lesson, reason))
            elif lesson not in out.lessons and len(out.lessons) < room:
                out.lessons.append(lesson)
        out.dropped.extend({"lesson": l, "reason": r} for l, r in rejected)

        if not rejected or len(out.lessons) >= room:
            return out
        feedback = (
            "\n\nNOTE: some lessons from your previous attempt were rejected: "
            + "; ".join(f"\"{l}\" ({r})" for l, r in rejected)
            + f". Write replacement lessons that are general and avoid those problems "
            f"(you may add at most {room - len(out.lessons)} more)."
        )

    if out.failure_analysis:  # produced valid output at least once; keep what passed lint
        return out
    raise ExtractionError(f"extractor failed after {MAX_ATTEMPTS} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Append-only store
# ---------------------------------------------------------------------------


class LessonStore:
    """Lessons plus a provenance log, in one JSON file. Append-only:
    lessons are never edited or removed once added."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"lessons": [], "log": []}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    @property
    def texts(self) -> list[str]:
        return [l["text"] for l in self.data["lessons"]]

    def add(self, after_iteration: int, outcome: ExtractionOutcome) -> None:
        next_id = len(self.data["lessons"]) + 1
        for offset, text in enumerate(outcome.lessons):
            self.data["lessons"].append({"id": next_id + offset, "text": text, "added_after_iteration": after_iteration})
        self.data["log"].append(
            {
                "after_iteration": after_iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "n_failures_seen": outcome.n_failures,
                "failure_analysis": outcome.failure_analysis,
                "added": outcome.lessons,
                "rejected_by_lint": outcome.dropped,
                "attempts": outcome.attempts,
                "note": outcome.note,
                "extractor_model": EXTRACTOR_MODEL,
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
            }
        )

    def render(self) -> str:
        return "\n".join(f"{i}. {t}" for i, t in enumerate(self.texts, 1))

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
