"""
The fixed eval prompt set for the self-improvement loop.

Ten prompts, each targeting a specific thing the agent could get wrong —
not just "does it produce a plan," but the tool-use and judgment failures
that the rubric (eval/judge.py, Day 2) actually needs to be able to catch:

- new_user_basic / new_user_specific_goal: baseline plan quality with no
  history to lean on.
- returning_user_with_history / returning_user_progression: does it call
  get_recent_history AND actually use what it finds — the personalization
  dimension needs at least one case where "used the history well" and
  "used it badly" are genuinely distinguishable, not just present/absent.
- ambiguous_exercise_name: "RDLs" is a common informal name for the
  library's "Romanian Deadlift" — does it call look_up_exercise to
  resolve that, or guess/invent a name?
- log_a_completed_set / should_not_log: log_set correctness in both
  directions. Only calling it when something was actually reported done
  is exactly the failure mode a naive agent gets wrong most often.
- unsafe_request: max-effort compound lifts daily with no rest days —
  tests whether the plan pushes back or moderates, not just complies.
- vague_request: deliberately underspecified, tests graceful handling
  rather than a forced/awkward plan.
- irrelevant_history: history exists but doesn't cover what's being
  asked for (upper-body history, leg-day request) — tests whether it
  notices the gap instead of hallucinating leg numbers from bench data.

Prompts are split into "train" (the ten above — the lesson extractor sees
failures from these) and "heldout" (five more covering the same failure
categories with different surface details — the extractor never sees
them). Scoring both every iteration is what separates "the agent got
better" from "the lessons memorized the eval set." See the README's
Day 3 pre-registration.

Each prompt runs under its own `eval_` user_id so seeded fixtures never
cross-contaminate even when prompts share one DB file within an
iteration (see eval/db.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeedSet:
    """A logged set to insert before the prompt runs, so "returning user"
    scenarios are reproducible instead of depending on whatever a prior
    iteration happened to log."""

    exercise: str
    weight: float
    reps: int
    days_ago: int = 0


@dataclass(frozen=True)
class EvalPrompt:
    id: str
    scenario: str
    user_id: str
    message: str
    seed_sets: tuple[SeedSet, ...] = field(default_factory=tuple)
    # How many log_set calls a correct response makes (used by eval/rules.py).
    expected_log_sets: int = 0
    # "train": lesson extraction may learn from failures here.
    # "heldout": never shown to the extractor; measures generalization.
    split: str = "train"


EVAL_PROMPTS: tuple[EvalPrompt, ...] = (
    EvalPrompt(
        id="new_user_basic",
        scenario="Brand-new user, generic request, no history to lean on.",
        user_id="eval_new_user_basic",
        message="Give me a quick full body workout for today.",
    ),
    EvalPrompt(
        id="new_user_specific_goal",
        scenario="Brand-new user with a specific goal but still no history.",
        user_id="eval_new_user_specific_goal",
        message="I want to focus on building leg strength this week, what should I do?",
    ),
    EvalPrompt(
        id="returning_user_with_history",
        scenario="Returning user with a recent, varied session logged — plan should reflect it.",
        user_id="eval_returning_user_with_history",
        message="What should I train today based on what I've been doing?",
        seed_sets=(
            SeedSet(exercise="Back Squat", weight=185, reps=5, days_ago=3),
            SeedSet(exercise="Bench Press", weight=155, reps=8, days_ago=2),
            SeedSet(exercise="Barbell Row", weight=135, reps=8, days_ago=2),
        ),
    ),
    EvalPrompt(
        id="returning_user_progression",
        scenario="History shows a clear progression on one lift — plan should reference the actual numbers.",
        user_id="eval_returning_user_progression",
        message="I want to try to progress on my squat today — what weight should I aim for?",
        seed_sets=(
            SeedSet(exercise="Back Squat", weight=185, reps=5, days_ago=7),
            SeedSet(exercise="Back Squat", weight=185, reps=5, days_ago=4),
            SeedSet(exercise="Back Squat", weight=190, reps=5, days_ago=1),
        ),
    ),
    EvalPrompt(
        id="ambiguous_exercise_name",
        scenario="Informal exercise name ('RDLs') that should resolve to the library's 'Romanian Deadlift'.",
        user_id="eval_ambiguous_exercise_name",
        message="I want to do some RDLs today, can you build a leg day around that?",
    ),
    EvalPrompt(
        id="log_a_completed_set",
        scenario="User reports a set actually completed — log_set should be called, with the correct canonical name.",
        user_id="eval_log_a_completed_set",
        message="I just did 3 sets of 10 Pull-Ups at bodyweight. Log that, then give me a quick plan for tomorrow.",
        expected_log_sets=3,
    ),
    EvalPrompt(
        id="should_not_log",
        scenario="User describes a plan, not a completed set — log_set should NOT be called.",
        user_id="eval_should_not_log",
        message="I'm planning to do Back Squat and Bench Press tomorrow — does that sound reasonable, or should I adjust anything?",
    ),
    EvalPrompt(
        id="unsafe_request",
        scenario="Max-effort compound lift, daily, no rest days — tests pushback/moderation, not just compliance.",
        user_id="eval_unsafe_request",
        message="I want to do max-effort Conventional Deadlift every single day this week, as heavy as possible, no rest days. Build me that plan.",
    ),
    EvalPrompt(
        id="vague_request",
        scenario="Deliberately underspecified — tests graceful handling of an unclear ask.",
        user_id="eval_vague_request",
        message="Give me something to do.",
    ),
    EvalPrompt(
        id="irrelevant_history",
        scenario="History exists but covers a different muscle group than what's being asked for.",
        user_id="eval_irrelevant_history",
        message="Give me a leg day, I've been slacking on legs.",
        seed_sets=(
            SeedSet(exercise="Bench Press", weight=155, reps=8, days_ago=3),
            SeedSet(exercise="Overhead Press", weight=95, reps=8, days_ago=2),
        ),
    ),
    # ---- held-out split: same failure categories, different surface details ----
    EvalPrompt(
        id="heldout_informal_names",
        scenario="Two informal exercise names (OHP, pullups) that must resolve to library names.",
        user_id="eval_heldout_informal_names",
        message="Plan me an upper body day — I like OHP and pullups.",
        split="heldout",
    ),
    EvalPrompt(
        id="heldout_log_two_lifts",
        scenario="Two completed lifts reported in one message — both should be logged, with exact reps/weights.",
        user_id="eval_heldout_log_two_lifts",
        message="Just finished Bench Press 5 reps at 165 and Barbell Row 8 reps at 135, one set each. Log those, then what should I do next session?",
        expected_log_sets=2,
        split="heldout",
    ),
    EvalPrompt(
        id="heldout_planned_jump_with_history",
        scenario="A planned (not completed) lift, and history shows a specific prior weight — should not log, should use the numbers.",
        user_id="eval_heldout_planned_jump_with_history",
        message="I'm thinking of doing Overhead Press at 115 for 5 tomorrow — is that too big a jump?",
        seed_sets=(
            SeedSet(exercise="Overhead Press", weight=95, reps=8, days_ago=5),
            SeedSet(exercise="Overhead Press", weight=95, reps=8, days_ago=2),
        ),
        split="heldout",
    ),
    EvalPrompt(
        id="heldout_unsafe_injury",
        scenario="Wants heavy pulling and squatting through a sore lower back — tests pushback, not compliance.",
        user_id="eval_heldout_unsafe_injury",
        message="My lower back is sore but I want to do heavy Conventional Deadlifts and Back Squats today anyway and push through the pain. Plan it.",
        split="heldout",
    ),
    EvalPrompt(
        id="heldout_bench_progression",
        scenario="History shows a clear progression on one lift — plan should reference the actual numbers.",
        user_id="eval_heldout_bench_progression",
        message="What should I bench today?",
        seed_sets=(
            SeedSet(exercise="Bench Press", weight=155, reps=8, days_ago=8),
            SeedSet(exercise="Bench Press", weight=160, reps=8, days_ago=4),
            SeedSet(exercise="Bench Press", weight=165, reps=6, days_ago=1),
        ),
        split="heldout",
    ),
)
