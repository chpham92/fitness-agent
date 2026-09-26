"""
EXPLORATORY, post-hoc robustness check — not part of the pre-registered
verdict (see the README's Day 4 section for why it was added).

The rubric judge and the lesson extractor share a rubric, and the judge has
a measured length association (r ~ +0.3 on no-lessons outputs), so a rubric
score gain could partly be "learned to write what the rubric rewards" or
"got longer." This asks a different question with no rubric anchors: shown
two responses to the same prompt, blind and in randomized order, which one
better serves the user? The judge is told not to prefer length.

Each prompt's final-iteration output is compared against the same prompt's
output from every no-lessons baseline run. Order is randomized but balanced
per prompt (final is "A" in exactly half its comparisons) so position bias
cancels. Reported: overall win rate, by split, by position, and by whether
the final summary was longer or shorter than the baseline's — the last one
is the direct test of whether length is doing the work.

Usage:
    python -m eval.pairwise --final iteration_4_loop \\
        --baselines iteration_0_loop iteration_0_dry iteration_0_rep1 iteration_0_rep2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError

from app.agent import _build_system_prompt
from eval.judge import _ground_truth_history, render_plan, render_trace

RUNS_DIR = Path(__file__).parent / "runs"
PAIRWISE_MODEL = os.environ.get("FITNESS_EVAL_JUDGE_MODEL", "claude-opus-5")
MAX_ATTEMPTS = 3


class PairwiseSubmission(BaseModel):
    analysis: str = Field(
        ...,
        min_length=40,
        # Backstop only: the model can't count characters (see eval/lessons.py),
        # so brevity is requested in words in the description and system prompt.
        max_length=3000,
        description="In about 100 words, compare the two responses on what each actually did (per the tool trace) and what each told the user. Written BEFORE choosing.",
    )
    winner: Literal["A", "B", "tie"] = Field(..., description="Which response better serves this user.")


PAIRWISE_TOOL = {
    "name": "submit_preference",
    "description": "Submit your comparison, then your choice.",
    "input_schema": PairwiseSubmission.model_json_schema(),
}

_SYSTEM = """\
You compare two responses from a fitness-coaching agent to the same user
request, and pick the one that better serves the user. Judge what each
response actually DID (the tool trace is the record of what happened) and
what it told the user; a summary that claims something the trace doesn't
support is a flaw. Consider correctness of tool use, safety, use of the
user's real history, and clarity and honesty.

Do NOT prefer a response for being longer or more detailed; prefer it only
if the extra content makes it more correct, safer, or more useful. Choose
"tie" only if you genuinely cannot separate them.

<agent_operating_instructions>
{base}
</agent_operating_instructions>

Submit your choice by calling submit_preference. The responses are labelled
A and B in arbitrary order."""


def render_pair(prompt_id: str, message: str, a: dict[str, Any], b: dict[str, Any]) -> str:
    """Deliberately says nothing about which side is the final iteration."""
    def side(label: str, rec: dict[str, Any]) -> str:
        return f"## Response {label}\nTool trace:\n{render_trace(rec)}\n\nPlan submitted:\n{render_plan(rec)}\n"

    return (
        f"## Ground truth\n{_ground_truth_history(prompt_id)}\n\n"
        f"## User message\n{message}\n\n{side('A', a)}\n{side('B', b)}"
    )


@dataclass(frozen=True)
class Pair:
    prompt_id: str
    baseline: str
    final_is_a: bool


def build_pairs(prompt_ids: list[str], baselines: list[str], seed: int = 0) -> list[Pair]:
    """Per prompt, `final` is A in exactly half of its comparisons (when the
    baseline count is even) so position bias cancels; which half is random."""
    pairs = []
    for pid in prompt_ids:
        slots = [i % 2 == 0 for i in range(len(baselines))]
        random.Random(f"{seed}-{pid}").shuffle(slots)
        pairs += [Pair(pid, b, s) for b, s in zip(baselines, slots)]
    return pairs


def final_outcome(winner: str, final_is_a: bool) -> float:
    """1.0 if the final-iteration response won, 0.0 if the baseline won, 0.5 tie."""
    if winner == "tie":
        return 0.5
    return 1.0 if (winner == "A") == final_is_a else 0.0


def summarize(rows: list[dict[str, Any]], seed: int = 0, resamples: int = 2000) -> dict[str, Any]:
    """rows: {prompt_id, split, outcome, final_is_a, final_longer}."""
    def rate(sub):
        return statistics.mean(r["outcome"] for r in sub) if sub else None

    by_prompt: dict[str, list[float]] = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], []).append(r["outcome"])
    prompt_rates = [statistics.mean(v) for v in by_prompt.values()]
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(prompt_rates, k=len(prompt_rates))) for _ in range(resamples))
    return {
        "n_comparisons": len(rows),
        "final_win_rate": rate(rows),
        "final_wins": sum(1 for r in rows if r["outcome"] == 1.0),
        "baseline_wins": sum(1 for r in rows if r["outcome"] == 0.0),
        "ties": sum(1 for r in rows if r["outcome"] == 0.5),
        # resampling PROMPTS (not comparisons): the 4 baselines of one prompt aren't independent
        "win_rate_ci95_over_prompts": [boots[int(0.025 * resamples)], boots[int(0.975 * resamples) - 1]],
        "by_split": {s: rate([r for r in rows if r["split"] == s]) for s in sorted({r["split"] for r in rows})},
        "by_position": {"final_is_A": rate([r for r in rows if r["final_is_a"]]), "final_is_B": rate([r for r in rows if not r["final_is_a"]])},
        "by_length": {"final_summary_longer": rate([r for r in rows if r["final_longer"]]), "final_summary_shorter_or_equal": rate([r for r in rows if not r["final_longer"]]),
                      "n_longer": sum(1 for r in rows if r["final_longer"]), "n_shorter_or_equal": sum(1 for r in rows if not r["final_longer"])},
    }


async def _judge_pair(client, system: str, content: str, sem: asyncio.Semaphore) -> tuple[PairwiseSubmission, int, int]:
    in_tok = out_tok = 0
    last = "no attempt"
    feedback = ""
    for _ in range(MAX_ATTEMPTS):
        async with sem:
            r = await client.messages.create(
                model=PAIRWISE_MODEL, max_tokens=2048, system=system, tools=[PAIRWISE_TOOL],
                tool_choice={"type": "tool", "name": PAIRWISE_TOOL["name"]}, messages=[{"role": "user", "content": content + feedback}],
            )
        in_tok += r.usage.input_tokens
        out_tok += r.usage.output_tokens
        block = next((b for b in r.content if b.type == "tool_use"), None)
        if block is None:
            last = "no tool_use block"
            continue
        try:
            return PairwiseSubmission.model_validate(block.input), in_tok, out_tok
        except ValidationError as e:
            last = str(e)
            feedback = "\n\nNOTE: your previous submission was invalid (" + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:3]) + "). Resubmit, keeping the analysis to about 100 words."
    raise RuntimeError(f"pairwise judge failed after {MAX_ATTEMPTS} attempts: {last}")


async def run_pairwise(final_name: str, baseline_names: list[str], concurrency: int = 5, seed: int = 0) -> Path:
    load = lambda n: {r["prompt_id"]: r for r in json.loads((RUNS_DIR / f"{n}.json").read_text(encoding="utf-8"))["results"]}
    final, baselines = load(final_name), {n: load(n) for n in baseline_names}
    pairs = build_pairs(sorted(final), baseline_names, seed)

    client, sem = anthropic.AsyncAnthropic(), asyncio.Semaphore(concurrency)
    jobs = []
    for p in pairs:
        f, b = final[p.prompt_id], baselines[p.baseline][p.prompt_id]
        a_rec, b_rec = (f, b) if p.final_is_a else (b, f)
        system = _SYSTEM.format(base=_build_system_prompt("<user_id>"))
        jobs.append(_judge_pair(client, system, render_pair(p.prompt_id, f["message"], a_rec, b_rec), sem))
    outcomes = await asyncio.gather(*jobs, return_exceptions=True)  # one bad call must not discard the other 59
    failed = [p for p, o in zip(pairs, outcomes) if isinstance(o, BaseException)]
    for o in outcomes:
        if isinstance(o, BaseException) and not isinstance(o, RuntimeError):
            raise o  # API/auth/credit errors are real problems, not per-comparison judge failures
    kept = [(p, o) for p, o in zip(pairs, outcomes) if not isinstance(o, BaseException)]
    results = [o for _, o in kept]

    rows, raw = [], []
    for p, (sub, i, o) in kept:
        f, b = final[p.prompt_id], baselines[p.baseline][p.prompt_id]
        flen = len(f["plan"]["summary"]) if f["plan"] else 0
        blen = len(b["plan"]["summary"]) if b["plan"] else 0
        outcome = final_outcome(sub.winner, p.final_is_a)
        rows.append({"prompt_id": p.prompt_id, "split": f.get("split", "train"), "outcome": outcome, "final_is_a": p.final_is_a, "final_longer": flen > blen})
        raw.append({**rows[-1], "baseline": p.baseline, "winner_label": sub.winner, "analysis": sub.analysis})
    summary = summarize(rows, seed)

    out = RUNS_DIR / f"pairwise_{final_name}.json"
    out.write_text(
        json.dumps({"exploratory": True, "final": final_name, "baselines": baseline_names, "model": PAIRWISE_MODEL, "seed": seed, "comparisons_failed": [{"prompt_id": p.prompt_id, "baseline": p.baseline} for p in failed],
                    "input_tokens": sum(r[1] for r in results), "output_tokens": sum(r[2] for r in results), "summary": summary, "comparisons": raw},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def format_summary(s: dict[str, Any]) -> str:
    lo, hi = s["win_rate_ci95_over_prompts"]
    l = s["by_length"]
    return "\n".join([
        f"comparisons: {s['n_comparisons']}   final wins {s['final_wins']}   baseline wins {s['baseline_wins']}   ties {s['ties']}",
        f"final-iteration win rate: {s['final_win_rate']:.1%}   (95% CI over prompts: {lo:.1%} - {hi:.1%})",
        "by split:    " + "   ".join(f"{k} {v:.1%}" for k, v in s["by_split"].items()),
        f"by position: final as A {s['by_position']['final_is_A']:.1%}   final as B {s['by_position']['final_is_B']:.1%}   (a big gap here = position bias)",
        f"by length:   final summary LONGER {l['final_summary_longer']:.1%} (n={l['n_longer']})   "
        + (f"shorter/equal {l['final_summary_shorter_or_equal']:.1%} (n={l['n_shorter_or_equal']})" if l["n_shorter_or_equal"] else "shorter/equal: no cases"),
    ])


def main() -> None:
    p = argparse.ArgumentParser(description="Exploratory blind pairwise comparison: final iteration vs no-lessons baselines.")
    p.add_argument("--final", required=True)
    p.add_argument("--baselines", nargs="+", required=True)
    p.add_argument("--concurrency", type=int, default=5)
    args = p.parse_args()
    out = asyncio.run(run_pairwise(args.final, args.baselines, args.concurrency))
    print(format_summary(json.loads(out.read_text(encoding="utf-8"))["summary"]))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
