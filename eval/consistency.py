"""
Judge self-consistency check: score the same saved outputs several times
in independent judge calls and measure how much the judge disagrees with
itself. This is the project's kill-condition test, run on the baseline
BEFORE any lesson-loop work is built on top of the judge.

`analyze` is a pure function (numbers in, verdict out) so the thresholds'
logic can be unit tested without any API calls. The thresholds themselves
are pre-registered in eval/rubric.py.

Usage:
    python -m eval.consistency --run iteration_0_haiku
    python -m eval.consistency --run iteration_0_haiku --repeats 3 --concurrency 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from eval.judge import JUDGE_MODEL, JudgeError, JudgeResult, judge_record
from eval.rubric import (
    CONSISTENCY_REPEATS,
    DIMENSIONS,
    HEADROOM_CEILING_MEAN,
    MAX_LARGE_DISAGREEMENT_RATE,
    MIN_EXACT_AGREEMENT_RATE,
)

RUNS_DIR = Path(__file__).parent / "runs"


def analyze(scores_by_prompt: dict[str, list[dict[str, int]]]) -> dict[str, Any]:
    """scores_by_prompt: prompt_id -> one {dimension: score} dict per repeat."""
    cells: list[tuple[str, str, list[int]]] = []
    for prompt_id, repeats in scores_by_prompt.items():
        for dim in DIMENSIONS:
            cells.append((prompt_id, dim, [r[dim] for r in repeats]))

    def rates(subset: list[tuple[str, str, list[int]]]) -> dict[str, float]:
        n = len(subset)
        large = sum(1 for _, _, s in subset if max(s) - min(s) >= 2)
        exact = sum(1 for _, _, s in subset if max(s) == min(s))
        return {
            "cells": n,
            "large_disagreement_rate": large / n,
            "exact_agreement_rate": exact / n,
            "mean_score": statistics.mean(x for _, _, s in subset for x in s),
            "within_cell_std": statistics.mean(statistics.pstdev(s) for _, _, s in subset),
            "between_cell_std": statistics.pstdev([statistics.mean(s) for _, _, s in subset])
            if n > 1
            else 0.0,
        }

    overall = rates(cells)
    per_dimension = {d: rates([c for c in cells if c[1] == d]) for d in DIMENSIONS}

    all_scores = [x for _, _, s in cells for x in s]
    distribution = {str(v): all_scores.count(v) for v in range(1, 6)}

    reliability_pass = (
        overall["large_disagreement_rate"] <= MAX_LARGE_DISAGREEMENT_RATE
        and overall["exact_agreement_rate"] >= MIN_EXACT_AGREEMENT_RATE
    )
    headroom_flag = overall["mean_score"] > HEADROOM_CEILING_MEAN

    return {
        "overall": overall,
        "per_dimension": per_dimension,
        "score_distribution": distribution,
        "reliability_pass": reliability_pass,
        "headroom_flag": headroom_flag,
        "thresholds": {
            "max_large_disagreement_rate": MAX_LARGE_DISAGREEMENT_RATE,
            "min_exact_agreement_rate": MIN_EXACT_AGREEMENT_RATE,
            "headroom_ceiling_mean": HEADROOM_CEILING_MEAN,
        },
    }


def format_report(a: dict[str, Any], failures: int) -> str:
    o = a["overall"]
    lines = [
        f"cells: {o['cells']}   judge failures (schema-invalid after retries): {failures}",
        "",
        f"{'':26s} {'large-disagree':>15s} {'exact-agree':>12s} {'mean':>6s} {'within-sd':>10s} {'between-sd':>11s}",
    ]
    for name, r in [("OVERALL", o), *a["per_dimension"].items()]:
        lines.append(
            f"{name:26s} {r['large_disagreement_rate']:>14.0%} {r['exact_agreement_rate']:>12.0%} "
            f"{r['mean_score']:>6.2f} {r['within_cell_std']:>10.2f} {r['between_cell_std']:>11.2f}"
        )
    t = a["thresholds"]
    lines += [
        "",
        f"score distribution (all judge scores): {a['score_distribution']}",
        "",
        f"RELIABILITY: {'PASS' if a['reliability_pass'] else 'FAIL'} "
        f"(need large-disagree <= {t['max_large_disagreement_rate']:.0%} and exact-agree >= {t['min_exact_agreement_rate']:.0%})",
        f"HEADROOM:    {'FLAG - baseline mean above ' + str(t['headroom_ceiling_mean']) + ', little room to improve' if a['headroom_flag'] else 'ok'}"
        f" (baseline mean {o['mean_score']:.2f})",
    ]
    return "\n".join(lines)


async def run_consistency(run_name: str, repeats: int, concurrency: int) -> Path:
    src = RUNS_DIR / f"{run_name}.json"
    records = json.loads(src.read_text(encoding="utf-8"))["results"]

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)

    jobs = [(r["prompt_id"], k) for r in records for k in range(repeats)]
    by_id = {r["prompt_id"]: r for r in records}
    outcomes = await asyncio.gather(
        *(judge_record(client, by_id[pid], sem) for pid, _ in jobs), return_exceptions=True
    )

    scores_by_prompt: dict[str, list[dict[str, int]]] = {r["prompt_id"]: [] for r in records}
    raw: dict[str, list[dict[str, Any]]] = {r["prompt_id"]: [] for r in records}
    failures = 0
    in_tok = out_tok = 0
    for (pid, _), outcome in zip(jobs, outcomes):
        if isinstance(outcome, JudgeError):
            failures += 1
            continue
        if isinstance(outcome, BaseException):
            raise outcome
        result: JudgeResult = outcome
        scores_by_prompt[pid].append(result.scores.scores())
        raw[pid].append({**result.scores.model_dump(), "attempts": result.attempts})
        in_tok += result.input_tokens
        out_tok += result.output_tokens

    complete = {pid: s for pid, s in scores_by_prompt.items() if len(s) == repeats}
    analysis = analyze(complete)
    print(format_report(analysis, failures))

    out_path = RUNS_DIR / f"judge_consistency_{run_name}.json"
    out_path.write_text(
        json.dumps(
            {
                "run": run_name,
                "judge_model": JUDGE_MODEL,
                "repeats": repeats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "judge_failures": failures,
                "prompts_excluded_for_missing_repeats": sorted(set(scores_by_prompt) - set(complete)),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "analysis": analysis,
                "raw_scores": raw,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\njudge tokens: {in_tok:,} in / {out_tok:,} out\nSaved: {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Measure the judge's self-consistency on saved outputs.")
    p.add_argument("--run", required=True, help="Run file stem in eval/runs/, e.g. iteration_0_haiku")
    p.add_argument("--repeats", type=int, default=CONSISTENCY_REPEATS)
    p.add_argument("--concurrency", type=int, default=5)
    args = p.parse_args()
    asyncio.run(run_consistency(args.run, args.repeats, args.concurrency))


if __name__ == "__main__":
    main()
