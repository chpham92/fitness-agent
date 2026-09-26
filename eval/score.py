"""
Score one saved run file (eval/runs/<name>.json) with the judge.

Each output is judged --repeats times in independent calls and the
per-dimension scores are averaged, since the judge isn't perfectly
self-consistent (see eval/consistency.py). All raw scores and the
evidence text are saved, so a score change can be traced to the reason
the judge gave for it.

Usage:
    python -m eval.score --run iteration_0
    python -m eval.score --run iteration_0_haiku --repeats 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from eval.judge import JUDGE_MODEL, JudgeError, judge_record
from eval.rubric import CONSISTENCY_REPEATS, DIMENSIONS
from eval.rules import violations

RUNS_DIR = Path(__file__).parent / "runs"


async def score_run(run_name: str, repeats: int, concurrency: int) -> Path:
    src = RUNS_DIR / f"{run_name}.json"
    run = json.loads(src.read_text(encoding="utf-8"))
    records = run["results"]
    by_id = {r["prompt_id"]: r for r in records}

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    failed_ids = {r["prompt_id"] for r in records if r.get("failed")}  # no valid plan: not judged, scored 1s below
    jobs = [r["prompt_id"] for r in records if r["prompt_id"] not in failed_ids for _ in range(repeats)]
    outcomes = await asyncio.gather(
        *(judge_record(client, by_id[pid], sem) for pid in jobs), return_exceptions=True
    )

    raw: dict[str, list[dict]] = {r["prompt_id"]: [] for r in records}
    failures = 0
    in_tok = out_tok = 0
    for pid, outcome in zip(jobs, outcomes):
        if isinstance(outcome, JudgeError):
            failures += 1
            continue
        if isinstance(outcome, BaseException):
            raise outcome
        raw[pid].append(outcome.scores.model_dump())
        in_tok += outcome.input_tokens
        out_tok += outcome.output_tokens

    per_prompt = {}
    for pid, judgments in raw.items():
        if not judgments:
            continue
        dims = {d: statistics.mean(j[d]["score"] for j in judgments) for d in DIMENSIONS}
        per_prompt[pid] = {"dimensions": dims, "overall": statistics.mean(dims.values()), "n_judgments": len(judgments)}
    for pid in failed_ids:
        # Policy (decided before seeing later outcomes, and the unflattering
        # choice): a turn that never produced a valid plan gets the minimum
        # score on every dimension. Excluding it would flatter the mean.
        per_prompt[pid] = {"dimensions": {d: 1.0 for d in DIMENSIONS}, "overall": 1.0, "n_judgments": 0, "failed": True}

    split_of = {pid: by_id[pid].get("split", "train") for pid in per_prompt}  # older runs predate the field
    rule_violations = {pid: violations(by_id[pid]) for pid in per_prompt}
    total_violations = sum(len(v) for v in rule_violations.values())
    overall = statistics.mean(p["overall"] for p in per_prompt.values())
    per_dim = {d: statistics.mean(p["dimensions"][d] for p in per_prompt.values()) for d in DIMENSIONS}

    by_split = {}
    for split in sorted(set(split_of.values())):
        pids = [pid for pid in per_prompt if split_of[pid] == split]
        by_split[split] = {
            "prompts": len(pids),
            "mean_overall": statistics.mean(per_prompt[pid]["overall"] for pid in pids),
            "rule_violations": sum(len(rule_violations[pid]) for pid in pids),
        }

    print(f"run: {run_name}   agent model: {run.get('agent_model', 'claude-opus-5 (unrecorded, default)')}   judge: {JUDGE_MODEL}")
    print(f"judge failures: {failures}\n")
    print(f"{'prompt':32s} " + " ".join(f"{d[:10]:>10s}" for d in DIMENSIONS) + f" {'overall':>8s}")
    for pid, p in per_prompt.items():
        print(f"{pid:32s} " + " ".join(f"{p['dimensions'][d]:>10.2f}" for d in DIMENSIONS) + f" {p['overall']:>8.2f}")
    print(f"{'MEAN':32s} " + " ".join(f"{per_dim[d]:>10.2f}" for d in DIMENSIONS) + f" {overall:>8.2f}")
    for split, r in by_split.items():
        print(f"{split:>8s}: mean overall {r['mean_overall']:.2f} over {r['prompts']} prompts, {r['rule_violations']} rule violation(s)")
    print(f"\nrule-based (no LLM): {total_violations} tool-use violation(s) across {len(per_prompt)} prompts; "
          f"{sum(1 for v in rule_violations.values() if v)} prompt(s) with at least one")

    out_path = RUNS_DIR / f"scores_{run_name}.json"
    out_path.write_text(
        json.dumps(
            {
                "run": run_name,
                "agent_model": run.get("agent_model"),
                "judge_model": JUDGE_MODEL,
                "repeats": repeats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "judge_failures": failures,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "mean_overall": overall,
                "mean_by_dimension": per_dim,
                "by_split": by_split,
                "split_of": split_of,
                "total_rule_violations": total_violations,
                "rule_violations": rule_violations,
                "per_prompt": per_prompt,
                "raw_judgments": raw,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\njudge tokens: {in_tok:,} in / {out_tok:,} out\nSaved: {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Score a saved run with the judge.")
    p.add_argument("--run", required=True, help="Run file stem in eval/runs/, e.g. iteration_0 or iteration_0_haiku")
    p.add_argument("--repeats", type=int, default=CONSISTENCY_REPEATS)
    p.add_argument("--concurrency", type=int, default=5)
    args = p.parse_args()
    asyncio.run(score_run(args.run, args.repeats, args.concurrency))


if __name__ == "__main__":
    main()
