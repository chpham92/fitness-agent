"""
The self-improvement loop, end to end:

    for each iteration n:
        run the agent on all prompts (with lessons learned so far)
        score the run with the judge (+ rule checks)
        extract new lessons from n's train-split failures  -> snapshot for n+1

Resumable: every step's output is a file, and a step whose output exists
is skipped, so a crash (or a deliberate stop after N iterations) never
re-spends API calls or double-adds lessons. The lessons active during
iteration n are frozen in lessons_iteration_n_<tag>.md, so every run has
an audit trail of exactly what the agent was told.

Usage:
    python -m eval.loop --tag dry --iterations 2
    python -m eval.loop --tag loop --iterations 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import anthropic

from eval.lessons import LessonStore, build_forbidden, extract_lessons, select_failures
from eval.rubric import CONSISTENCY_REPEATS
from eval.runner import RUNS_DIR, run_iteration
from eval.score import score_run


def _runs_dir() -> Path:
    return RUNS_DIR


async def run_loop(tag: str, iterations: int, repeats: int = CONSISTENCY_REPEATS, concurrency: int = 5) -> None:
    runs = _runs_dir()
    client = anthropic.AsyncAnthropic()
    store = LessonStore(runs / f"lessons_{tag}.json")
    forbidden = build_forbidden()

    for n in range(iterations):
        run_name = f"iteration_{n}_{tag}"
        run_path = runs / f"{run_name}.json"
        scores_path = runs / f"scores_{run_name}.json"
        snapshot = runs / f"lessons_iteration_{n}_{tag}.md"

        if run_path.exists():
            print(f"[{run_name}] agent run exists, skipping")
        else:
            if n > 0 and not snapshot.exists():
                raise RuntimeError(f"missing lessons snapshot {snapshot.name}; cannot run iteration {n}")
            print(f"[{run_name}] running agent" + (f" with lessons from {snapshot.name}" if n > 0 else " (no lessons)"))
            await run_iteration(n, snapshot if n > 0 else None, tag)

        if scores_path.exists():
            print(f"[{run_name}] scores exist, skipping")
        else:
            print(f"[{run_name}] scoring ({repeats} judge repeats)")
            await score_run(run_name, repeats, concurrency)

        if n == iterations - 1:
            break

        next_snapshot = runs / f"lessons_iteration_{n + 1}_{tag}.md"
        if next_snapshot.exists():
            print(f"[{run_name}] lessons for iteration {n + 1} exist, skipping extraction")
            continue

        if any(entry["after_iteration"] == n for entry in store.data["log"]):
            print(f"[{run_name}] lessons already extracted (crash between save and snapshot); writing snapshot")
        else:
            run = json.loads(run_path.read_text(encoding="utf-8"))
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
            failures = select_failures(run, scores)
            print(f"[{run_name}] extracting lessons from {len(failures)} train failure(s)")
            outcome = await extract_lessons(client, failures, store.texts, forbidden)
            store.add(n, outcome)
            store.save()
            print(f"[{run_name}] +{len(outcome.lessons)} lesson(s), {len(outcome.dropped)} rejected by lint")
        next_snapshot.write_text(store.render(), encoding="utf-8")

    print_trend(tag, iterations)


def print_trend(tag: str, iterations: int) -> None:
    runs = _runs_dir()
    print(f"\ntrend for tag '{tag}'  (mean overall score / rule violations)")
    print(f"{'iter':>4s} {'train':>14s} {'heldout':>14s}  lessons active")
    for n in range(iterations):
        p = runs / f"scores_iteration_{n}_{tag}.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))["by_split"]
        cell = lambda k: f"{s[k]['mean_overall']:.2f} / {s[k]['rule_violations']}" if k in s else "-"
        snap = runs / f"lessons_iteration_{n}_{tag}.md"
        n_lessons = len([l for l in snap.read_text(encoding='utf-8').splitlines() if l.strip()]) if snap.exists() else 0
        print(f"{n:>4d} {cell('train'):>14s} {cell('heldout'):>14s}  {n_lessons}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run the self-improvement loop.")
    p.add_argument("--tag", required=True, help="Run label; all files are named ..._<tag>.")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--repeats", type=int, default=CONSISTENCY_REPEATS)
    p.add_argument("--concurrency", type=int, default=5)
    args = p.parse_args()
    asyncio.run(run_loop(args.tag, args.iterations, args.repeats, args.concurrency))


if __name__ == "__main__":
    main()
