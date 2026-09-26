"""
Eval iteration runner: for each prompt in EVAL_PROMPTS, run one turn
against a fresh, identically-seeded DB and save the full result (plan +
tool trace + token usage) to eval/runs/iteration_N.json.

Day 1 scope: no judge, no lessons yet — this just needs to produce clean,
reproducible traces to build the rest of the loop on. --lessons-file
exists now so Day 3's lesson injection slots in without restructuring
this file: it appends the lessons text to the same base system prompt
app.agent._build_system_prompt produces, never touching app/agent.py.

Usage:
    python -m eval.runner --iteration 0
    python -m eval.runner --iteration 1 --lessons-file eval/lessons.md

The agent model is eval/config.py's AGENT_MODEL (Haiku 4.5 by default,
override with FITNESS_EVAL_AGENT_MODEL) — independent of the deployed
app's FITNESS_AGENT_MODEL. Saved runs are never overwritten.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from app import agent as _agent
from app.agent import _build_system_prompt
from eval.config import pin_agent_model
from eval.db import setup_eval_db
from eval.eval_loop import run_eval_turn
from eval.prompts import EVAL_PROMPTS

RUNS_DIR = Path(__file__).parent / "runs"


def _system_prompt_for(user_id: str, lessons_text: str | None) -> str:
    base = _build_system_prompt(user_id)
    if not lessons_text:
        return base
    return base + "\n\nLessons from previous attempts (apply these where relevant):\n" + lessons_text


async def run_iteration(iteration: int, lessons_file: Path | None, tag: str | None = None) -> Path:
    pin_agent_model()
    out_path = RUNS_DIR / (f"iteration_{iteration}_{tag}.json" if tag else f"iteration_{iteration}.json")
    if out_path.exists():
        raise FileExistsError(f"{out_path.name} already exists; refusing to overwrite a saved run. Pick a different --iteration/--tag or delete it deliberately.")
    lessons_text = None
    if lessons_file is not None:
        if not lessons_file.exists():
            raise FileNotFoundError(f"--lessons-file given but not found: {lessons_file}")
        lessons_text = lessons_file.read_text(encoding="utf-8").strip() or None

    client = anthropic.AsyncAnthropic()
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "eval.db"
        engine = await setup_eval_db(db_path)

        for prompt in EVAL_PROMPTS:
            system_prompt = _system_prompt_for(prompt.user_id, lessons_text)
            turn = await run_eval_turn(
                engine=engine,
                user_id=prompt.user_id,
                message=prompt.message,
                system_prompt=system_prompt,
                client=client,
            )
            results.append(
                {
                    "prompt_id": prompt.id,
                    "split": prompt.split,
                    "scenario": prompt.scenario,
                    "user_id": prompt.user_id,
                    "message": prompt.message,
                    "plan": turn.plan.model_dump(mode="json") if turn.plan else None,
                    "failed": turn.plan is None,
                    "failure": turn.failure,
                    "trace": [asdict(r) for r in turn.trace],
                    "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                }
            )
            tool_names = ", ".join(r.name for r in turn.trace) or "(none)"
            outcome = turn.plan.summary[:70] if turn.plan else f"!! NO VALID PLAN: {turn.failure[:90]}"
            print(f"  [{prompt.id}] tools: {tool_names} | {outcome}")

        await engine.dispose()

    RUNS_DIR.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "iteration": iteration,
                "agent_model": _agent.DEFAULT_MODEL,
                "tag": tag,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "lessons_file": lessons_file.name if lessons_file else None,  # name only: absolute paths leak the local username
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one eval iteration against the fitness agent.")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number, used in the output filename.")
    parser.add_argument("--lessons-file", type=Path, default=None, help="Optional lessons file appended to the system prompt.")
    parser.add_argument("--tag", default=None, help="Label appended to the output filename (e.g. a model name), so runs don't overwrite each other.")
    args = parser.parse_args()

    pin_agent_model()
    print(f"Running iteration {args.iteration} on {_agent.DEFAULT_MODEL} ({len(EVAL_PROMPTS)} prompts)...")
    out_path = asyncio.run(run_iteration(args.iteration, args.lessons_file, args.tag))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
