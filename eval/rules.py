"""
Deterministic tool-use checks computed from a saved trace — no LLM.

A second, judge-independent instrument. Day 2 used it to validate the
judge's tool_use_correctness scores (violation count vs. judge score:
Pearson r = -0.80 over 20 outputs); it also gives the trend run an
objective metric that can't share the judge's blind spots.

Only tool use has objective ground truth. Safety, personalization, and
communication quality do not, so nothing here validates those.
"""
from __future__ import annotations

import json
from typing import Any

from eval.prompts import EVAL_PROMPTS

_EXPECTED_LOGS = {p.id: p.expected_log_sets for p in EVAL_PROMPTS}


def violations(record: dict[str, Any]) -> list[str]:
    trace, pid = record["trace"], record["prompt_id"]
    found: list[str] = []

    if "get_recent_history" not in [t["name"] for t in trace]:
        found.append("never_checked_history")

    # An exercise counts as verified if a successful lookup or the user's
    # own history returned that exact name.
    known: set[str] = set()
    for t in trace:
        if t["name"] in ("look_up_exercise", "get_recent_history") and not t["is_error"]:
            content = json.loads(t["result_content"])
            known |= {m["name"] for m in content.get("matches", [])}
            known |= {s["exercise"] for s in content.get("sets", [])}
    if record.get("plan") is None:
        found.append("no_valid_plan")
    else:
        unverified = [e["exercise"] for e in record["plan"]["plan"] if e["exercise"] not in known]
        if unverified:
            found.append(f"{len(unverified)}_unverified_plan_exercises")

    logged = sum(1 for t in trace if t["name"] == "log_set" and not t["is_error"])
    expected = _EXPECTED_LOGS[pid]
    if logged != expected:
        found.append(f"log_set_count_{logged}_expected_{expected}")

    if any(t["is_error"] for t in trace):
        found.append("tool_error")
    return found
