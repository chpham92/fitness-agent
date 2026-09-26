"""
A turn where the agent never produces a valid plan is DATA, not a crash:
it must be recorded, scored as the worst outcome, and never skipped (which
would flatter the mean). API errors, by contrast, must still propagate.
"""
import json
from types import SimpleNamespace

import pytest

import anthropic
from eval import score
from eval.eval_loop import run_eval_turn
from eval.judge import JudgeResult
from eval.lessons import build_extraction_input, select_failures
from eval.rubric import DIMENSIONS, JudgeScores
from eval.rules import violations
from tests.fakes import FailingClient, make_rate_limit_error

USAGE = SimpleNamespace(input_tokens=1, output_tokens=1)


def _plan(summary="fine"):
    return {"summary": summary, "plan": [{"exercise": "Back Squat", "sets": 3, "reps": 5, "intensity": "moderate"}],
            "history_checked": True, "generated_at": "2026-01-01T00:00:00Z"}


class Scripted:
    """auto call: ends the turn with text. forced calls: return `forced_input`."""

    def __init__(self, forced_input=None, auto_stop="end_turn"):
        self.forced_input, self.auto_stop, self.calls = forced_input, auto_stop, 0
        self.messages = self

    async def create(self, **kw):
        self.calls += 1
        if kw["tool_choice"].get("type") == "tool":
            block = SimpleNamespace(type="tool_use", id=f"t{self.calls}", name="emit_plan", input=self.forced_input)
            return SimpleNamespace(stop_reason="tool_use", content=[block], usage=USAGE)
        return SimpleNamespace(stop_reason=self.auto_stop, content=[SimpleNamespace(type="text", text="hi")], usage=USAGE)


async def test_an_invalid_plan_after_forced_retries_becomes_a_recorded_failure():
    client = Scripted(forced_input=_plan(summary="x" * 501))  # over the 500-char schema limit
    result = await run_eval_turn(engine=None, user_id="u", message="m", system_prompt="s", client=client)

    assert result.plan is None
    assert "could not produce a valid workout plan" in result.failure
    assert "Invalid plan" in result.failure and "summary" in result.failure  # says WHY, from the validation error


async def test_an_unexpected_stop_reason_is_a_recorded_failure_not_a_crash():
    result = await run_eval_turn(engine=None, user_id="u", message="m", system_prompt="s", client=Scripted(auto_stop="max_tokens"))
    assert result.plan is None and "max_tokens" in result.failure


async def test_api_errors_still_propagate():
    with pytest.raises(anthropic.RateLimitError):
        await run_eval_turn(engine=None, user_id="u", message="m", system_prompt="s", client=FailingClient(make_rate_limit_error()))


def _failed_record(pid="heldout_informal_names", split="heldout"):
    return {"prompt_id": pid, "split": split, "scenario": "s", "message": f"message for {pid}", "plan": None, "failed": True,
            "failure": "could not produce a valid workout plan", "trace": []}


def test_rules_flag_a_missing_plan_without_crashing_and_still_check_the_trace():
    v = violations(_failed_record())
    assert "no_valid_plan" in v and "never_checked_history" in v
    assert not any("unverified" in x for x in v)  # nothing to verify


async def test_score_run_counts_a_failed_turn_as_all_ones_and_never_judges_it(tmp_path, monkeypatch):
    ok = {"prompt_id": "new_user_basic", "split": "train", "scenario": "s", "message": "m", "failed": False, "failure": None,
          "trace": [{"name": "get_recent_history", "input": {}, "result_content": '{"sets": []}', "is_error": False}],
          "plan": {"history_checked": True, "summary": "s", "plan": []}}
    (tmp_path / "iteration_0_t.json").write_text(json.dumps({"agent_model": "m", "results": [ok, _failed_record()]}))

    judged = []

    async def fake_judge(client, record, sem=None):
        judged.append(record["prompt_id"])
        scores = JudgeScores.model_validate({d: {"evidence": "e" * 25, "score": 5} for d in DIMENSIONS})
        return JudgeResult(scores, 10, 5, 1)

    monkeypatch.setattr(score, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(score, "judge_record", fake_judge)
    monkeypatch.setattr(score.anthropic, "AsyncAnthropic", lambda: object())

    out = json.loads((await score.score_run("iteration_0_t", repeats=3, concurrency=2)).read_text())

    assert judged == ["new_user_basic"] * 3  # the failed turn was never sent to the judge
    assert out["per_prompt"]["heldout_informal_names"]["overall"] == 1.0
    assert out["by_split"]["heldout"]["mean_overall"] == 1.0  # counted, not excluded
    assert out["by_split"]["train"]["mean_overall"] == 5.0
    assert "no_valid_plan" in out["rule_violations"]["heldout_informal_names"]


def test_failed_turns_are_learnable_failures_even_though_they_have_no_judge_evidence():
    run = {"results": [_failed_record("new_user_basic", split="train")]}
    scores = {"per_prompt": {"new_user_basic": {"overall": 1.0, "dimensions": {d: 1.0 for d in DIMENSIONS}}}, "raw_judgments": {"new_user_basic": []}}
    (case,) = select_failures(run, scores)
    assert all("never produced a valid plan" in ev for _, ev in case.weak_evidence.values())
    assert "NO VALID PLAN" in build_extraction_input([case], existing=[])
