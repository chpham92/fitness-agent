"""
The eval harness must run the agent on ONE model, end to end, and must
never silently overwrite a saved run. No API calls: a recording fake
client captures the `model` sent on every request.
"""
from types import SimpleNamespace

import pytest

import app.agent as agent
from eval import runner
from eval.config import AGENT_MODEL, pin_agent_model
from eval.eval_loop import run_eval_turn

VALID_PLAN = {
    "summary": "test plan",
    "plan": [{"exercise": "Back Squat", "sets": 3, "reps": 5, "intensity": "moderate"}],
    "history_checked": False,
    "generated_at": "2026-01-01T00:00:00Z",
}


@pytest.fixture(autouse=True)
def restore_default_model(monkeypatch):
    # pin_agent_model mutates app.agent.DEFAULT_MODEL for the whole process;
    # registering the current value here makes monkeypatch restore it after
    # each test so nothing leaks into the rest of the suite.
    monkeypatch.setattr(agent, "DEFAULT_MODEL", agent.DEFAULT_MODEL)


class RecordingClient:
    """First (auto) call ends the turn with text, so the loop must go
    through _force_finalize; the forced call returns a valid plan."""

    def __init__(self):
        self.models: list[str] = []
        self.messages = self

    async def create(self, **kw):
        self.models.append(kw["model"])
        usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        if kw["tool_choice"].get("type") == "tool":
            block = SimpleNamespace(type="tool_use", id="t1", name="emit_plan", input=VALID_PLAN)
            return SimpleNamespace(stop_reason="tool_use", content=[block], usage=usage)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="hi")], usage=usage)


def test_eval_default_is_haiku_independent_of_the_deployed_default():
    assert AGENT_MODEL == "claude-haiku-4-5"


def test_pin_overrides_the_apps_model_global():
    agent.DEFAULT_MODEL = "some-other-deployed-model"
    assert pin_agent_model() == AGENT_MODEL
    assert agent.DEFAULT_MODEL == AGENT_MODEL


async def test_main_loop_and_forced_finalize_use_the_same_pinned_model():
    agent.DEFAULT_MODEL = "some-other-deployed-model"
    pin_agent_model()
    client = RecordingClient()

    result = await run_eval_turn(engine=None, user_id="u", message="hi", system_prompt="s", client=client)

    assert result.plan.summary == "test plan"
    assert len(client.models) == 2  # one auto call + one forced-finalize call
    assert set(client.models) == {AGENT_MODEL}


async def test_runner_refuses_to_overwrite_a_saved_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    (tmp_path / "iteration_9_x.json").write_text("{}")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        await runner.run_iteration(9, None, "x")
    assert (tmp_path / "iteration_9_x.json").read_text() == "{}"
