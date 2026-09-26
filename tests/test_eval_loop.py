"""
Loop orchestration with the expensive steps faked: what matters here is
wiring — which lessons reach which iteration, and that resuming never
re-spends or double-adds.
"""
import json

import pytest

from eval import loop
from eval.lessons import ExtractionOutcome, LessonStore
from eval.rubric import DIMENSIONS


class Harness:
    def __init__(self, tmp_path, monkeypatch):
        self.dir = tmp_path
        self.agent_runs: list[tuple[int, str | None]] = []
        self.scored: list[str] = []
        self.extractions = 0
        monkeypatch.setattr(loop, "RUNS_DIR", tmp_path)
        monkeypatch.setattr(loop.anthropic, "AsyncAnthropic", lambda: object())
        monkeypatch.setattr(loop, "run_iteration", self.fake_run)
        monkeypatch.setattr(loop, "score_run", self.fake_score)
        monkeypatch.setattr(loop, "extract_lessons", self.fake_extract)

    async def fake_run(self, n, lessons_file, tag):
        text = lessons_file.read_text() if lessons_file else None
        self.agent_runs.append((n, text))
        rec = {"prompt_id": "new_user_basic", "split": "train", "scenario": "s", "message": "m",
               "trace": [], "plan": {"history_checked": True, "summary": "s", "plan": []}}
        path = self.dir / f"iteration_{n}_{tag}.json"
        path.write_text(json.dumps({"results": [rec]}))
        return path

    async def fake_score(self, run_name, repeats, concurrency):
        self.scored.append(run_name)
        dims = {d: 2.0 for d in DIMENSIONS}
        raw = [{d: {"score": 2, "evidence": "e" * 25} for d in DIMENSIONS}]
        scores = {"per_prompt": {"new_user_basic": {"overall": 2.0, "dimensions": dims}},
                  "raw_judgments": {"new_user_basic": raw},
                  "by_split": {"train": {"mean_overall": 2.0, "rule_violations": 1, "prompts": 1}}}
        (self.dir / f"scores_{run_name}.json").write_text(json.dumps(scores))

    async def fake_extract(self, client, failures, existing, forbidden):
        self.extractions += 1
        assert failures, "loop should pass the train failures through"
        return ExtractionOutcome(lessons=[f"Lesson number {self.extractions}, written out fully."], n_failures=len(failures))


@pytest.fixture
def h(tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch)


async def test_each_iteration_receives_exactly_the_lessons_learned_before_it(h):
    await loop.run_loop("t", iterations=3)

    assert h.agent_runs[0] == (0, None)
    assert h.agent_runs[1] == (1, "1. Lesson number 1, written out fully.")
    assert h.agent_runs[2] == (2, "1. Lesson number 1, written out fully.\n2. Lesson number 2, written out fully.")
    assert h.scored == ["iteration_0_t", "iteration_1_t", "iteration_2_t"]
    assert h.extractions == 2  # after iterations 0 and 1 — never after the last one


async def test_snapshots_freeze_what_the_agent_was_told(h):
    await loop.run_loop("t", iterations=3)
    assert (h.dir / "lessons_iteration_1_t.md").read_text() == "1. Lesson number 1, written out fully."
    # a later extraction must not rewrite an earlier snapshot
    assert "Lesson number 2" not in (h.dir / "lessons_iteration_1_t.md").read_text()


async def test_resuming_skips_finished_steps_and_does_not_re_extract(h):
    await loop.run_loop("t", iterations=2)
    first_agent, first_extractions = list(h.agent_runs), h.extractions

    await loop.run_loop("t", iterations=2)  # nothing left to do
    assert h.agent_runs == first_agent and h.extractions == first_extractions

    await loop.run_loop("t", iterations=3)  # extend: only iteration 2 (+ one extraction) is new
    assert [n for n, _ in h.agent_runs] == [0, 1, 2]
    assert h.extractions == first_extractions + 1


async def test_crash_between_saving_lessons_and_writing_the_snapshot_does_not_double_add(h):
    await loop.run_loop("t", iterations=1)  # iteration 0 run + scored, no extraction (last iteration)

    store = LessonStore(h.dir / "lessons_t.json")
    store.add(0, ExtractionOutcome(lessons=["Lesson saved just before the simulated crash."], n_failures=1))
    store.save()  # saved, but lessons_iteration_1_t.md never written

    await loop.run_loop("t", iterations=2)

    assert h.extractions == 0  # did NOT extract again
    assert h.agent_runs[-1] == (1, "1. Lesson saved just before the simulated crash.")
    assert len(LessonStore(h.dir / "lessons_t.json").texts) == 1


def test_trend_table_reads_the_scores_files(h, capsys):
    (h.dir / "scores_iteration_0_t.json").write_text(json.dumps({"by_split": {
        "train": {"mean_overall": 3.06, "rule_violations": 8, "prompts": 10},
        "heldout": {"mean_overall": 2.9, "rule_violations": 4, "prompts": 5}}}))
    loop.print_trend("t", 1)
    out = capsys.readouterr().out
    assert "3.06 / 8" in out and "2.90 / 4" in out
