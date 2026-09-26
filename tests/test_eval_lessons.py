"""
Lesson extraction machinery, tested without the API: a scripted fake
extractor returns canned tool inputs so lint, retry, cap, and the
"what the extractor is allowed to see" boundaries are all checkable.
"""
from types import SimpleNamespace

import pytest

from eval.lessons import (
    LOW_SCORE_THRESHOLD,
    MAX_LESSON_CHARS,
    MAX_TOTAL_LESSONS,
    ExtractionError,
    ExtractionOutcome,
    LessonStore,
    build_extraction_input,
    build_forbidden,
    extract_lessons,
    lint_lesson,
    select_failures,
)
from eval.rubric import DIMENSIONS

GOOD = "Before recommending any exercise, confirm it exists in the library with a lookup and use the exact returned name."
GOOD2 = "When the user reports completed work, record each set separately before answering the rest of the request."
ANALYSIS = "The agent skipped verification and named exercises from memory across several responses."


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------


def test_lint_accepts_a_general_lesson():
    assert lint_lesson(GOOD, build_forbidden()) is None


@pytest.mark.parametrize("name", ["Romanian Deadlift", "romanian deadlift", "BACK SQUAT"])
def test_lint_rejects_library_exercise_names_any_case(name):
    reason = lint_lesson(f"Always warm up before {name} sets, however light.", build_forbidden())
    assert reason and "library exercise" in reason


def test_lint_rejects_a_five_word_run_from_a_train_prompt():
    # train prompt: "Give me something to do."
    reason = lint_lesson("If a user says give me something to do, ask one clarifying question first.", build_forbidden())
    assert reason and "eval prompt" in reason


def test_lint_also_guards_held_out_prompts():
    # held-out prompt: "What should I bench today?" is only 4 words, so use a longer one:
    # "My lower back is sore but I want to do heavy ..."
    reason = lint_lesson("Note that my lower back is sore but I want to continue is a red flag.", build_forbidden())
    assert reason and "eval prompt" in reason


def test_lint_boundary_is_five_words_not_fewer():
    # exactly the 5-word run 'give me something to do' -> rejected
    assert lint_lesson("Treat any request to give me something to do as underspecified.", build_forbidden()) is not None
    # only a 3-word overlap ('something to do') -> allowed
    assert lint_lesson("Requests like 'something to do' with no goal are underspecified: ask or state assumptions.", build_forbidden()) is None


# ---------------------------------------------------------------------------
# what the extractor is allowed to learn from
# ---------------------------------------------------------------------------


def _record(pid, split="train", scenario="SCENARIO_MARKER_DO_NOT_LEAK"):
    return {
        "prompt_id": pid,
        "split": split,
        "scenario": scenario,
        "message": f"message for {pid}",
        "trace": [{"name": "get_recent_history", "input": {}, "result_content": '{"sets": []}', "is_error": False}],
        "plan": {"history_checked": True, "summary": "s", "plan": []},
    }


def _scores(entries):
    """entries: pid -> (overall, {dim: [score per repeat]})"""
    per_prompt, raw = {}, {}
    for pid, (overall, dim_scores) in entries.items():
        per_prompt[pid] = {"overall": overall, "dimensions": {d: sum(v) / len(v) for d, v in dim_scores.items()}}
        raw[pid] = [
            {d: {"score": dim_scores[d][k], "evidence": f"evidence {pid} {d} #{k}"} for d in DIMENSIONS}
            for k in range(len(next(iter(dim_scores.values()))))
        ]
    return {"per_prompt": per_prompt, "raw_judgments": raw}


def _dims(v):
    return {d: v for d in DIMENSIONS}


def test_select_failures_takes_only_low_scoring_train_outputs():
    run = {"results": [_record("new_user_basic"), _record("new_user_specific_goal"), _record("heldout_informal_names", split="heldout")]}
    scores = _scores(
        {
            "new_user_basic": (2.0, _dims([2, 2, 2])),
            "new_user_specific_goal": (LOW_SCORE_THRESHOLD, _dims([4, 3, 4])),  # exactly at threshold: not a failure
            "heldout_informal_names": (1.0, _dims([1, 1, 1])),  # low, but held-out
        }
    )
    assert [f.prompt_id for f in select_failures(run, scores)] == ["new_user_basic"]


def test_records_from_older_runs_without_a_split_count_as_train():
    rec = _record("new_user_basic")
    del rec["split"]
    scores = _scores({"new_user_basic": (2.0, _dims([2, 2, 2]))})
    assert len(select_failures({"results": [rec]}, scores)) == 1


def test_weak_evidence_uses_the_lowest_judgment_and_skips_strong_dimensions():
    dims = _dims([5, 5, 5])
    dims["personalization"] = [3, 1, 2]
    scores = _scores({"new_user_basic": (3.0, dims)})
    (case,) = select_failures({"results": [_record("new_user_basic")]}, scores)
    assert set(case.weak_evidence) == {"personalization"}
    assert case.weak_evidence["personalization"][1] == "evidence new_user_basic personalization #1"  # the score-1 repeat


def test_extractor_input_never_contains_scenario_text_or_heldout_messages():
    run = {"results": [_record("new_user_basic"), _record("heldout_informal_names", split="heldout")]}
    scores = _scores({"new_user_basic": (2.0, _dims([2, 2, 2])), "heldout_informal_names": (1.0, _dims([1, 1, 1]))})
    text = build_extraction_input(select_failures(run, scores), existing=[GOOD])
    assert "SCENARIO_MARKER_DO_NOT_LEAK" not in text
    assert "message for heldout_informal_names" not in text
    assert "message for new_user_basic" in text
    assert GOOD in text  # existing lessons are shown so it can avoid duplicating them


# ---------------------------------------------------------------------------
# extraction: retry, cap, errors
# ---------------------------------------------------------------------------


def _sub(*lessons):
    return {"failure_analysis": ANALYSIS, **{f"lesson_{i}": l for i, l in enumerate(lessons, 1)}}


class ScriptedExtractor:
    def __init__(self, *tool_inputs):
        self.script = list(tool_inputs)
        self.calls = []
        self.messages = self

    async def create(self, **kw):
        self.calls.append(kw)
        inp = self.script.pop(0)
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[SimpleNamespace(type="tool_use", input=inp)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


def _failures():
    run = {"results": [_record("new_user_basic")]}
    return select_failures(run, _scores({"new_user_basic": (2.0, _dims([2, 2, 2]))}))


async def test_clean_lessons_are_returned_and_recorded():
    client = ScriptedExtractor(_sub(GOOD, GOOD2))
    out = await extract_lessons(client, _failures(), existing=[])
    assert out.lessons == [GOOD, GOOD2]
    assert out.dropped == [] and out.attempts == 1
    assert (out.input_tokens, out.output_tokens) == (10, 5)
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": "submit_lessons"}


async def test_a_lint_rejection_triggers_one_retry_with_feedback_and_is_logged():
    leaky = "Always squat deep when programming Back Squat sets for anyone, every single time."
    client = ScriptedExtractor(
        _sub(leaky, GOOD),
        _sub(GOOD2),
    )
    out = await extract_lessons(client, _failures(), existing=[])
    assert out.lessons == [GOOD, GOOD2]
    assert out.attempts == 2
    assert [d["lesson"] for d in out.dropped] == [leaky]
    assert "library exercise" in out.dropped[0]["reason"]
    assert "were rejected" in client.calls[1]["messages"][0]["content"]  # feedback reached the 2nd call


async def test_lessons_still_leaky_after_the_retry_are_dropped_not_kept():
    leaky = "Always squat deep when programming Back Squat sets for anyone, every single time."
    client = ScriptedExtractor(
        _sub(leaky),
        _sub(leaky),
        _sub(leaky),
    )
    out = await extract_lessons(client, _failures(), existing=[])
    assert out.lessons == []
    assert len(out.dropped) == 3


async def test_never_exceeds_the_remaining_room_under_the_total_cap():
    existing = [f"existing lesson number {i} written out in full" for i in range(MAX_TOTAL_LESSONS - 1)]
    client = ScriptedExtractor(_sub(GOOD, GOOD2))
    out = await extract_lessons(client, _failures(), existing=existing)
    assert out.lessons == [GOOD]  # only one slot left


async def test_at_the_cap_it_does_not_call_the_api():
    client = ScriptedExtractor()
    out = await extract_lessons(client, _failures(), existing=["x" * 30] * MAX_TOTAL_LESSONS)
    assert out.lessons == [] and "cap" in out.note and client.calls == []


async def test_no_failures_means_no_api_call():
    client = ScriptedExtractor()
    out = await extract_lessons(client, [], existing=[])
    assert out.lessons == [] and client.calls == []


async def test_zero_lessons_is_a_valid_answer():
    client = ScriptedExtractor(_sub())
    out = await extract_lessons(client, _failures(), existing=[])
    assert out.lessons == [] and out.dropped == []


async def test_repeated_schema_failures_raise():
    bad = {"failure_analysis": "too short"}
    with pytest.raises(ExtractionError):
        await extract_lessons(ScriptedExtractor(bad, bad, bad), _failures(), existing=[])


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


def test_store_is_append_only_persists_and_logs_rejections(tmp_path):
    path = tmp_path / "lessons.json"
    store = LessonStore(path)
    store.add(0, ExtractionOutcome(lessons=[GOOD], failure_analysis=ANALYSIS, n_failures=4, dropped=[{"lesson": "bad", "reason": "why"}]))
    store.add(1, ExtractionOutcome(lessons=[GOOD2], failure_analysis=ANALYSIS, n_failures=2))
    store.save()

    again = LessonStore(path)
    assert again.texts == [GOOD, GOOD2]
    assert [l["id"] for l in again.data["lessons"]] == [1, 2]
    assert [l["added_after_iteration"] for l in again.data["lessons"]] == [0, 1]
    assert again.data["log"][0]["rejected_by_lint"] == [{"lesson": "bad", "reason": "why"}]
    assert again.render() == f"1. {GOOD}\n2. {GOOD2}"


def test_empty_store_renders_empty_text(tmp_path):
    assert LessonStore(tmp_path / "x.json").render() == ""


# ---------------------------------------------------------------------------
# wire format regression: array/nested params failed on forced tool calls
# ---------------------------------------------------------------------------


def test_extraction_schema_is_flat_scalars_only():
    from eval.lessons import SUBMIT_TOOL

    props = SUBMIT_TOOL["input_schema"]["properties"]
    assert list(props) == ["failure_analysis", "lesson_1", "lesson_2", "lesson_3"]
    assert all(p["type"] == "string" for p in props.values())
    assert "$defs" not in SUBMIT_TOOL["input_schema"]


def test_a_string_where_an_array_was_expected_is_simply_an_unknown_field():
    # the observed failure: 'lessons' came back as a string of raw markup
    from eval.lessons import ExtractionSubmission

    sub = ExtractionSubmission.model_validate({"failure_analysis": ANALYSIS, "lessons": '<parameter name="lessons">["x"]'})
    assert sub.lessons == []  # yields no lessons rather than garbage lessons


def test_overlong_and_stubby_lessons_are_rejected():
    from eval.lessons import ExtractionSubmission
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExtractionSubmission.model_validate({"failure_analysis": ANALYSIS, "lesson_1": "x" * (MAX_LESSON_CHARS + 1)})
    with pytest.raises(ValidationError):
        ExtractionSubmission.model_validate({"failure_analysis": ANALYSIS, "lesson_1": "too short"})


async def test_a_schema_failure_retry_carries_the_error_back_to_the_model():
    bad = {"failure_analysis": ANALYSIS, "lesson_1": "x" * (MAX_LESSON_CHARS + 1)}
    client = ScriptedExtractor(bad, _sub(GOOD))
    out = await extract_lessons(client, _failures(), existing=[])
    assert out.lessons == [GOOD] and out.attempts == 2
    assert "invalid" in client.calls[1]["messages"][0]["content"]
    assert "lesson_1" in client.calls[1]["messages"][0]["content"]  # says WHICH field
