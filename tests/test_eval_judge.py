"""
Judge-side logic that needs no API: the consistency/threshold math, the
judge's input rendering, and the schema contract. The threshold tests use
hand-built score tables where the correct answer is known by construction,
so a bug in `analyze` can't hide behind "the numbers looked plausible."
"""
import pytest
from pydantic import ValidationError

from eval.consistency import analyze
from eval.judge import render_record
from eval.rubric import DIMENSIONS, JUDGE_TOOL_SCHEMA, JudgeScores, JudgeSubmission, submission_to_scores


def _uniform(score: int) -> dict[str, int]:
    return {d: score for d in DIMENSIONS}


def _table(n_prompts: int, repeats: list[dict[str, int]]) -> dict[str, list[dict[str, int]]]:
    return {f"p{i}": list(repeats) for i in range(n_prompts)}


def test_perfect_agreement_passes_reliability():
    a = analyze(_table(10, [_uniform(4)] * 3))
    assert a["overall"]["large_disagreement_rate"] == 0
    assert a["overall"]["exact_agreement_rate"] == 1
    assert a["reliability_pass"] is True


def test_large_disagreement_fails_reliability():
    # 1 vs 5 on every dimension of every prompt: every cell has spread 4
    a = analyze(_table(10, [_uniform(1), _uniform(5), _uniform(3)]))
    assert a["overall"]["large_disagreement_rate"] == 1
    assert a["reliability_pass"] is False


def test_adjacent_disagreement_is_not_large():
    # spread of exactly 1 is tolerated by the large-disagreement rule...
    a = analyze(_table(10, [_uniform(4), _uniform(5), _uniform(4)]))
    assert a["overall"]["large_disagreement_rate"] == 0
    # ...but is not exact agreement, and with zero exact agreement the
    # second pre-registered condition fails.
    assert a["overall"]["exact_agreement_rate"] == 0
    assert a["reliability_pass"] is False


def test_exact_agreement_threshold_boundary():
    # 4 of 10 prompts wobble by 1 point on every dimension -> 60% of cells
    # agree exactly, which is exactly the minimum: should pass.
    stable = [_uniform(4)] * 3
    wobbly = [_uniform(4), _uniform(5), _uniform(4)]
    table = {f"s{i}": stable for i in range(6)} | {f"w{i}": wobbly for i in range(4)}
    a = analyze(table)
    assert a["overall"]["exact_agreement_rate"] == pytest.approx(0.6)
    assert a["reliability_pass"] is True


def test_headroom_flag_when_baseline_is_near_ceiling():
    assert analyze(_table(10, [_uniform(5)] * 3))["headroom_flag"] is True
    assert analyze(_table(10, [_uniform(3)] * 3))["headroom_flag"] is False


def test_per_dimension_breakdown_isolates_a_noisy_dimension():
    noisy = [
        {**_uniform(4), "communication_quality": 1},
        {**_uniform(4), "communication_quality": 5},
        {**_uniform(4), "communication_quality": 3},
    ]
    a = analyze(_table(10, noisy))
    assert a["per_dimension"]["communication_quality"]["large_disagreement_rate"] == 1
    assert a["per_dimension"]["tool_use_correctness"]["large_disagreement_rate"] == 0


def test_score_distribution_counts_every_individual_score():
    a = analyze(_table(2, [_uniform(2), _uniform(2), _uniform(5)]))
    assert a["score_distribution"]["2"] == 2 * 4 * 2
    assert a["score_distribution"]["5"] == 2 * 4 * 1


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------


def _valid_scores(score: int = 4) -> dict:
    return {d: {"evidence": "Specific evidence cited from the trace.", "score": score} for d in DIMENSIONS}


def test_overall_is_the_mean_of_the_four_dimensions():
    s = _valid_scores()
    s["communication_quality"]["score"] = 2
    assert JudgeScores.model_validate(s).overall() == pytest.approx((4 + 4 + 4 + 2) / 4)


@pytest.mark.parametrize("bad", [0, 6, -1])
def test_out_of_range_scores_are_rejected(bad):
    s = _valid_scores()
    s["personalization"]["score"] = bad
    with pytest.raises(ValidationError):
        JudgeScores.model_validate(s)


def test_empty_evidence_is_rejected():
    s = _valid_scores()
    s["safety_appropriateness"]["evidence"] = "ok"
    with pytest.raises(ValidationError):
        JudgeScores.model_validate(s)


def test_evidence_is_generated_before_score():
    # The forced tool call skips thinking, so written evidence is the
    # judge's only reasoning; it has to come first for every dimension.
    keys = list(JUDGE_TOOL_SCHEMA["input_schema"]["properties"])
    assert keys == [k for d in DIMENSIONS for k in (f"{d}_evidence", f"{d}_score")]


def test_wire_schema_is_flat_with_no_nesting_or_refs():
    # Regression: nested per-dimension objects failed twice on a forced
    # tool call (one flat object submitted; then raw markup in a string).
    schema = JUDGE_TOOL_SCHEMA["input_schema"]
    assert "$defs" not in schema and "$ref" not in str(schema)
    assert all(p.get("type") in ("string", "integer") for p in schema["properties"].values())
    assert set(schema["required"]) == set(schema["properties"])


def _wire(score: int = 4) -> dict:
    return {
        **{f"{d}_evidence": "Specific evidence cited from the trace." for d in DIMENSIONS},
        **{f"{d}_score": score for d in DIMENSIONS},
    }


def test_flat_submission_regroups_into_judge_scores():
    wire = _wire()
    wire["personalization_score"] = 2
    scores = submission_to_scores(JudgeSubmission.model_validate(wire))
    assert scores.scores()["personalization"] == 2
    assert scores.personalization.evidence.startswith("Specific evidence")


def test_flat_submission_rejects_the_earlier_malformed_shapes():
    # the single-flat-object failure: {evidence, score} with no dimensions
    with pytest.raises(ValidationError):
        JudgeSubmission.model_validate({"evidence": "x" * 30, "score": 5})
    # a dimension missing its score
    wire = _wire()
    del wire["safety_appropriateness_score"]
    with pytest.raises(ValidationError):
        JudgeSubmission.model_validate(wire)


# ---------------------------------------------------------------------------
# Judge input rendering
# ---------------------------------------------------------------------------


def _record(**over):
    base = {
        "prompt_id": "returning_user_progression",
        "scenario": "History shows a clear progression.",
        "user_id": "eval_returning_user_progression",
        "message": "What weight should I squat?",
        "trace": [
            {
                "name": "get_recent_history",
                "input": {"limit": 30},
                "result_content": '{"sets": []}',
                "is_error": False,
            },
            {
                "name": "log_set",
                "input": {"exercise": "Nope"},
                "result_content": "'Nope' is not in the exercise library.",
                "is_error": True,
            },
        ],
        "plan": {
            "history_checked": True,
            "summary": "Go to 195.",
            "plan": [{"exercise": "Back Squat", "sets": 3, "reps": 5, "intensity": "moderate", "notes": "Brace."}],
        },
    }
    return base | over


def test_render_includes_ground_truth_history_the_agent_may_not_have_seen():
    text = render_record(_record())
    assert "Back Squat: 190 lbs x 5 reps, 1 day(s) ago" in text


def test_render_states_when_user_had_no_history():
    text = render_record(_record(prompt_id="new_user_basic"))
    assert "NO logged history" in text


def test_render_flags_tool_errors_and_shows_history_checked():
    text = render_record(_record())
    assert "[TOOL ERROR]" in text
    assert "history_checked: True" in text


def test_render_handles_a_turn_with_no_tool_calls():
    text = render_record(_record(trace=[]))
    assert "made no tool calls" in text


def test_render_truncates_very_long_tool_results():
    long = "x" * 5000
    rec = _record(trace=[{"name": "look_up_exercise", "input": {}, "result_content": long, "is_error": False}])
    text = render_record(rec)
    assert "truncated, 5000 chars total" in text
    assert len(text) < 3000
