"""The deterministic rule checks, against synthetic traces with known answers."""
import json

from eval.rules import violations


def _call(name, result, is_error=False, inp=None):
    return {"name": name, "input": inp or {}, "result_content": json.dumps(result) if not isinstance(result, str) else result, "is_error": is_error}


def _rec(pid, trace, exercises):
    plan = [{"exercise": e, "sets": 3, "reps": 5, "intensity": "moderate"} for e in exercises]
    return {"prompt_id": pid, "trace": trace, "plan": {"plan": plan}}


HISTORY = _call("get_recent_history", {"sets": [], "count": 0})
LOOKUP_SQUAT = _call("look_up_exercise", {"matches": [{"name": "Back Squat"}], "count": 1})


def test_clean_record_has_no_violations():
    assert violations(_rec("new_user_basic", [HISTORY, LOOKUP_SQUAT], ["Back Squat"])) == []


def test_never_checking_history_is_flagged():
    assert "never_checked_history" in violations(_rec("new_user_basic", [LOOKUP_SQUAT], ["Back Squat"]))


def test_unverified_plan_exercises_are_counted():
    v = violations(_rec("new_user_basic", [HISTORY, LOOKUP_SQUAT], ["Back Squat", "Made Up Lift", "Also Fake"]))
    assert "2_unverified_plan_exercises" in v


def test_exercise_named_in_users_own_history_counts_as_verified():
    hist = _call("get_recent_history", {"sets": [{"exercise": "Bench Press"}], "count": 1})
    assert violations(_rec("new_user_basic", [hist], ["Bench Press"])) == []


def test_a_failed_lookup_verifies_nothing():
    bad = _call("look_up_exercise", "boom", is_error=True)
    v = violations(_rec("new_user_basic", [HISTORY, bad], ["Back Squat"]))
    assert "1_unverified_plan_exercises" in v and "tool_error" in v


def test_logging_when_nothing_was_completed_is_flagged():
    logged = _call("log_set", {"success": True})
    assert "log_set_count_1_expected_0" in violations(_rec("should_not_log", [HISTORY, LOOKUP_SQUAT, logged], ["Back Squat"]))


def test_expected_log_count_for_the_completed_set_prompt():
    logs = [_call("log_set", {"success": True})] * 3
    ok = violations(_rec("log_a_completed_set", [HISTORY, LOOKUP_SQUAT, *logs], ["Back Squat"]))
    assert not any(v.startswith("log_set_count") for v in ok)
    missed = violations(_rec("log_a_completed_set", [HISTORY, LOOKUP_SQUAT], ["Back Squat"]))
    assert "log_set_count_0_expected_3" in missed


def test_a_failed_log_set_does_not_count_as_logged():
    failed = _call("log_set", "not in library", is_error=True)
    v = violations(_rec("log_a_completed_set", [HISTORY, LOOKUP_SQUAT, failed], ["Back Squat"]))
    assert "log_set_count_0_expected_3" in v and "tool_error" in v
