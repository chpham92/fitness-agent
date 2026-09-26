"""
The blind pairwise check's mechanics. The dangerous bugs are silent ones:
a mapping error that inverts who won, an unbalanced order that bakes in
position bias, or a prompt that leaks which response is the final iteration.
"""
import pytest

from eval.pairwise import PAIRWISE_TOOL, Pair, build_pairs, final_outcome, render_pair, summarize

BASELINES = ["b0", "b1", "b2", "b3"]


def test_final_is_in_position_a_for_exactly_half_of_each_prompts_comparisons():
    pairs = build_pairs(["p1", "p2", "p3"], BASELINES, seed=7)
    for pid in ("p1", "p2", "p3"):
        mine = [p for p in pairs if p.prompt_id == pid]
        assert len(mine) == 4 and sum(p.final_is_a for p in mine) == 2
        assert {p.baseline for p in mine} == set(BASELINES)  # every baseline compared exactly once


def test_pairing_is_deterministic_per_seed_and_varies_across_prompts():
    assert build_pairs(["p1", "p2"], BASELINES, seed=1) == build_pairs(["p1", "p2"], BASELINES, seed=1)
    orders = {tuple(p.final_is_a for p in build_pairs([f"p{i}"], BASELINES, seed=1)) for i in range(12)}
    assert len(orders) > 1  # not the same pattern for every prompt


@pytest.mark.parametrize(
    "winner, final_is_a, expected",
    [("A", True, 1.0), ("B", True, 0.0), ("A", False, 0.0), ("B", False, 1.0), ("tie", True, 0.5), ("tie", False, 0.5)],
)
def test_who_won_maps_back_to_final_vs_baseline_in_both_positions(winner, final_is_a, expected):
    assert final_outcome(winner, final_is_a) == expected


def _row(pid, outcome, final_is_a=True, longer=True, split="train"):
    return {"prompt_id": pid, "split": split, "outcome": outcome, "final_is_a": final_is_a, "final_longer": longer}


def test_summary_counts_and_breakdowns():
    rows = [_row("p1", 1.0), _row("p1", 1.0, final_is_a=False), _row("p2", 0.0, longer=False, split="heldout"), _row("p2", 0.5, final_is_a=False, longer=False, split="heldout")]
    s = summarize(rows, resamples=200)
    assert (s["final_wins"], s["baseline_wins"], s["ties"]) == (2, 1, 1)
    assert s["final_win_rate"] == pytest.approx(0.625)
    assert s["by_split"] == {"heldout": pytest.approx(0.25), "train": pytest.approx(1.0)}
    assert s["by_position"] == {"final_is_A": pytest.approx(0.5), "final_is_B": pytest.approx(0.75)}
    assert s["by_length"]["final_summary_longer"] == 1.0 and s["by_length"]["n_shorter_or_equal"] == 2
    lo, hi = s["win_rate_ci95_over_prompts"]
    assert 0.0 <= lo <= s["final_win_rate"] <= hi <= 1.0


def _rec(marker):
    return {"trace": [{"name": "get_recent_history", "input": {}, "result_content": f"RESULT_{marker}", "is_error": False}],
            "plan": {"history_checked": True, "summary": f"SUMMARY_{marker}", "plan": []}}


def test_rendered_prompt_puts_records_in_the_given_order_and_leaks_nothing_about_origin():
    text = render_pair("new_user_basic", "Give me a quick full body workout for today.", _rec("FIRST"), _rec("SECOND"))
    assert text.index("SUMMARY_FIRST") < text.index("Response B") < text.index("SUMMARY_SECOND")
    for leak in ("iteration", "baseline", "lesson", "final", "loop", "haiku"):
        assert leak not in text.lower(), leak


def test_wire_schema_is_flat_and_reasoning_comes_before_the_choice():
    props = PAIRWISE_TOOL["input_schema"]["properties"]
    assert list(props) == ["analysis", "winner"]
    assert props["winner"]["enum"] == ["A", "B", "tie"]
    assert "$defs" not in PAIRWISE_TOOL["input_schema"]
