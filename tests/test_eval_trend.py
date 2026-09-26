"""
The pre-registered success criterion, applied mechanically. The cases that
matter are the ones that could flatter a result: a positive gain that's
inside the noise, a gain with WORSE rule violations, and a train-only gain.
"""
import json
import xml.etree.ElementTree as ET

import pytest

from eval.rubric import DIMENSIONS
from eval.trend import analyze_run, format_report, noise_floor, render_svg, verdict


def _run(train, held, tv=5, hv=2):
    return {"train": {"mean": train, "violations": tv}, "heldout": {"mean": held, "violations": hv}}


REPS = [_run(3.0, 3.0), _run(3.2, 3.2), _run(3.1, 3.1), _run(3.1, 3.1)]  # floor 0.2, baseline 3.1


def test_noise_floor_is_the_largest_gap_between_any_two_replicates():
    assert noise_floor([3.0, 3.2, 3.1]) == pytest.approx(0.2)
    assert noise_floor([3.0]) == 0.0


def test_real_improvement_needs_a_heldout_gain_beyond_the_noise_floor():
    v = verdict(REPS, _run(3.6, 3.6, tv=2, hv=1))
    assert v["heldout"]["gain"] == pytest.approx(0.5)
    assert v["heldout"]["beats_noise_floor"] and v["real_improvement"] is True


def test_a_positive_gain_inside_the_noise_is_not_an_improvement():
    v = verdict(REPS, _run(3.25, 3.25, hv=0))  # +0.15 < 0.2 floor
    assert v["heldout"]["gain"] > 0
    assert v["real_improvement"] is False


def test_a_gain_exactly_at_the_floor_does_not_count():
    assert verdict(REPS, _run(3.3, 3.3, hv=0))["real_improvement"] is False  # +0.2 == floor, must EXCEED


def test_a_score_gain_with_more_rule_violations_is_not_an_improvement():
    v = verdict(REPS, _run(3.9, 3.9, hv=5))  # baseline held-out violations = 2
    assert v["heldout"]["beats_noise_floor"] is True
    assert v["heldout"]["violations_not_higher"] is False
    assert v["real_improvement"] is False


def test_train_only_gain_is_flagged_as_overfitting_and_is_not_a_win():
    v = verdict(REPS, _run(3.9, 3.1, hv=0))
    assert v["train"]["beats_noise_floor"] and not v["heldout"]["beats_noise_floor"]
    assert v["overfitting_flag"] is True and v["real_improvement"] is False


def test_a_regression_is_reported_as_negative_gain_not_hidden():
    v = verdict(REPS, _run(2.5, 2.5))
    assert v["heldout"]["gain"] < 0 and v["real_improvement"] is False


# ---------------------------------------------------------------------------
# end to end from score files
# ---------------------------------------------------------------------------


def _scores_file(train, held, tv, hv):
    pids = {"t1": "train", "t2": "train", "h1": "heldout"}
    dims = lambda v: {d: v for d in DIMENSIONS}
    return {
        "by_split": {"train": {"mean_overall": train, "rule_violations": tv, "prompts": 2},
                     "heldout": {"mean_overall": held, "rule_violations": hv, "prompts": 1}},
        "split_of": pids,
        "per_prompt": {"t1": {"dimensions": dims(train)}, "t2": {"dimensions": dims(train)}, "h1": {"dimensions": dims(held)}},
    }


def test_analyze_run_counts_the_loops_own_iteration_0_as_a_replicate(tmp_path):
    for name, args in {
        "iteration_0_loop": (3.0, 3.0, 4, 2), "iteration_1_loop": (3.3, 3.2, 3, 2), "iteration_2_loop": (3.8, 3.7, 2, 1),
        "iteration_0_repA": (3.1, 3.1, 5, 2), "iteration_0_repB": (3.2, 3.2, 4, 2),
    }.items():
        (tmp_path / f"scores_{name}.json").write_text(json.dumps(_scores_file(*args)))

    a = analyze_run("loop", 3, ["iteration_0_repA", "iteration_0_repB"], runs_dir=tmp_path)

    assert a["verdict"]["n_replicates"] == 3  # loop_0 + repA + repB
    assert [round(p["mean"], 2) for p in a["series"]["heldout"]] == [3.0, 3.2, 3.7]
    assert a["verdict"]["heldout"]["noise_floor"] == pytest.approx(0.2)
    assert a["verdict"]["real_improvement"] is True  # 3.7 - 3.1 = 0.6 > 0.2, violations 1 <= 2


def test_report_states_the_verdict_and_the_numbers(tmp_path):
    for name, args in {"iteration_0_loop": (3.0, 3.0, 4, 2), "iteration_1_loop": (3.05, 3.05, 4, 2), "iteration_0_r": (3.2, 3.2, 4, 2)}.items():
        (tmp_path / f"scores_{name}.json").write_text(json.dumps(_scores_file(*args)))
    text = format_report(analyze_run("loop", 2, ["iteration_0_r"], runs_dir=tmp_path))
    assert "REAL IMPROVEMENT (pre-registered criterion): NO" in text and "noise floor" in text


def test_svg_is_well_formed_and_labelled():
    series = {s: [{"mean": 3.0 + 0.1 * i, "violations": 4 - i} for i in range(5)] for s in ("train", "heldout")}
    svg = render_svg(series, verdict(REPS, _run(3.4, 3.4)))
    root = ET.fromstring(svg)  # raises if malformed
    assert root.tag.endswith("svg")
    assert "Held-out" in svg and "Rule violations" in svg and "noise floor" in svg
