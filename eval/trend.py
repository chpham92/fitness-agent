"""
Day 4 analysis: apply the README's pre-registered success criterion to a
finished loop run, and draw the trend.

The criterion was fixed before any lesson was extracted. Restated:

  noise floor (per split) = largest difference in mean overall score
      between any two no-lessons replicate runs (all 15 prompts each).
  REAL IMPROVEMENT  iff  the final iteration's HELD-OUT mean overall
      exceeds the replicate mean by more than the held-out noise floor,
      AND held-out rule violations are not higher than the replicate mean.
  Train-only gains with no held-out gain are reported as overfitting.

`verdict` is a pure function so the logic is unit tested; nothing here
reads a threshold from anywhere but the arguments.

The chart is hand-written SVG on purpose: no plotting dependency, and it
renders directly in a GitHub README.

Usage:
    python -m eval.trend --tag loop --iterations 5 \\
        --replicates iteration_0_dry iteration_0_rep1 iteration_0_rep2
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

from eval.rubric import DIMENSIONS

RUNS_DIR = Path(__file__).parent / "runs"
SPLITS = ("train", "heldout")


def _load(name: str, runs_dir: Path) -> dict[str, Any]:
    return json.loads((runs_dir / f"scores_{name}.json").read_text(encoding="utf-8"))


def split_dimension_means(scores: dict[str, Any], split: str) -> dict[str, float]:
    pids = [p for p, s in scores["split_of"].items() if s == split]
    return {d: statistics.mean(scores["per_prompt"][p]["dimensions"][d] for p in pids) for d in DIMENSIONS}


def summarize(scores: dict[str, Any]) -> dict[str, dict[str, float]]:
    """{split: {mean, violations}} for one scored run."""
    return {s: {"mean": scores["by_split"][s]["mean_overall"], "violations": scores["by_split"][s]["rule_violations"]} for s in SPLITS}


def noise_floor(means: list[float]) -> float:
    return max((abs(a - b) for a, b in itertools.combinations(means, 2)), default=0.0)


def verdict(replicates: list[dict], final: dict) -> dict[str, Any]:
    """replicates / final: outputs of `summarize`."""
    out: dict[str, Any] = {"n_replicates": len(replicates)}
    for s in SPLITS:
        means = [r[s]["mean"] for r in replicates]
        base = statistics.mean(means)
        floor = noise_floor(means)
        gain = final[s]["mean"] - base
        out[s] = {
            "replicate_means": means,
            "baseline_mean": base,
            "noise_floor": floor,
            "final_mean": final[s]["mean"],
            "gain": gain,
            "beats_noise_floor": gain > floor,
            "baseline_violations": statistics.mean(r[s]["violations"] for r in replicates),
            "final_violations": final[s]["violations"],
        }
    h, t = out["heldout"], out["train"]
    h["violations_not_higher"] = h["final_violations"] <= h["baseline_violations"]
    out["real_improvement"] = bool(h["beats_noise_floor"] and h["violations_not_higher"])
    out["overfitting_flag"] = bool(t["beats_noise_floor"] and not h["beats_noise_floor"])
    return out


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

_COLORS = {"train": "#2563eb", "heldout": "#ea580c"}


def _panel(x0, y0, w, h, title, ylim, xs, lines, band=None, dots=None, fmt="{:.1f}"):
    """lines: [(label, color, [y per x])]; band: (lo, hi); dots: [y] plotted at x=0."""
    lo, hi = ylim
    px = lambda i: x0 + 40 + (w - 60) * (i / max(1, len(xs) - 1))
    py = lambda v: y0 + h - 25 - (h - 50) * ((v - lo) / (hi - lo))
    o = [f'<text x="{x0 + 40}" y="{y0 + 12}" font-size="13" font-weight="600" fill="#111">{title}</text>']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        o.append(f'<line x1="{x0 + 40}" x2="{x0 + w - 20}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="#e5e7eb"/>')
        o.append(f'<text x="{x0 + 34}" y="{py(v) + 4:.1f}" font-size="10" text-anchor="end" fill="#6b7280">{fmt.format(v)}</text>')
    for i, x in enumerate(xs):
        o.append(f'<text x="{px(i):.1f}" y="{y0 + h - 8}" font-size="10" text-anchor="middle" fill="#6b7280">{x}</text>')
    if band:
        o.append(f'<rect x="{x0 + 40}" y="{py(band[1]):.1f}" width="{w - 60}" height="{py(band[0]) - py(band[1]):.1f}" fill="#9ca3af" opacity="0.25"/>')
    for d in dots or []:
        o.append(f'<circle cx="{px(0):.1f}" cy="{py(d):.1f}" r="4" fill="none" stroke="#6b7280"/>')
    for label, color, ys in lines:
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ys))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for i, v in enumerate(ys):
            o.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3" fill="{color}"/>')
    return "\n".join(o)


def render_svg(series: dict[str, list[dict]], result: dict[str, Any]) -> str:
    n = len(series["train"])
    xs = [str(i) for i in range(n)]
    w, h = 330, 230
    panels = []
    for k, s in enumerate(SPLITS):
        r = result[s]
        vals = [p["mean"] for p in series[s]] + r["replicate_means"]
        lo, hi = min(vals + [r["baseline_mean"] - r["noise_floor"]]) - 0.2, max(vals + [r["baseline_mean"] + r["noise_floor"]]) + 0.2
        panels.append(
            _panel(
                k * w, 0, w, h, f"{'Train' if s == 'train' else 'Held-out'}: mean judge score (1-5), by iteration", (lo, hi), xs,
                [(s, _COLORS[s], [p["mean"] for p in series[s]])],
                band=(r["baseline_mean"] - r["noise_floor"], r["baseline_mean"] + r["noise_floor"]),
                dots=r["replicate_means"],
            )
        )
    vmax = max(max(p["violations"] for p in series[s]) for s in SPLITS) + 1
    panels.append(
        _panel(2 * w, 0, w, h, "Rule violations (no LLM), by iteration", (0, vmax), xs,
               [(s, _COLORS[s], [p["violations"] for p in series[s]]) for s in SPLITS], fmt="{:.0f}")
    )
    legend = (
        f'<text x="20" y="{h + 22}" font-size="11" fill="#374151">'
        f'<tspan fill="{_COLORS["train"]}">&#9679; train (10 prompts)</tspan>   '
        f'<tspan dx="16" fill="{_COLORS["heldout"]}">&#9679; held-out (5 prompts)</tspan>'
        f'<tspan dx="16" fill="#6b7280">&#9675; no-lessons replicate runs</tspan>'
        f'<tspan dx="16" fill="#6b7280">grey band = baseline &#177; noise floor</tspan></text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {3 * w} {h + 34}" width="100%" '
        f'font-family="-apple-system, Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#ffffff"/>'
        + "\n".join(panels) + legend + "</svg>"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def analyze_run(tag: str, iterations: int, replicate_names: list[str], runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    loop_scores = [_load(f"iteration_{n}_{tag}", runs_dir) for n in range(iterations)]
    series = {s: [summarize(sc)[s] for sc in loop_scores] for s in SPLITS}
    # the loop's own iteration 0 is itself a no-lessons run, so it counts as a replicate
    replicates = [summarize(sc) for sc in [loop_scores[0]] + [_load(n, runs_dir) for n in replicate_names]]
    result = verdict(replicates, summarize(loop_scores[-1]))
    result["dimension_means"] = {
        s: {"iteration_0": split_dimension_means(loop_scores[0], s), "final": split_dimension_means(loop_scores[-1], s)} for s in SPLITS
    }
    return {"series": series, "verdict": result}


def format_report(a: dict[str, Any]) -> str:
    v, s = a["verdict"], a["series"]
    lines = [f"{'iter':>4s} {'train':>14s} {'held-out':>14s}   (mean overall / rule violations)"]
    for i in range(len(s["train"])):
        c = lambda k: f"{s[k][i]['mean']:.2f} / {s[k][i]['violations']:.0f}"
        lines.append(f"{i:>4d} {c('train'):>14s} {c('heldout'):>14s}")
    lines.append(f"\nno-lessons replicates: {v['n_replicates']}")
    for k in SPLITS:
        r = v[k]
        lines.append(
            f"{k:>8s}: baseline {r['baseline_mean']:.3f} (replicates {', '.join(f'{m:.2f}' for m in r['replicate_means'])}), "
            f"noise floor {r['noise_floor']:.3f}, final {r['final_mean']:.3f}, gain {r['gain']:+.3f} "
            f"-> {'BEATS' if r['beats_noise_floor'] else 'does not beat'} noise floor; "
            f"violations {r['baseline_violations']:.1f} -> {r['final_violations']}"
        )
    lines.append(f"\nREAL IMPROVEMENT (pre-registered criterion): {'YES' if v['real_improvement'] else 'NO'}")
    if v["overfitting_flag"]:
        lines.append("OVERFITTING FLAG: train gain beats its noise floor but held-out does not.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Apply the pre-registered criterion to a finished loop run.")
    p.add_argument("--tag", required=True)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--replicates", nargs="+", required=True, help="Scored no-lessons run names, e.g. iteration_0_dry")
    args = p.parse_args()

    a = analyze_run(args.tag, args.iterations, args.replicates)
    print(format_report(a))
    (RUNS_DIR / f"trend_{args.tag}.json").write_text(json.dumps(a, indent=2), encoding="utf-8")
    (RUNS_DIR / f"trend_{args.tag}.svg").write_text(render_svg(a["series"], a["verdict"]), encoding="utf-8")
    print(f"\nSaved: trend_{args.tag}.json, trend_{args.tag}.svg")


if __name__ == "__main__":
    main()
