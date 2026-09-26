# Self-Improvement Loop

A genuine in-context self-improvement loop built on top of the fitness
coaching agent: a fixed eval set, an LLM-judge scorer, lesson extraction
from low-scoring responses, and re-injection of those lessons into the
system prompt on the next pass — with a logged score trend reported
honestly, whether it improves, plateaus, or regresses.

**Start with [WRITEUP.md](WRITEUP.md)** — the caveat-first summary. This
file is the chronological engineering log behind it.

Not fine-tuning. Not a new domain from scratch — this reuses the fitness
agent's actual tools, models, and validation logic (`app/`) as the system
under test, because the eval-design work is the point, not re-proving
tool-calling works.

## Status

- [x] Day 1 — Eval harness: fixed 10-prompt set, isolated per-iteration DB
      fixtures, a trace-capturing loop, and a runner producing
      reproducible iteration files. First run (now `iteration_0_opus.json`) executed
      and verified against the live API.
- [x] Day 2 — Rubric + LLM judge, self-consistency check (passed), and a
      headroom problem found and diagnosed (Opus baseline is at the
      ceiling; see "Day 2 results"). Decided: Haiku 4.5 is the agent under
      test for the loop.
- [x] Day 3 — Lesson extraction, re-injection, held-out split, leakage
      lint, and a resumable loop orchestrator: built, tested (102 tests),
      and pipeline-verified end to end on a dry run. The dry run's second
      judging pass was cut short by exhausted API credits (see "Day 3
      results"); nothing about the official result depends on it.
- [x] Day 4 — Full 5-iteration run with 4 no-lessons replicates for the
      noise floor; trend plotted; pre-registered criterion applied
      mechanically (result: YES, with real caveats — see "Day 4 results").
      One robustness check (a pairwise negative control) is unfinished
      because API credits ran out again.
- [x] Day 5 — Second independent loop run (reproduced by the pre-registered
      criterion, but improving different things), pairwise negative control
      (passed), and the writeup (`WRITEUP.md`).

## Design decisions (Day 1)

**Isolated harness, `app/agent.py` untouched.** The eval loop
(`eval/eval_loop.py`) does not call `app.agent.run_agent_loop` — that
function's `trace` param only records tool *names*, and it always builds
its own system prompt internally with no injection point. Rather than
edit a working, already-deployed production file to add an experimental
capability, `eval/eval_loop.py` imports `app/agent.py`'s actual
validation/dispatch logic (`_execute_client_tool`, `_validate_plan`,
`_force_finalize`, `TOOLS`) — the part that has to stay correct — and
wraps its own thin loop around it, capturing full tool-call records
(name, input, result, error) and taking `system_prompt` as an explicit
parameter. This is the same "share validation/dispatch, not control
flow" pattern `app/agent.py` already uses between `run_agent_loop` and
`stream_agent_events` (see the main README's Day 3 log) — a third
sibling loop for a third use case, not a fork of an existing one.
Deployed `/chat` and `/chat/stream` are exactly as they were before this
project started.

**Ten fixed prompts, each targeting a specific failure mode** — not just
"does it produce *a* plan." Two baseline (no-history) prompts, two that
require correctly reading and using seeded history (including one where
the *right* answer requires referencing specific historical numbers), an
informal-name-resolution case ("RDLs" → "Romanian Deadlift"), a matched
pair testing `log_set` correctness in *both* directions (should log /
should not log — over-eager logging is the failure mode a naive agent
hits first), a safety/pushback case (daily max-effort deadlifts, no rest
days), a deliberately vague request, and a case where history exists but
doesn't cover what's being asked (upper-body history, leg-day request —
tests whether it notices the gap instead of hallucinating leg numbers
from bench data). Full list and rationale in `eval/prompts.py`.

**Fixtures re-seeded fresh every run, never accumulated.**
`eval/db.py::setup_eval_db` builds a brand-new temp SQLite file and
inserts every prompt's `seed_sets` under that prompt's own `eval_`
user_id before any prompt runs. Without this, iteration 2's "returning
user" scenario would include whatever iteration 1 happened to log,
confounding any score comparison across iterations with drifting input
data instead of isolating the one thing that's supposed to change
(the system prompt). Locked in by `tests/test_eval_db.py`.

**Known gap, accepted for now:** per-turn token accounting in
`EvalTurnResult` doesn't include tokens spent inside `_force_finalize`'s
forced-retry calls, since that helper doesn't expose its own usage.
Undercounts the rare turn that hits the forced-finalize path. Not worth
changing `app/agent.py`'s return type to fix, given this number is a
rough per-iteration cost figure for the eventual writeup, not a billing
record.

## Pre-registered thresholds (Day 2, written before the first judge call)

The rubric (`eval/rubric.py`) has four dimensions scored 1-5:
tool-use correctness, safety/appropriateness, personalization,
communication quality. The judge is `claude-opus-5` (same model as the
agent — see Day 2 notes for why that's a known limitation), sees the
prompt, ground-truth history, tool trace, and final plan, and must write
evidence *before* each score.

Judge reliability is tested by scoring the same saved outputs 3
independent times. A "cell" is one (output, dimension) pair.

- **Reliability (kill condition):** usable only if at most 10% of cells
  have a spread of 2+ points across repeats, AND at least 60% of cells
  agree exactly. Otherwise: pivot to a rule-based/hybrid rubric.
- **Headroom (flag, not a kill condition):** if the baseline mean
  overall score is above 4.5, there's under half a point of room for the
  loop to show improvement, and the rubric must be tightened before
  iterating on lessons means anything.

These numbers are fixed in `eval/rubric.py` and were chosen before
seeing any judge output.

## Pre-registered for the loop (Day 3, written before any lesson was extracted)

**The overfitting problem.** Lessons are extracted from failures on the
eval prompts, and then the eval prompts are re-scored. A score gain could
be the agent genuinely improving, or lessons that memorized the test set.
Two guards, fixed in advance:

1. *Held-out split.* 10 `train` prompts (the extractor may learn from
   their failures) + 5 `heldout` prompts (same failure categories,
   different surface details; the extractor never sees them). Both splits
   are scored every iteration. Train up + held-out flat = overfitting, and
   will be reported as such.
2. *Leakage lint.* Every extracted lesson is rejected if it names any
   exercise from the library or reproduces a 5-word run from a training
   prompt. Rejected lessons are logged, never silently kept.

**Loop mechanics.** 5 scored iterations (0-4); lessons extracted after
iterations 0-3. The extractor sees only `train` outputs with mean overall
score below **3.5**, with the tool trace, plan, rule violations, and the
judge's evidence (not the eval's "what this tests" descriptions). It may
add at most **3** lessons per iteration, **12** total, and must avoid
duplicating existing lessons. Lessons are appended to the *base* agent
system prompt in a fixed wording; the judge keeps grading against the base
spec only. Judge, rubric, and prompts are frozen for the whole run.

**Noise floor.** The agent is stochastic too, and Day 2 only measured
*judge* noise. Before claiming anything, at least 3 no-lessons replicate
runs on all 15 prompts; the noise floor for each split is the largest
difference in mean overall score between any two replicates.

**What counts as a real improvement (decided now):** the final
iteration's *held-out* mean overall score exceeds the no-lessons replicate
mean by more than the noise floor, AND held-out rule violations are not
higher than baseline. Anything less is reported as no demonstrated
improvement, whatever the train numbers say.

**Change policy.** After the Day 3 dry run, only bug fixes are allowed to
the extractor, lint, and injection. Changes made to raise the score are
not; any change at all after the dry run is logged here with its reason.

## Day 2 results

**Reliability: passed, twice.** On the Opus baseline (10 outputs x 3
independent judge calls, 40 cells): 0% large disagreements, 85% exact
agreement, 0 schema failures. Because that baseline was nearly all 4s and
5s (easy mode for consistency), I re-ran the same pre-registered analysis
on mid-scale scores from a weaker agent (Haiku 4.5): 0% large
disagreements, 80% exact, judge noise (within-cell sd 0.09) roughly 9x
smaller than the real spread between outputs (between-cell sd 0.78). The
kill condition never triggered, so no pivot to a rule-based rubric was
needed.

**Headroom: flagged, and it's the real finding.** The Opus 5 baseline
scored a mean of **4.84** (101 of 120 individual scores were 5, none below
4) — inside the pre-registered "no room to improve" zone (>4.5). A score
trend from 4.84 to 4.9 would prove nothing. The judge was *reliable* but
the setup wasn't *informative*; those are different properties and only
the first was in the kill condition. An early hint: on its very first
call the judge named a "minor inefficiency" that the rubric's own 4-anchor
describes, then scored 5 anyway.

**Diagnostic:** same 10 prompts, same judge, Haiku 4.5 as the agent:
mean **3.06** (range 2.5-3.9 per prompt, scores spanning 2-5). Haiku skips
`look_up_exercise` on most prompts, made zero tool calls on
`unsafe_request` (never checked history), and its one `log_set` on
`log_a_completed_set` errored and it gave up (0 of 3 sets logged). Real
failures a loop can learn from. The judge separating a strong agent from a
weak one by ~1.8 points is also weak evidence it isn't rubber-stamping.

**Validity (tool-use dimension only).** Consistency isn't validity — a
judge can be reliably wrong. Tool use is the one dimension with objective
ground truth, so `eval/rules.py` computes deterministic violations from the
trace (history never checked, plan exercises never verified, wrong number
of `log_set`s, tool errors) with no LLM involved. Over all 20 outputs
(Opus + Haiku): outputs with no violations averaged a judge
`tool_use_correctness` of 4.56; outputs with one or more averaged 2.19
(Pearson r = -0.80). **This does not validate the other three
dimensions** — safety, personalization, and communication have no
objective check here. They rest on the judge's self-consistency and the
Opus-vs-Haiku separation, which is weaker.

The rule check is kept as a permanent second instrument: on Day 4, an
improvement that shows up in both the judge score *and* the rule-violation
count is much stronger evidence than either alone.

**Two schema failures on the way, both caught by Pydantic, neither
silent.** The judge's first schema nested one object per dimension. Attempt
1: the model returned a single flat `{evidence, score}` object (it read the
shared `$ref` as *the* input type). Attempt 2, after inlining the refs: it
put raw `<parameter ...>` markup inside a string value. The fix was a flat
wire format (`<dimension>_evidence`, `<dimension>_score`, evidence first),
regrouped into a nested model after validation. Worth remembering: a
forced tool call is only as reliable as its schema shape, and the
validation boundary is what turned "the judge is quietly returning
garbage" into an error I could read. Locked in by tests.

**Other things worth knowing:**
- Current Claude models reject `temperature`, so judge variance can't be
  pinned — it's measured, not controlled.
- A forced tool call skips extended thinking, so the judge's reasoning is
  the evidence text it must write *before* each score. Confirmed live that
  forced `tool_choice` works on `claude-opus-5`; that incidentally means
  the agent's own `_force_finalize` safety net is valid against the real
  API, not just fakes.
- The judge grades against the *base* agent instructions, not whatever
  lessons-augmented prompt a later iteration uses, so every iteration is
  scored against the same fixed spec.
- `iteration_0_opus.json` (formerly `iteration_0.json`) predates the
  `agent_model` field; it was produced by the default `claude-opus-5`.

**Known limitations, not fixed:** same-family judge (Opus 5 judging Opus 5
is subject to self-preference; judging Haiku outputs with Opus is cleaner);
no human-labeled scores; only 10 prompts and 3 repeats; validity only
checked on one of four dimensions.

**Decision: Haiku 4.5 is the agent under test for the loop** (chosen
after seeing the numbers above). It's the only setup with real headroom
(3.06), it's cheaper per iteration, and it makes the judge cross-model
(Opus judging Haiku, not itself). The cost is a reframing that the final
writeup has to state plainly: **the loop improves a Haiku-based fitness
agent, not the deployed Opus one.** Sonnet 5 was never tested and might
sit in a useful middle.

Enforced in code, not just prose (`eval/config.py`): the eval model is
independent of the app's `FITNESS_AGENT_MODEL`, and is pinned onto
`app.agent.DEFAULT_MODEL` at run time (never at import, so tests can't be
silently retargeted). The pin matters because `_force_finalize` reads that
module global — without it, a "Haiku" run's forced-finalize calls would
quietly go to the deployed default (Opus), mixing two models in one
result. A recording-client test checks every request in a turn carries the
same model. Saved runs are also never overwritten (`FileExistsError`), and
every run file records the `agent_model` it used.

**Files:** the Haiku loop baseline is `iteration_0_haiku.json` (scores in
`scores_iteration_0_haiku.json`). `iteration_0_opus.json` is the Opus
diagnostic (renamed from `iteration_0.json` so an untagged Haiku run can't
be confused with it; its `agent_model`/`tag` fields were backfilled by
hand and say so).

**Day 2 cost** (judge calls unless noted): Opus consistency 131k in / 29k
out; Haiku scoring 106k in / 28k out; plus one Haiku agent run (10 prompts).

## Day 3 results

**Built:** held-out split (`eval/prompts.py`, 10 train + 5 held-out),
lesson extractor + leakage lint + append-only store (`eval/lessons.py`),
resumable orchestrator (`eval/loop.py`: run -> score -> extract ->
frozen lesson snapshot per iteration), per-split scoring. 102 tests pass
with the live-API test excluded (it needs credits; see below).

**Dry run (tag `dry`, pipeline check only — not evidence).** Iteration 0
(no lessons) ran and was judged. Extraction saw 7 train failures, made 1
call, and produced 3 lessons with 0 rejected by lint — all general, none
naming an exercise:
1. Look up *every* exercise before listing it, use the canonical name,
   never repeat an identical failing call.
2. State the evidence basis plainly in the summary when history is empty
   or irrelevant.
3. Keep intensity conservative when history is unknown or the user reports
   detraining, and make the summary match the listed exercises.

They were frozen into `lessons_iteration_1_dry.md`, injected, and the
agent's iteration-1 behavior visibly changed (more lookups, more explicit
"no history found" summaries). Deterministic rule violations went from 6
to 3 on train and 2 to 1 on held-out. **That is n=1 on a dry run and I'm
not reading it as a result** — but it confirms the mechanism reaches the
agent. Two violations survived: `never_checked_history` on both unsafe
prompts (the agent made zero tool calls on each). None of the three
lessons addressed that pattern, which is a fair preview of a limit:
lessons only cover what the extractor chose to write down.

**What the lessons actually are.** Lesson 1 is mostly a *sharpened
restatement of an instruction the agent already had* ("call
`look_up_exercise` if you're not certain" became "for EVERY exercise").
That's a legitimate thing for this loop to do, but it means it mainly
converts under-followed instructions into concrete ones, and the rule
metric and the lessons share a source (both derive from the agent's own
spec), so they are not fully independent evidence. The held-out split,
not the rule count, is the real generalization check.

**Run-to-run noise (first data point).** Dry iteration 0 is a second
no-lessons run of the same agent on the same 10 train prompts as the Day 2
baseline. Means: 3.058 vs 3.050 (diff 0.008). But per-prompt scores
differed by 0.32 on average and up to 0.75 (`ambiguous_exercise_name`
3.17 vs 2.42). The aggregates agreeing is mostly cancellation across ten
prompts, so a tiny gap between two runs should not be trusted as a
detection limit — this is why the pre-registered noise floor requires at
least 3 replicates. Judge noise (Day 2) was ~0.09; agent sampling noise is
several times larger.

**Bugs the dry run and preview caught** (all in the tooling, none in
`app/`):
- *Array-valued tool params fail on forced calls* — the extractor returned
  its `lessons` array as a string of raw `<parameter ...>` markup. Third
  time the same failure class has appeared (judge: nested objects, twice).
  Rule now: forced tool calls use flat scalar fields only. Extraction uses
  `lesson_1..3`.
- *The model ignored `maxItems` and `maxLength` in the schema* and
  returned 4 lessons of 250-360 chars against limits of 3 and 300.
- *My retry re-sent the identical request*, so a validation failure was
  repeated verbatim three times. Retries now feed the specific validation
  error back (the same self-correction pattern the agent uses).
- *The model can't count characters:* it exceeded a 300, then a 400,
  character cap, even with the error fed back. Brevity is now requested
  in words ("under 50 words") with a 700-character backstop that only
  bounds prompt growth.

**Change log under the pre-registered change policy.** After the dry run
began, exactly one change was made: `MAX_LESSON_CHARS` 400 -> 700 plus the
"under 50 words" line in the extractor prompt. Reason: the extractor could
not produce valid output, which crashed the pipeline. It was not made to
raise scores (no iteration-1 score had been seen — judging hadn't
completed). Earlier changes (flat schema, error-feedback retry) were made
during pre-run testing, before any loop iteration existed.

**Operational finding:** the run stopped mid-way with `Your credit balance
is too low to access the Anthropic API` (HTTP 400, not a rate limit). The
loop is resumable, so it needs only credits, then
`python -m eval.loop --tag dry --iterations 2` finishes the one missing
step (judging `iteration_1_dry`; its agent run is saved). I'm not
finishing it: the pipeline check is done, and the credits are better spent
on the official run.

**Also:** a correction to my own Day 2 note — the Haiku baseline has 9
rule-violation entries across 7 prompts, not 8.

**Estimate for Day 4** (list-price arithmetic from measured token counts,
~157k in / ~42k out per judged run, so roughly $2 each): 5 official
iterations + 2 extra no-lessons replicates = 7 judged runs, plus Haiku
agent runs and 4 extractions. On the order of **$15**, mostly the Opus
judge.

## Day 4 run log

*Written as things happened, before the final numbers.*

- **Baseline replicates (no lessons, all 15 prompts):** dry iter 0, rep1,
  rep2, plus the loop's own iteration 0 = 4 runs (pre-registered minimum
  was 3).
- **Crash in the official loop, iteration 2's agent run:** `AgentError:
  Claude could not produce a valid workout plan after forced retries`.
  Iterations 0-1 were complete and scored; the crashed attempt saved
  nothing. **Disclosure: while diagnosing it I saw iteration 1's scores
  (train 4.11, held-out 4.20 vs. baselines near 3.15 / 3.55).** The success
  criterion is mechanical and nothing about the extractor, lint, or
  injection was changed because of that.
- **Likely cause, and it's an effect of the loop itself:** the plan schema
  caps `summary` at 500 characters. Mean summary length went 289 -> 386
  chars from iteration 0 to 1 (one summary at 492/500), because the
  lessons ask for more explicit summaries. I can't prove the crashed turn
  was a length failure (its output died with the crash).
- **Change under the change policy (a bug fix, and the unflattering
  direction):** a turn that never produces a valid plan used to crash the
  whole iteration, destroying the other 14 prompts' results. It is now a
  recorded result (`failed: true` + the validation error), scored **1 on
  every dimension** and counted as a `no_valid_plan` rule violation; it
  is never skipped and never sent to the judge. Only the agent's own
  `AgentError` is absorbed; API errors still propagate. No earlier run had
  a failed turn, so no earlier score changes. Failed turns also feed the
  lesson extractor as failures to learn from.
- **Credits ran out a second time**, during the exploratory pairwise check
  (below). That run's negative control did not complete.
- **Disclosure: iteration 2 was re-drawn after the crash.** The crashed
  attempt saved no data, and under the new policy a repeat of that failure
  would now show up in the data as a 1 rather than being retried away.

## Day 4 results

![Score and rule-violation trend](runs/trend_loop.svg)

*Left/middle: mean judge score by iteration; hollow dots are the four
no-lessons runs, the grey band is baseline +/- the noise floor. Right:
deterministic rule violations (no LLM).*

**Setup:** Haiku 4.5 as the agent, Opus 5 as judge (3 repeats per output),
15 prompts (10 train + 5 held-out), 5 iterations, lessons extracted after
each iteration from train failures below 3.5.

| iter | train score / violations | held-out score / violations | lessons active |
|---|---|---|---|
| 0 | 3.14 / 6 | 3.28 / 2 | 0 |
| 1 | 4.11 / 3 | 4.20 / 1 | 3 |
| 2 | 4.42 / 1 | 4.45 / 0 | 6 |
| 3 | 4.08 / 3 | 4.35 / 0 | 6 |
| 4 | 4.31 / 1 | 4.25 / 2 | 9 |

**Pre-registered verdict (mechanical, `eval/trend.py`): REAL IMPROVEMENT =
YES.** No-lessons baseline (4 runs): train 3.158, held-out 3.487. Noise
floor (largest gap between any two of those runs): train 0.183, held-out
0.300. Final iteration: train +1.150, held-out +0.763 — both beat their
floors. Held-out rule violations 2.0 -> 2: not higher, so the second
clause is met, **but only by equality**. The overfitting flag did not fire.

**How I'd actually describe this (and what I wouldn't claim):**

- *The gain is real relative to the noise I measured, is large, and
  appears on held-out prompts the extractor never saw.* Every dimension
  rose, on both splits.
- *It is a step, not a trend.* Nearly all of it arrives after the first
  three lessons (iteration 1). After that the scores plateau and wobble
  (train 4.42 -> 4.08 -> 4.31); held-out at the final iteration is below
  its iteration-2 peak. More lessons did not reliably mean more improvement.
- *The loop stopped learning by its own rule.* After iteration 2 there were
  zero train outputs below the 3.5 threshold, so no extraction ran; there
  are 9 lessons, not the 12 allowed. 0 lessons were rejected by the
  leakage lint at any point.
- *The objective instrument confirms train but not held-out at the
  endpoint.* Train violations 6.2 -> 1. Held-out went 2 -> 1 -> 0 -> 0 -> 2:
  better in the middle, back to baseline at the end (two unverified plan
  exercises on two held-out prompts). With 5 held-out prompts and 0-2
  violations, this instrument can't say much either way.
- *The judge stayed reliable throughout.* At every scored run: 0% large
  disagreements, exact agreement 60-77% (`iteration_0_loop` sat exactly on
  the 60% minimum).
- *I would not claim* that this generalizes beyond these 15 prompts or this
  agent (Opus had no headroom to test on), that it is "better coaching"
  rather than "follows the rubric and its own instructions better" (the
  extractor learns from judge evidence, and lesson 1 is largely a sharpened
  restatement of an instruction the agent already had), or that a second
  run of the loop would produce the same trajectory (n = 1 loop run; the
  noise floor measures no-lessons variance, not loop-to-loop variance).

**Robustness checks I ran on my own result** (all post-hoc; only the
verdict above was pre-registered):

- *Verbosity bias.* The judge has a real length association with no
  lessons at all (r = +0.32 across 60 no-lessons outputs, +0.07 to +0.50
  per run), and lessons did lengthen summaries (~289 -> ~380 chars). But at
  the measured slope, length alone could explain at most ~0.18 points of
  a 0.76-1.15 gain, even if fully causal.
- *A blind pairwise comparison with no rubric* (`eval/pairwise.py`): for
  each prompt, the final-iteration output vs each of the 4 no-lessons
  outputs, randomized and position-balanced, judge told not to prefer
  length. **57 wins, 0 losses, 3 ties of 60 (97.5%; 95% CI over prompts
  94-100%).** Same on both splits; no position bias (95% / 100%); 100%
  even in the 12 comparisons where the final summary was *shorter*. It
  also agrees with the rubric at the level of individual comparisons (the
  rubric scored the final output higher in 59/60, baseline higher in 0/60),
  and its 3 ties were on prompts where both responses did essentially the
  same thing. Its stated reasons were overwhelmingly about tool
  verification and use of history (50/60 and 60/60 analyses), rarely
  length or detail (6/60).
- **What that check does NOT show, and why I distrust a 97.5%:** it is
  Opus judging the same outputs the rubric judge saw, so the two
  instruments' errors are correlated; a result this lopsided should be
  treated skeptically until a negative control (two no-lessons runs
  compared against each other, expected ~50%) shows the method doesn't
  favor one group by construction. **I started that control and credits
  ran out; it is not done.** Until it is, read the pairwise result as
  "consistent with the rubric," not as independent confirmation.

**Bugs and mistakes in this stage** (tooling only; `app/` untouched):
- Repeated a mistake I'd already documented: a hard character cap on a
  free-text field in the pairwise judge, which the model can't count. It
  also used `asyncio.gather` without `return_exceptions`, so one bad call
  discarded ~59 completed, paid-for calls. Both fixed (word guidance +
  backstop; failures tallied; API/credit errors still propagate).
- Chart shipped with a hard-coded pixel width (cropped at narrow widths)
  and collapsed legend spacing; caught only by rendering it in a browser,
  not by the well-formedness test. Fixed and regenerated.
- A test for the pairwise prompt's blindness failed because the word
  "final" appeared in a plan heading; I removed the word from the prompt
  rather than loosen the test.
- One failure mode is still live: plan summaries reached 498 and 499 of the
  500-character schema limit in iterations 2 and 4. The crashed attempt
  shows it can trip. The reported numbers contain no failure penalty
  because the re-drawn iterations had none.

**Measured cost:** judge scoring (7 runs) 1.17M in / 0.30M out; pairwise
(60 comparisons) 0.16M / 0.02M; extraction 0.02M / 0.002M; Haiku agent turns
0.84M / 0.09M. About $14.8 of Opus at list price plus ~$1.3 of Haiku,
*excluding* the crashed attempt, the discarded first pairwise batch, and
the earlier dry run.

**Still open:** the pairwise negative control (needs ~45 calls); a second,
independent run of the whole loop; any human-labeled check; a non-Haiku
agent with headroom.

## Day 5 pre-registration (written before either run below was launched)

**1. Pairwise negative control.** The blind pairwise judge showed a 97.5%
win rate for the final iteration, which is suspiciously clean. Control:
run the *identical* procedure with a no-lessons run (`iteration_0_rep1`)
in the "final" role against the other three no-lessons runs (15 prompts x 3
baselines = 45 comparisons). All four come from the same distribution, so
an unbiased judge should land near 50%. **Pass** = win rate within
35-65% AND the 95% CI over prompts contains 50%. **Fail** = the pairwise
method is treated as biased and the 97.5% result is discounted entirely
(reported, but not used as evidence).

**2. Independent replication of the whole loop.** A second full 5-iteration
loop (`--tag loop2`), fresh lessons store, same code and config as the
first: nothing is tuned between the two runs; only bug fixes are allowed
and any are logged. Its noise-floor replicates are its own no-lessons
iteration 0 plus the four existing no-lessons runs (5 total). Same
mechanical criterion as before (`eval/trend.py`). Both loops are reported
side by side whatever happens. **If loop 2 does not meet the criterion, the
first result is described as "not reproduced,"** not as the "real" run.
Credit cutoffs are handled by resuming the loop, never by re-drawing an
iteration.

## Day 5 results

**Pairwise negative control: PASSED (pre-registered criterion).** A
no-lessons run (`iteration_0_rep1`) in the "final" role against the other
three no-lessons runs, 45 comparisons: 21 wins, 17 losses, 7 ties = **54.4%
win rate, 95% CI over prompts 37.8-70.0%** (needed: within 35-65% and CI
containing 50%). For contrast, the real final iteration went 57 wins / 0
losses / 3 ties (CI 94-100%) — the two intervals don't overlap. So the
blind judge is not simply favoring whichever group sits in the "final"
slot: given equivalent outputs it splits roughly evenly and ties ~15% of
the time. Two small artifacts, reported rather than smoothed over: a mild
position skew (final-as-A 58% vs final-as-B 47%, n = 45) and a mild length
lean (a longer "final" summary won 62.5% of 16 vs 50% of 29). Neither is
close to large enough to produce 57-0. The pairwise result stays
"consistent with the rubric, and not an artifact of the procedure," not
"independent proof": both instruments are still Opus judging the same
outputs, and a bias toward the *style* the lessons induce (thorough
verification, explicit statements about missing history) is not something
this control can detect.

**Loop 2 (independent replication, `--tag loop2`, identical code and
config): REAL IMPROVEMENT = YES** by the same mechanical criterion, with 5
no-lessons runs for the noise floor (its own iteration 0 + the four
existing). Train 3.187 -> 4.083 (+0.897; floor 0.250); held-out 3.503 ->
4.333 (+0.830; floor 0.300); held-out violations 2.0 -> 1. So the loop-1
result is **reproduced** under the rule I fixed beforehand.

| iter | train score / violations | held-out score / violations |
|---|---|---|
| 0 | 3.30 / 5 | 3.57 / 2 |
| 1 | 3.87 / 4 | 3.97 / 2 |
| 2 | 3.93 / 7 | 4.18 / 2 |
| 3 | 4.11 / 3 | 3.90 / 1 |
| 4 | 4.08 / 5 | 4.33 / 1 |

**But the two loops improved different things, and that matters more than
the shared YES.** Train `tool_use_correctness` (the one dimension with an
objective check): loop 1 3.13 -> 4.53, loop 2 3.37 -> 3.67. Train
violations: loop 1 6.2 -> 1, loop 2 6.0 -> 5 (peaking at 7 at iteration 2).
Loop 2's gains are in personalization (+1.20), communication (+0.93), and
safety (+0.70), where there is no objective check. Loop 2 even wrote a very
explicit verification lesson (L7: never call `emit_plan` until every name
has been returned by a lookup) and Haiku still left 3 unverified exercises
at iteration 4, where loop 1's similar lesson worked. So: the judge-score
improvement replicated; the objective corroboration did not. Lessons also
differed in content: loop 2 spent them on plan completeness, recency
claims, honoring scope constraints, and summary/tool-output consistency.
Both loops made 9 lessons with 0 rejected by lint, and both stopped
extracting once all train outputs cleared 3.5 (after iteration 2 in loop 1,
after iteration 3 in loop 2). No failed turns, no judge failures in either.

**Judge validity at scale.** Across all 150 loop outputs (both loops, all
iterations), the judge's tool-use score vs. the deterministic violation
count: Pearson r = -0.84 (Day 2, n = 20: -0.80); outputs with no violations
averaged 4.54 (n = 102), outputs with at least one 2.51 (n = 48). Judge
self-consistency held at every scored run: 0% large disagreements across
all 12 runs, exact agreement 60-77%. Length bound (same method as Day 4):
final summaries ~84 chars (loop 1) / ~101 chars (loop 2) longer than
no-lessons, which at the measured slope explains at most ~0.18 / ~0.22
points.

**Pairwise on loop 2:** 53 wins / 5 losses / 2 ties = **90.0%** (95% CI over
prompts 75.8-99.2%), vs. 97.5% for loop 1. Held-out 92.5%, train 88.8%;
position 86.7% / 93.3%. Length lean is stronger than in the control: final
summary longer 93.5% (n = 46) vs. 78.6% (n = 14). The judge picking a
baseline five times, and scoring the two loops differently, is the evidence
that it is discriminating rather than rubber-stamping.

**Day 5 measured cost:** Opus judge 0.86M in / 0.22M out, pairwise (2 runs)
0.26M / 0.04M, extraction 0.018M / 0.002M ~= $12.2 at list; Haiku agent turns
0.62M / 0.06M ~= $0.9. Nothing discarded this time.

The consolidated, caveat-first version of all of this is in
[WRITEUP.md](WRITEUP.md).

## Baseline (`iteration_0_opus.json`, Day 1 smoke data)

10/10 prompts completed against the live API with no `AgentError`s.
Spot-checked the two prompts most likely to reveal a harness bug:
- `log_a_completed_set` ("3 sets of 10 Pull-Ups") correctly produced
  three separate `log_set` calls (`Pull-Up`, 0 lbs, 10 reps each) — the
  schema logs individual sets, not "3 sets" as one row, and the agent
  matched that.
- `should_not_log` (a stated plan, nothing completed) correctly called
  zero `log_set`s.

Total baseline cost: 107,053 input / 17,595 output tokens across all 10
prompts.

## Running it

```bash
python -m eval.runner --iteration 1 --tag haiku                        # agent = Haiku 4.5 (eval/config.py)
python -m eval.runner --iteration 2 --tag haiku --lessons-file eval/lessons.md   # Day 3+
python -m eval.score --run iteration_0_haiku          # judge x3 + rule checks
python -m eval.consistency --run iteration_0_haiku    # judge self-consistency
python -m eval.loop --tag loop --iterations 5         # the whole loop (resumable)
python -m eval.trend --tag loop --iterations 5 --replicates iteration_0_dry iteration_0_rep1 iteration_0_rep2
python -m eval.pairwise --final iteration_4_loop --baselines iteration_0_loop iteration_0_dry iteration_0_rep1 iteration_0_rep2  # exploratory
```

Requires `ANTHROPIC_API_KEY` (loaded from `.env` the same way the main
app does). Output: `eval/runs/iteration_N[_tag].json`; existing runs are never overwritten.
