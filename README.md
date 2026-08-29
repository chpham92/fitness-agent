# Fitness Coaching Agent

A tool-calling LLM agent (raw Anthropic Messages API, no framework) that
answers coaching questions by calling tools against a real SQLite
database — exercise lookup, set logging, and history retrieval — then
returns a Pydantic-validated structured workout plan.

Built to demonstrate: async Python, tool-calling with schema validation
at the argument boundary, structured output enforcement, and SSE
streaming, deployed as a live service rather than a notebook.

## Live demo

**https://fitness-coaching-agent.fly.dev** — deployed on Fly.io, scale-to-zero
(the first request after idle takes a few extra seconds to cold-start).

```bash
curl https://fitness-coaching-agent.fly.dev/health
```

```bash
curl -X POST https://fitness-coaching-agent.fly.dev/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "demo", "message": "Give me a quick upper body plan for today."}'
```

```bash
# -N disables curl's output buffering — required to actually see events
# arrive incrementally instead of all at once when they're done.
curl -N -X POST https://fitness-coaching-agent.fly.dev/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id": "demo", "message": "Give me a quick upper body plan for today."}'
```

## Status

- [x] Day 1 — SQLite schema, seed data, Pydantic models, tool functions + unit tests (17/17 passing)
- [x] Day 2 — Raw Claude tool-use loop, `/chat` endpoint
- [x] Day 3 — `/chat/stream` (SSE), tool-error handling
- [x] Day 4 — Dockerfile + fly.toml + DEPLOY.md; verified locally (non-root process, volume persistence, in-container streaming) *and* deployed — live at the URL below, with `/health`, `/chat`, and `/chat/stream` all confirmed working in production
- [x] Day 5 — Edge-case tests (empty input, end-turn-without-emit_plan, duplicate sets), a naming-clarity fix, and this writeup

## Architecture

**Why raw tool-calling, not LangChain/LangGraph:** the agent loop here
is simple enough (3 tools, single conversation, no branching state
machine) that a framework would be overhead, not leverage. The Anthropic
Messages API tool-use loop is ~40 lines. Framework fluency is a separate
skill from understanding what a framework is abstracting — this project
demonstrates the latter directly.

**Why SQLAlchemy Core, not the ORM:** two tables, no relationships worth
mapping to Python objects. Core gives real async engine/connection
semantics without ORM ceremony.

**Validation boundary:** every tool argument the LLM produces is parsed
into a Pydantic model *before* it touches the database. Malformed or
hallucinated arguments (negative weight, nonexistent exercise) are
rejected at that boundary, not caught downstream.

## Tools

| Tool | Purpose |
|---|---|
| `look_up_exercise` | Search the exercise library by name or muscle group |
| `log_set` | Record a completed set (validates the exercise exists first) |
| `get_recent_history` | Retrieve a user's recent logged sets |
| `emit_plan` | Terminal tool — the only way the agent delivers its final answer |

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `POST /chat` | `{user_id, message}` → runs one full agent turn → validated `WorkoutPlanResponse` JSON |
| `POST /chat/stream` | Same turn, as Server-Sent Events — see event schema below |

### SSE event schema (`/chat/stream`)

Each event is a standard `event: <type>\ndata: <json>\n\n` line pair. Types:

| Event | `data` shape | When |
|---|---|---|
| `text` | `{"delta": str}` | Each incremental chunk of Claude's own text, as it streams in |
| `tool_call` | `{"id", "name", "input"}` | A tool call, once its input JSON is fully assembled (not partial) |
| `tool_result` | `{"id", "name", "content", "is_error"}` | After a tool runs — `content` is the same string that goes back to Claude as the `tool_result` |
| `final` | The validated `WorkoutPlanResponse`, as JSON | Once `emit_plan`'s input passes validation — the turn is over |
| `error` | `{"message": str}` | An unrecoverable failure (see below) — always the last event |

Two things worth knowing before wiring a client against this:
- `emit_plan` also gets a `tool_call` event like any other tool, but only a
  *failing* validation gets a matching `tool_result` (`is_error: true`) —
  a successful one goes straight to `final` instead of a redundant
  success `tool_result` that would just repeat the same payload.
- The forced-finalize safety net (Claude ending its turn, or the
  iteration budget running out, without calling `emit_plan`) falls back
  to one non-streaming call, so a `final` event can occasionally arrive
  with no `text`/`tool_call` events immediately before it in that round.
- The HTTP response is always `200`, even on failure — by the time an
  error is known, the SSE headers are typically already sent, so success
  or failure is communicated by which event type shows up, not by the
  status code.

## Project layout

```
app/
  db.py       # SQLAlchemy Core schema + async engine
  models.py   # Pydantic: tool args, tool results, structured output
  agent.py    # The raw Claude tool-use loop (blocking + streaming)
  main.py     # FastAPI app: GET /health, POST /chat, POST /chat/stream
  seed.py     # Idempotent exercise-library seeding
  tools/      # One module per tool: schema + implementation
tests/        # Unit tests, HTTP-level tests, fake-client error simulation,
              # and one live-API integration test (isolated per-test SQLite file)
data/         # Seed data source (exercises.json)
Dockerfile              # Single-stage build, non-root runtime user
docker-entrypoint.sh    # Fixes volume ownership, then execs uvicorn as non-root
fly.toml                # Fly.io app config (region, volume, health check, sizing)
DEPLOY.md               # Full deploy runbook — already run once (see Status above)
```

## Running locally

Requires `ANTHROPIC_API_KEY` in the environment or in a `.env` file at the
project root (loaded automatically via `python-dotenv`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.seed          # creates fitness_agent.db, loads 20 exercises
pytest tests/ -v             # skips the live-API test if ANTHROPIC_API_KEY is unset
uvicorn app.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "chris", "message": "Give me a quick upper body plan for today."}'
```

```bash
# -N disables curl's output buffering — required to actually see events
# arrive incrementally instead of all at once when they're done.
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id": "chris", "message": "Give me a quick upper body plan for today."}'
```

Model defaults to `claude-opus-5`; override with `FITNESS_AGENT_MODEL`.

## Engineering writeup

This was built solo over five days as a portfolio project, not a
production system — the scope was chosen deliberately to be small enough
to build *and* verify properly in that time, rather than large enough to
look impressive and thin everywhere. What follows is meant to survive
someone asking "wait, does it actually do X" in an interview, which is a
different bar than a README that just describes what's there.

### What's cut for scope, on purpose

- **No auth.** `/chat` takes a bare `user_id` string in the request body
  and trusts it completely — there's no session, token, or login behind
  it. The one thing that *is* enforced is that Claude can't override the
  `user_id` the request claims (see the tool_result self-correction
  section below) — but nothing stops a client from claiming to be anyone.
  A real product needs real authentication in front of this; this project
  demonstrates the agent loop, not an auth system, and building a fake one
  would have added surface area without adding anything worth showing.
- **Single SQLite file, not shared storage.** The whole app is one
  process talking to one file on one Fly volume. That's a hard ceiling on
  horizontal scaling — see the Fly sizing entry below for why the config
  actively refuses to scale machine count rather than silently
  corrupting data if someone tries. Multi-user in the "many people use
  this at once" sense works fine (it's already how the live demo is
  shared); multi-user in the "runs across multiple machines" sense would
  need a real database first.
- **No framework.** Deliberate, and defended at length below — but the
  honest tradeoff is that LangChain/LangGraph would have given some
  things for free that this project doesn't have: built-in retry/backoff
  policies beyond what the Anthropic SDK itself does, tracing/eval
  tooling, and prior art other engineers on a team already know. Writing
  the loop by hand was the right call *for demonstrating that I
  understand what the loop does* — it is not automatically the right call
  for a team shipping a second agent next quarter.
- **No production observability.** There's no structured logging, no
  metrics, no tracing, no error-reporting integration (Sentry or
  similar). What exists is uvicorn's default access log and whatever Fly
  captures via `fly logs`. If this were serving real traffic, "a request
  failed" would currently mean grepping raw logs, not looking at a
  dashboard.
- **No app-level retries or backoff beyond what the SDK already does.**
  The `anthropic` SDK retries connection errors and 429/5xx with
  exponential backoff by default (`max_retries=2`) — this project doesn't
  add anything on top of that (no circuit breaker, no queue, no
  request-level retry policy of its own).
- **No conversation memory across requests.** Every `/chat` and
  `/chat/stream` call is a fresh turn — the tool-calling loop only
  persists *within* one request while Claude is deciding what to call
  next; there's no thread/session ID carrying history from one HTTP
  request to the next. A real coaching product would need that; this one
  answers one question at a time.

### What's verified live vs. only unit/mock-tested

Being specific about this matters more than it sounds — "tested" can mean
very different things, and it's worth being honest about which is which:

- **Verified against the real Anthropic API and/or the real deployed
  service:** the core happy path (`/chat` producing a valid plan), the
  `UnknownExerciseError` → `is_error` tool_result → Claude self-correction
  path (Day 2, watched it happen against a live model, not just asserted
  it would), real incremental SSE delivery — both locally and again
  inside the Docker container and again against the live Fly deployment,
  each time confirmed with `curl -N` and per-line timestamps showing real
  gaps between events, not a batched dump. One automated test
  (`test_agent.py`) also runs a real live call and asserts 2+ sequential
  tool calls happen before `emit_plan` — it's skipped, not failed, when
  `ANTHROPIC_API_KEY` isn't set, so this only runs where it can be trusted.
- **Verified against a real container, not against the real API:** Day 4's
  non-root-user check (`docker top`, not `docker exec whoami`, which
  would've given a false pass), the volume-ownership `chown` step (proven
  load-bearing against a deliberately root-owned volume, not just assumed
  from reading the entrypoint script), and data persistence across a full
  `docker rm` + recreate.
- **Only ever exercised against fakes, never against a real failure from
  Anthropic:** every error-handling path — rate limits, timeouts, the
  end-turn-without-`emit_plan` safety net, the forced-retry-then-give-up
  path — is proven correct against faithful stand-ins for the SDK's own
  exception types and response shapes (see `tests/fakes.py`), not against
  an actual observed rate-limit response in production. That's a
  reasonable trade for a project at this scale (deliberately triggering a
  real rate limit isn't practical to test against), but it's a real gap:
  nothing here has confirmed these paths still behave correctly if
  Anthropic ever changes the *shape* of these errors, only that the code
  branches correctly on the types as currently documented.
- **Pure unit tests, no API involved at all:** every Pydantic
  validation-boundary test (rejecting bad tool args, the `user_id`-spoofing
  regression tests, the duplicate-set behavior test) — correctly so, since
  these test our own validation code, not Claude's behavior, and don't
  need a model in the loop to be meaningful.

### The three decisions I'd defend hardest

1. **Raw tool-calling over a framework.** The loop is short enough
   (~200 lines across two functions sharing every validation/dispatch
   helper) that a framework would trade a small amount of boilerplate for
   a real loss of legibility — every "why does it do that" question in
   this README has a direct answer in `agent.py`, not "that's the
   framework's default." Full reasoning and the honest cost of this
   choice: above, and in the log below.
2. **Structured output forced via `tool_choice` on a validated `emit_plan`
   tool, not parsed from free text.** This is the actual guarantee that
   `/chat` always returns something `WorkoutPlanResponse`-shaped: Claude
   physically cannot "answer" without going through a tool call whose
   input gets Pydantic-validated before anything is returned to the
   client. Text parsing would be a best-effort convention; this is an
   enforced boundary.
3. **`is_error` tool_result as the *only* recoverable-failure mechanism,
   applied uniformly — including to `emit_plan` itself.** One pattern
   handles malformed tool args, an unrecognized exercise name, *and* a
   malformed final plan: catch it, hand Claude a clear message, let it
   retry in the same turn. This wasn't just designed, it was watched
   working live (the un-logged-exercise case in Day 2's manual testing) —
   the kind of thing that's easy to get subtly wrong (swallowing the
   error instead of surfacing it, or surfacing it in a shape Claude can't
   act on) and worth having actually seen work, not just implemented.

### Full decision log

The entries below are the detailed, chronological record each of the
above is drawn from — kept because "here's the actual tradeoff and what
I checked" is more useful in an interview than a summary alone.

**`emit_plan` as a fourth tool, validated like the other three (Day 2):**
Structured output is forced by giving Claude a terminal `emit_plan` tool
whose `input_schema` is generated directly from `WorkoutPlanResponse`
(`WorkoutPlanResponse.model_json_schema()`), so the tool schema and the
Pydantic validator can never drift apart. Rather than special-casing it,
the loop treats `emit_plan` exactly like `look_up_exercise`, `log_set`,
and `get_recent_history`: its input is validated with Pydantic, and a
`ValidationError` becomes an `is_error` tool_result that Claude sees and
can correct on the next turn — same mechanism, same error-recovery path.
The only thing that makes it "terminal" is that a *successful* validation
ends the loop instead of continuing it.

To guarantee every request actually resolves to a plan (this is the
model's decision to make, within limits — see below), `tool_choice` stays
`"auto"` through the loop so Claude can call `emit_plan` whenever it's
genuinely ready. Two situations force it instead: Claude ending its turn
without calling any tool (`stop_reason == "end_turn"`), and the loop
hitting `max_iterations` (8) tool-calling rounds. Both fall back to a
`tool_choice: {"type": "tool", "name": "emit_plan"}` call, which gets one
retry on a validation failure before raising `AgentError` (surfaced as
HTTP 502) — a small, bounded safety net rather than an unbounded retry
loop that would burn tokens on a truly broken response.

**`is_error` tool_result, not exceptions, for recoverable failures:**
Two failure modes are expected during a turn — malformed tool arguments
(Pydantic `ValidationError`) and `log_set` naming an exercise that isn't
in the library (`UnknownExerciseError`). Both are caught in the loop and
turned into a `tool_result` block with `is_error: true` and a human-
readable message, then handed straight back to Claude in the next
message instead of raising. This is what let the un-logged-exercise
case in manual testing resolve itself correctly: Claude got "'Zzz Made
Up Curl' is not in the exercise library" as a tool result, told the user
their set wasn't logged, and moved on — no retry logic needed on the
server side, no crashed turn.

**`user_id` is injected server-side, not trusted from the tool call:**
`log_set` and `get_recent_history` both take `user_id` as an argument in
their tool schemas (so Claude's mental model of the tool is simple and
self-contained), but the loop overwrites whatever value Claude passes
with the `user_id` from the actual `/chat` request before validating
args. The system prompt tells Claude the current user's id so it doesn't
need to guess — but the loop doesn't rely on it getting that right. Same
reasoning extends to `emit_plan.generated_at`: the model's timestamp is
discarded and replaced with the server's own `utcnow()` after validation.
These are small versions of the same validation-boundary principle from
Day 1 (never trust the LLM with something the caller already knows for
certain), applied to identity and time instead of exercise data.

**Manual loop, not the beta tool runner:** `anthropic`'s
`client.beta.messages.tool_runner()` would remove some of this
boilerplate, but it's a beta dependency and its per-turn hooks don't
cleanly express "validate before executing, treat one specific tool as
terminal, force it via `tool_choice` under specific conditions." Writing
the ~200-line loop by hand keeps every decision above visible and
explainable, which matters more here than the line count.

**Streaming shares validation/dispatch, not control flow, with `/chat`
(Day 3):** `stream_agent_events` (used by `/chat/stream`) and
`run_agent_loop` (used by `/chat`) both call the exact same
`_execute_client_tool`, `_validate_plan`, `_tool_result`,
`_build_system_prompt`, and `_force_finalize` — there is exactly one
place tool arguments get validated and exactly one place a plan gets
validated, regardless of which endpoint is serving the request. What
*isn't* shared is the outer tool-calling-round loop itself: one version
calls `client.messages.create()` and returns a value, the other calls
`client.messages.stream()` and yields events as they happen, and forcing
those two shapes into one function would have meant a generator that
`run_agent_loop` drains just to throw the events away — more indirection
for the reader, not less. I considered making `run_agent_loop` a thin
wrapper over `stream_agent_events` (drain the stream, return the `final`
event) to get to a single control-flow implementation, and deliberately
didn't: Day 2's loop was already written and verified against the live
API, and rebuilding its transport on top of the newer streaming path
right before hardening error handling seemed like the wrong moment to
introduce that risk. The result is a small, intentional amount of
duplication (the ~15 lines that iterate `tool_use` blocks and branch on
`emit_plan`) in exchange for not touching working code — a trade I'd
revisit if the two loops needed to grow a third variant.

**SSE error handling: typed exceptions first, one bare `except Exception`
at the transport boundary (Day 3):** Both endpoints catch the same
most-specific-first chain — `AgentError`, then `anthropic.RateLimitError`
/ `APITimeoutError` / `APIConnectionError` / `APIStatusError` — and turn
each into a clean response. For `/chat` that's an `HTTPException` with an
appropriate status code (429 for rate limits, 504 for timeouts, 502 for
everything else API-side); FastAPI's default handling already keeps a
truly unexpected exception from leaking a stack trace to the client, so
no catch-all is needed there. `/chat/stream` is different: once the SSE
response has started, there's no HTTP status code left to change, so a
genuinely unanticipated exception either becomes a clean `error` event or
the connection just dies mid-stream with no explanation. That's the one
place in the app with a deliberate, commented bare `except Exception` —
a transport-level safety net, not a tool-call error path. Every
tool-call and validation failure is still caught by its specific typed
exception well before it could reach that fallback.

**Tool-error hardening audit (Day 3):** went looking for exactly the class
of bug Pydantic can't catch — a tool_use input that's well-formed but
semantically wrong. The concrete case: could Claude's tool call name a
different `user_id` than the one making the request, and have that
silently read or write someone else's data? No — this was already closed
in Day 2's `_execute_client_tool`, which overwrites `user_id` with the
request's value before Pydantic ever validates the args; Day 3 just adds
`test_log_set_ignores_spoofed_user_id` and
`test_get_recent_history_ignores_spoofed_user_id` as regression tests so
it stays closed. One related case I checked and left alone: `log_set`
does an exact, case-sensitive match against the exercise name
(`exercise_exists`'s `==`, vs. `look_up_exercise`'s `ilike`), so a
lowercased or misspelled exercise name fails as `UnknownExerciseError` —
already a typed, recoverable, correctly-handled failure, not a silent
corruption, so it didn't need a fix. Separately, confirmed every tool
dispatch path only ever catches `ValidationError` and
`UnknownExerciseError` (see `_execute_client_tool`) — no bare
`except Exception` anywhere in the tool-calling path — and added tests
that inject a fake client raising `anthropic.RateLimitError` mid-loop,
both at the `agent.py` level (confirming the error propagates as itself,
unmodified — nothing swallows it into a generic string) and at the HTTP
level (confirming `/chat` returns 429 with a clean `detail` and
`/chat/stream` emits a clean `error` event, neither with a stack trace in
the body).

**Fixed a dependency-hygiene issue in the test suite (pre-Day-4 cleanup):**
`tests/fakes.py` originally built fake Anthropic errors using `httpx2`
(`anthropic`'s transport library) directly — it worked, but `httpx2` was
never declared in `requirements.txt`, only present because `anthropic`
happens to depend on it today. Refactored to construct the real
`anthropic.RateLimitError` / `anthropic.APITimeoutError` instances via
their own public constructors (`response=`, `body=`, `request=`) with a
minimal duck-typed stand-in object satisfying only what our own error
handling actually reads (`.status_code`) — this fakes at the boundary our
code actually depends on (the `anthropic` package's exception types) 
instead of an internal implementation detail one level further down that
could change on an unrelated `anthropic` upgrade. Verified by installing
`requirements.txt` into a brand-new venv and running the full suite
(27/27 passing) with no other packages present. Also noticed
`make_timeout_error` was defined but never called by any test — added
`test_run_agent_loop_propagates_timeout_error` and
`test_chat_surfaces_timeout_as_clean_error` rather than leaving dead code
in a test-fakes module.

**Dockerfile: single-stage, non-root via a `gosu` entrypoint (Day 4):**
Checked whether multi-stage would actually shrink the image before
reaching for it — it wouldn't: every package in `requirements.txt`,
including `uvicorn[standard]`'s C-extension deps (`uvloop`, `httptools`)
and `pydantic`'s Rust core (`pydantic-core`), resolves to a prebuilt
`manylinux` wheel on a `python:3.12-slim` (glibc) base, verified with
`pip download --only-binary=:all:`. No compiler runs during `pip install`,
so there's no build-stage artifact to discard — multi-stage would just
add a second `FROM` that copies everything across. The container starts
as root (needed once, to `chown` a freshly mounted Fly volume, which
always arrives root-owned regardless of what the image set at build
time), then `docker-entrypoint.sh` drops to a dedicated `appuser` via
`gosu` before `exec`-ing uvicorn — `gosu` instead of `su` specifically
because `su` forks a shell around the child process, which can absorb the
`SIGTERM` a graceful shutdown sends; `gosu` execs directly so the signal
still reaches uvicorn.

**Seed data ships in the image; the database never does (Day 4):**
`data/exercises.json` (the seed *source*) is copied into the image — it's
static input the seed script needs regardless of environment. The SQLite
*database* is deliberately never baked in: `FITNESS_AGENT_DB` points at
`/data/fitness_agent.db`, a path that only becomes real, persistent
storage once a Fly volume is mounted there at runtime, and `app/main.py`'s
existing idempotent `seed()` call (already run on every boot since Day 2)
populates whatever's on that volume — the image itself ships zero rows.

**Fly.io sizing: scale-to-zero, single machine, and why (Day 4):**
`min_machines_running = 0` + `auto_stop_machines = "stop"` on the smallest
machine size (`shared-cpu-1x`, 256mb) trades a few seconds of cold-start
latency for not paying for idle compute — the right trade for a demo an
interviewer clicks on sporadically, not for production traffic. The
volume is pinned to one machine on purpose: a Fly volume is
machine-attached storage, not shared network storage, so scaling machine
count above 1 wouldn't give SQLite more capacity, it would silently give
each machine its *own* divergent copy of the database. That's not a bug
to work around here — it's the same "SQLite is the right amount of
database for this project" call from Day 1's SQLAlchemy-Core decision,
just showing up again at the infra layer. `DEPLOY.md` documents this as a
hard constraint (don't `fly scale count`) rather than something the config
quietly protects against, since flyctl doesn't have a "max machines" knob
to enforce it structurally.

**Deploy docs are verified against current Fly.io docs, not memory (Day 4):**
Fly's config format and CLI have changed more than once (the Nomad→
Machines migration, `[[services]]` → `[http_service]`), so before writing
`fly.toml` or `DEPLOY.md` I checked the current syntax for `[http_service]`,
`[[http_service.checks]]`, `[mounts]`, `[[vm]]`, and the rollback flow
against Fly's live docs rather than trusting a remembered shape — and
where the docs didn't have a clear answer (whether `fly apps destroy`
also deletes attached volumes), `DEPLOY.md` says so explicitly and gives
a verify-don't-assume teardown sequence instead of a confident-sounding
guess. This paid off in practice: `fly launch` diverged slightly from the
hand-written `fly.toml` (`[mounts]` came back as `[[mounts]]`, plus an
auto-added `memory_mb` alongside `memory`) — cosmetic, not a config bug,
but exactly the kind of drift that makes "verify against current docs"
worth doing rather than trusting the file as originally written.

**Naming clarity: `based_on_history` → `history_checked` (Day 5):** the
field was `true` any time `get_recent_history` had been *called* that
turn, including for a brand-new user where it returned zero sets — which
reads as "this plan reflects your history" when what actually happened is
"the check ran and found nothing." Renamed rather than adding a second
field (e.g. "checked but empty" vs. "checked and used"): that distinction
would require Claude to accurately self-report a judgment call — did the
(possibly empty) history actually change what I planned? — which is a
harder and less reliably honest thing to ask a model to certify than "did
the tool run," and the schema complexity isn't worth it for what this
field is actually used for (a debugging/transparency signal, not
something branched on downstream).

**Duplicate sets: not deduplicated, on purpose (Day 5):** logging
"Back Squat, 225 lbs × 5" twice in the same session is normal, real
training data — two straight sets at the same weight — not obviously a
mistake. `log_set` has no concept of "this looks like the last one," and
guessing wrong (silently merging or rejecting a genuine second set) loses
real data, which is worse than the rare case of an actual accidental
double-submission producing an extra row. No deduplication logic was
added; `test_logging_the_same_set_twice_creates_two_rows` locks in that
this is the intended behavior, not an untested gap.

**Edge cases proven, not just reasoned about (Day 5):** two things this
project's own logic already implied were true, turned into actual
regression tests rather than left as "should work" — (1) empty/whitespace
`message` is now rejected by `ChatRequest`'s own validation before any
Anthropic call happens (a clean 422, not a wasted API call); (2) Claude
ending its turn with plain text instead of calling `emit_plan` is handled
by the exact same `end_turn` branch as every other iteration, not a
special last-iteration case — proven with a scripted fake client
(`EndTurnThenForcedClient` in `tests/fakes.py`) that always ends the turn
with text under `tool_choice: "auto"`, for *both* `run_agent_loop` and
`stream_agent_events`, since Day 3's design log already noted these two
loops don't share control flow and a fix or regression in one wouldn't
necessarily show up in a test of the other.
