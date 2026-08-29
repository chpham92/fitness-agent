# Deploy runbook (Fly.io)

Written for future-you, who has forgotten all of this. Read the whole
thing once before running anything — the order matters, and step 3
(volume) has to happen before the first real deploy or it will fail.

**Local Docker verification is done (see "Local verification" below);
nothing has touched Fly yet.** `fly launch` / `fly volumes create` /
`fly deploy` have never been run — no app, volume, or machine exists on
Fly for this project as of this writing.

## Local verification (done — `docker build`/`docker run`, no Fly involved)

Before trusting this config against a real account, built and ran the
image locally end to end:

- `docker build` succeeds clean — confirms the "every dependency ships a
  prebuilt wheel, no compiler needed" claim behind the single-stage
  decision.
- Confirmed via `docker top` (not `docker exec whoami`, which spawns a
  *new* process that defaults to root regardless of what PID 1 is doing)
  that the actual running `uvicorn` process has UID `10001` (`appuser`),
  not root.
- Confirmed the entrypoint's `chown` step is real, load-bearing work, not
  a no-op: forced a Docker volume to genuinely root-owned (matching a
  freshly created, never-mounted Fly volume) and confirmed the
  unmodified entrypoint correctly re-owns it to `appuser` before
  `uvicorn` starts, and that `appuser` is correctly denied writes
  elsewhere (`/root`, `/usr`) — i.e. the drop to non-root is real, not
  just cosmetic.
- `/health` → 200, `/chat` → a validated plan, `/chat/stream` → real
  incremental SSE (distinct timestamped `text` deltas, not a single
  batched dump) — all reproduced *inside the container*, not just
  locally without Docker.
- Confirmed `ANTHROPIC_API_KEY` never appears in the image: checked
  `docker image inspect`'s `Config.Env`, and ran a fresh container with
  no `--env-file` to confirm zero `ANTHROPIC_*` env vars and no stray
  `.env` file anywhere in the image filesystem. The key only ever enters
  via `docker run --env-file .env` (locally) or `fly secrets` (on Fly).
- Stopped and fully removed the container, started a new one against the
  *same* bind-mounted directory, and confirmed a set logged before the
  teardown was still there afterward — via a real `/chat` call asking for
  the user's history, not just a raw SQL check.

**Methodology gotcha worth knowing if you ever redo this test:** Docker's
own named volumes (`docker volume create` + `-v name:/data`) have a
convenience feature where an *empty* volume's first mount copies in the
image's own directory content and ownership at that path. Since this
image's `/data` is baked in as `appuser`-owned, a plain named-volume test
would show everything working *even with the entrypoint's `chown` step
deleted* — a false positive. Fly's real volumes don't get this treatment
(they're raw block devices Fly formats and mounts, not something routed
through `dockerd`'s volume driver), so the meaningful local test is
either a bind-mount (used above for the main walkthrough) or a named
volume you've explicitly forced back to root ownership first (used above
to isolate the `chown` step specifically). Don't trust a plain named-volume
test alone to validate this again later.

No code changes were needed to the Dockerfile, entrypoint, or fly.toml —
everything worked as designed on the first real build. This section
exists so "we verified it" means something concrete instead of "it looked
right on paper," which is exactly the gap this session closed.

## Prerequisites

- `flyctl` installed (`brew install flyctl` on macOS, or see
  https://fly.io/docs/flyctl/install/)
- A Fly.io account with billing set up (even the smallest machine needs a
  card on file, though this config is sized to cost very little — see
  "Cost shape" below)
- Your `ANTHROPIC_API_KEY` on hand — it goes into Fly's secret store, not
  any file in this repo

## 0. Log in

```bash
fly auth login
```

## 1. Register the app (no deploy yet)

From the project root, where `fly.toml` already exists:

```bash
fly launch --no-deploy
```

`fly launch` is generally good about detecting an existing `fly.toml` /
`Dockerfile` and offering to reuse them rather than generating fresh
ones — but flyctl's exact interactive prompts change across versions, so
don't take that on faith: **back up `fly.toml` before running this**
(`cp fly.toml fly.toml.bak`), answer prompts to keep the existing config
where offered, and diff the result against the backup afterward
(`diff fly.toml fly.toml.bak`) to catch anything it rewrote — particularly
the `[mounts]` and `[http_service]` blocks, which are the parts worth
protecting. It will also ask to confirm (or change) the app name —
`fitness-coaching-agent` in `fly.toml` is a placeholder; Fly app names are
globally unique, so if it's taken, let it pick a free one and update the
`app = "..."` line to match before continuing, so the two stay in sync.

`--no-deploy` means this step only registers the app name with Fly — it
does not build or deploy anything yet.

## 2. Create the persistent volume

The `[mounts]` block in `fly.toml` references a volume named
`fitness_agent_data` by name — it has to exist before the first deploy
that mounts it, or the deploy will fail outright.

```bash
fly volumes create fitness_agent_data --region iad --size 1
```

(`--size` is in GB; 1 is Fly's practical floor and wildly more than a
SQLite file with 20 exercises and some logged sets needs — see the
`initial_size` comment in `fly.toml`.)

**Important — do this once.** If you ever accidentally run `fly launch`
or `fly volumes create` again for this app, don't create a *second*
volume in the same region unless you mean to abandon the first one's
data. `fly volumes list` shows what exists.

## 3. Set the API key secret

```bash
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
```

This is encrypted at rest by Fly and injected as an environment variable
at runtime — it never touches `fly.toml`, the image, or this repo.
Setting a secret triggers a new deploy on its own if the app already has
a release; on a brand-new app (like right now) it just stages the value
for the first deploy.

## 4. Deploy

```bash
fly deploy
```

This builds the image from `Dockerfile` (remotely, on Fly's builders, by
default — no local Docker install required) and rolls it out. First
deploy will also attach the volume created in step 2.

## 5. Verify

```bash
fly status
curl https://<your-app-name>.fly.dev/health
# {"status":"ok"}
```

If `min_machines_running = 0` has already kicked in and the machine is
stopped, the first `curl` will take a few seconds (cold start) before
responding — that's expected, not a failure.

## Seeding the production database

**Nothing manual is required.** `app/main.py`'s startup hook calls the
same idempotent `seed()` used locally, every time the app boots, against
whatever's at `FITNESS_AGENT_DB` (`/data/fitness_agent.db` — the mounted
volume). The first real request after the first deploy will have already
triggered that boot, so the exercise library is populated by the time you
run the health check above.

To double-check, or to re-run it manually for any reason:

```bash
fly ssh console -C "python -m app.seed"
# Seeded 20 new exercise(s).   (or "Seeded 0 ..." if already populated — that's correct, it's idempotent)
```

To peek at what's actually in the volume's database without a `sqlite3`
CLI (not installed in the image — kept out deliberately to keep it lean):

```bash
fly ssh console -C "python -c \"
import sqlite3
conn = sqlite3.connect('/data/fitness_agent.db')
print(conn.execute('select count(*) from exercises').fetchone())
print(conn.execute('select count(*) from sets').fetchone())
\""
```

## Everyday redeploys

Once the above has run once, shipping a code change is just:

```bash
fly deploy
```

No volume/secret steps needed again unless you're rotating the API key
(`fly secrets set ANTHROPIC_API_KEY=...` again) or the volume is gone.

## Logs

```bash
fly logs                 # tail, real-time
fly logs --no-tail        # recent history, then exit
```

## Rollback

```bash
fly releases              # list past releases: version, status, who/when
fly releases --image      # same, but with each release's image reference
fly deploy --image <image_ref_from_a_previous_release>
```

`fly deploy --image ...` redeploys that exact previously-built image
without rebuilding — this is the actual rollback mechanism (there's no
separate `fly releases rollback` command). Confirm the working version
with `fly status` / a `/health` + a real `/chat` request afterward.

## Cost shape (why this config is sized the way it is)

- `shared-cpu-1x`, `256mb` — the smallest machine size, plenty for one
  uvicorn process serving low, bursty demo traffic.
- `min_machines_running = 0` + `auto_stop_machines = "stop"` — the
  machine stops entirely when idle and boots on the next request. You
  only pay for compute while it's actually running, at the cost of a
  cold start (a few seconds) on the first request after a quiet period.
- The volume (1gb) bills separately and continuously regardless of
  whether the machine is running — it's the only always-present cost
  here, and it's Fly's minimum size.
- Check `fly platform pricing` or the Fly dashboard for current rates
  before relying on any of the above as a dollar figure — pricing is
  Fly's to change, not something to hardcode into a runbook.

## Known limitation: don't scale machine count above 1

**Do not run `fly scale count 2` (or higher) with this config as-is.** A
Fly volume is attached storage for *one specific machine*, not shared
network storage — running more than one machine with a `[mounts]` block
like this gives each machine its *own independent* volume, meaning each
one would get its own separate, silently diverging SQLite file. This
isn't a hypothetical edge case, it's the direct consequence of the same
"why SQLAlchemy Core, not the ORM" tradeoff from Day 1: SQLite is the
right amount of database for a single-instance portfolio demo, and the
deploy topology has to match that. If this ever needs real concurrent
capacity, the fix is switching to a shared database (e.g. Fly Postgres)
before scaling machine count — not a Fly config change on its own.

## Tearing everything down

If this ever needs to stop costing anything at all, destroy the app
*and* explicitly check for a leftover volume — Fly's own docs describe
volumes as able to exist "unattached," independent of the app/machine
that created them, and don't clearly document whether `apps destroy`
cascades to volumes. Don't assume it does; verify.

```bash
fly apps destroy fitness-coaching-agent
fly volumes list                              # check for fitness_agent_data
fly volumes destroy <volume-id>               # if it's still listed
```

This is irreversible — the volume (and everything logged in it) is gone
once destroyed. There's no separate "pause billing but keep the data"
mode beyond the scale-to-zero behavior already configured above.
