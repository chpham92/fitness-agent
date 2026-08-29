# Single-stage: every dependency in requirements.txt resolves to a
# prebuilt manylinux wheel for this base image (verified with
# `pip download --only-binary=:all: --platform manylinux2014_x86_64
# --python-version 3.12 -r requirements.txt` — including uvicorn[standard]'s
# C-extension deps uvloop/httptools and pydantic's Rust core pydantic-core).
# No compiler ever runs during `pip install`, so there's no build stage
# producing intermediate artifacts a second stage would need to discard —
# multi-stage would just add a COPY --from=builder that copies everything.
FROM python:3.12-slim

# gosu, not `su`: the entrypoint drops from root to the app user via exec,
# not a fork. `su` wraps the child in a shell, which can swallow the
# SIGTERM a graceful `docker stop` / Fly deploy sends, delaying shutdown
# until the hard-kill timeout. gosu execs directly, so the signal reaches
# uvicorn.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Dependencies before app code: editing app/ shouldn't bust the pip-install
# layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY data/ data/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# /data is where a Fly volume gets mounted at runtime (see fly.toml). Only
# the seed *source* (data/exercises.json, copied above) is baked into the
# image — the sqlite database itself never is. app/main.py's startup hook
# calls the idempotent seed() against this path on every boot, so it's the
# volume's data that gets populated, not the image's.
ENV FITNESS_AGENT_DB=/data/fitness_agent.db
RUN mkdir -p /data && chown appuser:appuser /data

# ANTHROPIC_API_KEY is intentionally absent from this file — it's supplied
# at runtime via `fly secrets set` (see DEPLOY.md), never baked into a layer.

EXPOSE 8000

# No static `USER appuser` here, deliberately: the container has to start
# as root so the entrypoint can chown the freshly mounted volume (a
# non-root user can't chown a directory it doesn't already own). The
# entrypoint drops to appuser via gosu immediately after that, before
# uvicorn ever starts — root only exists for that one fix-permissions step.
ENTRYPOINT ["docker-entrypoint.sh"]
