#!/bin/sh
set -e

# A freshly mounted Fly volume always arrives owned by root, regardless of
# what the image chowned at build time — the volume is a separate
# filesystem swapped in at container start, not a layer, so it overrides
# whatever ownership was baked in. Fix it here, every boot, before
# dropping to the non-root user. Cheap and idempotent on repeat deploys
# against an already-owned volume.
chown -R appuser:appuser "$(dirname "$FITNESS_AGENT_DB")"

# uvicorn binds $PORT if the platform sets it (the Heroku/Render/Cloud Run
# convention); Fly instead declares the port in fly.toml's internal_port,
# so this falls back to a fixed default that matches it there.
exec gosu appuser uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
