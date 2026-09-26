"""
Which model the eval harness runs the agent under test on.

Decided on Day 2: Haiku 4.5. The Opus 5 baseline scored 4.84/5 (no room
for a loop to improve); Haiku scored 3.06 with real, learnable failures.
Independent of the deployed app's FITNESS_AGENT_MODEL on purpose — the
loop is an experiment on a Haiku-based agent, not a change to production.

`pin_agent_model` must be called explicitly (eval/runner.py does). It is
deliberately NOT applied at import: tests import eval modules, and an
import-time override of app.agent.DEFAULT_MODEL would silently retarget
every other test in the same process.

Why pin app.agent.DEFAULT_MODEL at all: app.agent._force_finalize reads
that module global at call time, so without the pin a Haiku run's
forced-finalize calls would quietly go to whatever the deployed default
is (Opus), mixing two models inside one "Haiku" result.
"""
from __future__ import annotations

import os

AGENT_MODEL = os.environ.get("FITNESS_EVAL_AGENT_MODEL", "claude-haiku-4-5")


def pin_agent_model() -> str:
    import app.agent as agent

    agent.DEFAULT_MODEL = AGENT_MODEL
    return AGENT_MODEL
