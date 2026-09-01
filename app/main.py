"""
FastAPI app.

- GET  /health        — liveness
- POST /chat          — one full agent turn, blocking, returns the validated plan
- POST /chat/stream    — the same turn, as Server-Sent Events (see README for
                         the event schema)

Both /chat and /chat/stream catch a most-specific-first chain of Anthropic
SDK exceptions (rate limits, timeouts, connection errors, other API status
errors) plus our own AgentError, and turn each into a clean, typed response
instead of letting a raw exception reach the client. Tool-call failures
never surface here at all — those are handled inside app/agent.py and
turned into is_error tool_results before the model ever produces a final
answer.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.agent import AgentError, run_agent_loop, stream_agent_events
from app.db import get_engine, init_db
from app.models import WorkoutPlanResponse
from app.seed import seed


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    await init_db(engine)
    await seed()  # idempotent — safe to run on every startup
    app.state.engine = engine
    app.state.anthropic_client = anthropic.AsyncAnthropic()
    yield
    await engine.dispose()


app = FastAPI(title="Fitness Coaching Agent", lifespan=lifespan)
GITHUB_REPO_URL = "https://github.com/chpham92/fitness-agent"

LANDING_HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fitness Coaching Agent</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 720px; margin: 4rem auto; padding: 0 1.5rem;
    line-height: 1.55; color: #1a1a1a;
  }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
  p.tagline {{ color: #555; margin-top: 0; }}
  code, pre {{ background: #f4f4f4; border-radius: 6px; font-size: 0.85rem; }}
  code {{ padding: 0.15rem 0.4rem; }}
  pre {{ padding: 0.9rem 1rem; overflow-x: auto; }}
  .endpoint {{ margin: 1.75rem 0; }}
  .endpoint h2 {{ font-size: 1rem; margin-bottom: 0.4rem; }}
  a {{ color: #0a5cd6; }}
  footer {{ margin-top: 3rem; font-size: 0.85rem; color: #777; }}
  @media (prefers-color-scheme: dark) {{
    body {{ color: #e6e6e6; }}
    p.tagline {{ color: #aaa; }}
    code, pre {{ background: #1e1e1e; }}
    a {{ color: #6ea8ff; }}
    footer {{ color: #999; }}
  }}
</style>
</head>
<body>
  <h1>Fitness Coaching Agent</h1>
  <p class="tagline">
    A tool-calling LLM agent (raw Anthropic Messages API, FastAPI, Pydantic,
    SQLite) deployed live on Fly.io — not a notebook.
  </p>
  <div class="endpoint">
    <h2>GET /health</h2>
    <pre>curl https://fitness-coaching-agent.fly.dev/health</pre>
  </div>
  <div class="endpoint">
    <h2>POST /chat</h2>
    <pre>curl -X POST https://fitness-coaching-agent.fly.dev/chat \\
  -H "Content-Type: application/json" \\
  -d '{{"user_id": "demo", "message": "give me a quick upper body workout"}}'</pre>
  </div>
  <div class="endpoint">
    <h2>POST /chat/stream (Server-Sent Events)</h2>
    <pre>curl -N -X POST https://fitness-coaching-agent.fly.dev/chat/stream \\
  -H "Content-Type: application/json" \\
  -d '{{"user_id": "demo", "message": "give me a quick upper body workout"}}'</pre>
  </div>
  <footer>
    <a href="{GITHUB_REPO_URL}">Source on GitHub</a>
  </footer>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return LANDING_HTML

class ChatRequest(BaseModel):
    user_id: str
    # Bounded and non-blank for the same reason every tool arg in models.py
    # is: an untrusted request body shouldn't reach the Anthropic API (a
    # real network call) before the cheapest possible check has run. An
    # empty/whitespace message would otherwise become a Claude turn with
    # nothing to respond to — a wasted call, not a crash, but not useful
    # either — so it's rejected here with a 422 instead.
    message: str = Field(..., max_length=4000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message cannot be empty or whitespace-only")
        return v


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=WorkoutPlanResponse)
async def chat(req: ChatRequest):
    try:
        return await run_agent_loop(
            engine=app.state.engine,
            client=app.state.anthropic_client,
            user_id=req.user_id,
            message=req.message,
        )
    except AgentError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited by the model provider. Please retry shortly.")
    except anthropic.APITimeoutError:
        raise HTTPException(status_code=504, detail="The model provider timed out. Please retry.")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=502, detail="Could not reach the model provider.")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Model provider error ({e.status_code}).")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_source():
        try:
            async for evt in stream_agent_events(
                engine=app.state.engine,
                client=app.state.anthropic_client,
                user_id=req.user_id,
                message=req.message,
            ):
                yield _sse(evt["event"], evt["data"])
        except AgentError as e:
            yield _sse("error", {"message": str(e)})
        except anthropic.RateLimitError:
            yield _sse("error", {"message": "Rate limited by the model provider. Please retry shortly."})
        except anthropic.APITimeoutError:
            yield _sse("error", {"message": "The model provider timed out. Please retry."})
        except anthropic.APIConnectionError:
            yield _sse("error", {"message": "Could not reach the model provider."})
        except anthropic.APIStatusError as e:
            yield _sse("error", {"message": f"Model provider error ({e.status_code})."})
        except Exception:
            # Last-resort transport guard, deliberately bare: once the SSE
            # response has started, there's no HTTP status code left to
            # change, so the only options are a clean "error" event or a
            # connection that just dies mid-stream. Every failure mode we
            # actually anticipate is caught above with a typed exception —
            # this only exists to catch a genuine bug and still close the
            # stream cleanly rather than dropping it silently.
            yield _sse("error", {"message": "Internal error."})

    return StreamingResponse(event_source(), media_type="text/event-stream")
