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
from fastapi.responses import StreamingResponse
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
