"""
Test doubles for simulating Anthropic API failures without hitting the
network — used to verify the agent loop and FastAPI routes surface these
as clean, typed errors instead of a stack trace reaching the client.

These build real `anthropic.*` exception instances via the SDK's own
public constructors (`response=`, `body=`, `request=` kwargs — the same
shape shown in Anthropic's own error-handling docs), not the transport
library's `Response`/`Request` classes. `anthropic` currently builds those
exceptions on top of `httpx2`, but `httpx2` is an internal implementation
detail of the `anthropic` package — it's not declared in requirements.txt,
and depending on it directly here would mean these fakes could break on
an `anthropic` upgrade that changes its transport, for no benefit to what
we're actually testing (that our code branches correctly on exception
*type*, and reads `.status_code` off `APIStatusError`).
"""
from types import SimpleNamespace

import anthropic


class _FakeHTTPResponse:
    """Just enough of an httpx-response shape for APIStatusError.__init__
    (status_code, .headers.get(...), .request) — nothing more, since
    that's all our error-handling code ever reads off these exceptions."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        self.headers: dict = {}
        self.request = None


def make_rate_limit_error() -> anthropic.RateLimitError:
    message = "Number of request tokens has exceeded your per-minute rate limit."
    body = {"error": {"type": "rate_limit_error", "message": message}}
    return anthropic.RateLimitError(message, response=_FakeHTTPResponse(429), body=body)


def make_timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=None)


class _FailingMessages:
    def __init__(self, error: Exception):
        self._error = error

    async def create(self, **kwargs):
        raise self._error

    def stream(self, **kwargs):
        raise self._error


class FailingClient:
    """Stands in for AsyncAnthropic; every messages.create/stream call raises `error`."""

    def __init__(self, error: Exception):
        self.messages = _FailingMessages(error)


# ---------------------------------------------------------------------------
# Scripted client: Claude ends its turn with plain text instead of calling
# emit_plan, then produces a valid plan once tool_choice forces it. Used to
# prove the end_turn safety net (app/agent.py's _force_finalize) actually
# resolves to a valid plan, for both the blocking and streaming loops,
# rather than just asserting it "should" work from reading the code.
# ---------------------------------------------------------------------------


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(block_id: str, name: str, input_: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=input_)


def _message(stop_reason: str, content: list) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class _FakeStream:
    """Just enough of the streaming context-manager protocol for
    stream_agent_events: async with ... as stream / async for event in
    stream / await stream.get_final_message(). Yields no delta events —
    only the final message matters for these tests."""

    def __init__(self, final_message: SimpleNamespace):
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def __aiter__(self):
        return self._empty()

    async def _empty(self):
        return
        yield  # pragma: no cover - unreachable; makes this an async generator

    async def get_final_message(self):
        return self._final_message


class EndTurnThenForcedClient:
    """First call (tool_choice: auto) always ends the turn with plain text
    and no tool call — simulating Claude just chatting instead of calling
    emit_plan. Any forced call (tool_choice: {"type": "tool", ...}) returns
    a valid emit_plan tool_use block instead, so _force_finalize's retry
    succeeds on the first attempt. `calls` records (method, tool_choice)
    for each invocation, so a test can confirm which path was taken."""

    def __init__(self, valid_plan_input: dict, text: str = "Sure! Here's some general advice..."):
        self.calls: list[tuple[str, dict | None]] = []
        self._valid_plan_input = valid_plan_input
        self._end_turn_text = text
        self.messages = self._Messages(self)

    def _end_turn_message(self) -> SimpleNamespace:
        return _message("end_turn", [_text_block(self._end_turn_text)])

    def _forced_plan_message(self) -> SimpleNamespace:
        return _message(
            "tool_use",
            [_tool_use_block("toolu_fake_emit_plan", "emit_plan", self._valid_plan_input)],
        )

    class _Messages:
        def __init__(self, outer: "EndTurnThenForcedClient"):
            self._outer = outer

        def stream(self, **kwargs) -> _FakeStream:
            self._outer.calls.append(("stream", kwargs.get("tool_choice")))
            return _FakeStream(self._outer._end_turn_message())

        async def create(self, **kwargs) -> SimpleNamespace:
            tool_choice = kwargs.get("tool_choice")
            self._outer.calls.append(("create", tool_choice))
            if tool_choice and tool_choice.get("type") == "tool":
                return self._outer._forced_plan_message()
            return self._outer._end_turn_message()
