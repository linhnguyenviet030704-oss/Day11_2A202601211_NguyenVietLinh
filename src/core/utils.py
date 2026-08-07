"""
Lab 11 — Helper Utilities
"""
import asyncio
import os
import time
from collections import deque
from contextlib import asynccontextmanager

from google.genai import types

from core.config import MODEL_RPM, MODEL_TPM

# Assume this much output when reserving budget, before the real usage is known.
_ASSUMED_OUTPUT_TOKENS = 400

# The shared Gemma endpoint returns 503 under load. Those calls never reached the
# model, so retrying is the difference between a real answer and an empty failure.
_MAX_ATTEMPTS = 5
_RETRY_BASE_SECONDS = 8

# Neither google-genai nor ADK sets a socket timeout, so a stalled request hangs
# the whole run. Cap it and let the retry path treat the timeout as transient.
_CALL_TIMEOUT_SECONDS = float(os.environ.get("LAB_CALL_TIMEOUT", "120"))
_TRANSIENT_MARKERS = (
    "503", "unavailable", "overloaded", "high demand",
    "500", "internal error", "429", "resource_exhausted", "deadline",
)


def _is_transient(exc: BaseException) -> bool:
    """True when the provider failed for a reason a later identical call may not hit."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = f"{type(exc).__name__} {exc}".lower()
        if any(marker in text for marker in _TRANSIENT_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


async def _resolve_session(runner, app_name: str, user_id: str, session_id):
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
            if session is not None:
                return session
        except (ValueError, KeyError):
            pass
    return await runner.session_service.create_session(
        app_name=app_name, user_id=user_id
    )


class RateGate:
    """Keep every LLM call inside a requests-per-minute and tokens-per-minute budget.

    Calls are also serialized: the caller holds this gate for the whole request,
    so two prompts are never in flight at once. That is what makes the token
    accounting trustworthy — a parallel burst would commit tokens the window has
    not accounted for yet.

    ponytail: single-process sliding windows, no persistence. It assumes this
    program is the only consumer of the API key; a second process running at the
    same time would double the real rate. Upgrade path is a shared counter
    (Redis INCR on a per-minute key) keyed by the API key instead of these deques.
    """

    WINDOW = 60.0

    def __init__(self, rpm: int = MODEL_RPM, tpm: int = MODEL_TPM):
        self.rpm = rpm
        self.tpm = tpm
        self.requests: deque[float] = deque()
        self.tokens: deque[list] = deque()      # [timestamp, token_count]
        self.waits = 0
        self.total_wait_seconds = 0.0
        self.retries = 0
        self._lock = asyncio.Lock()

    def _prune(self, now: float):
        cutoff = now - self.WINDOW
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
        while self.tokens and self.tokens[0][0] <= cutoff:
            self.tokens.popleft()

    def _tokens_in_window(self) -> int:
        return sum(entry[1] for entry in self.tokens)

    def _seconds_until_room(self, now: float, reserve: int) -> float:
        """How long before both budgets can accept one more call of this size."""
        waits = [0.0]
        if len(self.requests) >= self.rpm:
            waits.append(self.requests[0] + self.WINDOW - now)
        if self._tokens_in_window() + reserve > self.tpm:
            # Release oldest token entries until the reservation fits.
            freed = 0
            need = self._tokens_in_window() + reserve - self.tpm
            for ts, count in self.tokens:
                freed += count
                if freed >= need:
                    waits.append(ts + self.WINDOW - now)
                    break
            else:
                # Reservation alone exceeds TPM; wait out the whole window.
                waits.append(self.WINDOW)
        return max(waits)

    async def reserve(self, reserve_tokens: int):
        """Block until this call fits in both budgets, then claim a slot."""
        while True:
            now = time.monotonic()
            self._prune(now)
            delay = self._seconds_until_room(now, reserve_tokens)
            if delay <= 0:
                break
            self.waits += 1
            self.total_wait_seconds += delay
            print(f"  [rate gate] waiting {delay:.1f}s to stay under {self.rpm} RPM / {self.tpm} TPM")
            await asyncio.sleep(delay)
        now = time.monotonic()
        self.requests.append(now)
        self.tokens.append([now, reserve_tokens])

    def settle(self, actual_tokens: int | None):
        """Replace the reservation with the token count the provider reported."""
        if actual_tokens and self.tokens:
            self.tokens[-1][1] = actual_tokens

    @asynccontextmanager
    async def slot(self, reserve_tokens: int):
        """Hold the gate for one whole call: serialize, reserve, then reconcile.

        Yields a dict the caller fills with ``tokens`` once the provider reports
        real usage; the estimate is corrected on exit.
        """
        async with self._lock:
            await self.reserve(reserve_tokens)
            usage = {"tokens": None}
            try:
                yield usage
            finally:
                self.settle(usage["tokens"])

    def snapshot(self) -> dict:
        now = time.monotonic()
        self._prune(now)
        return {
            "rpm_limit": self.rpm,
            "tpm_limit": self.tpm,
            "requests_last_minute": len(self.requests),
            "tokens_last_minute": self._tokens_in_window(),
            "throttle_waits": self.waits,
            "throttle_wait_seconds": round(self.total_wait_seconds, 1),
            "transient_retries": self.retries,
        }


# One gate for the whole process — agents, judges and red team share the quota.
rate_gate = RateGate()


def estimate_tokens(text: str) -> int:
    """Rough pre-call size. ~4 characters per token is close enough to reserve on."""
    return max(1, len(text or "") // 4) + _ASSUMED_OUTPUT_TOKENS


def _usage_total(obj) -> int | None:
    usage = getattr(obj, "usage_metadata", None)
    return getattr(usage, "total_token_count", None) if usage else None


async def throttled_generate(client, *, model: str, contents: str, config=None):
    """Run a raw LLM call through the same gate as the agents.

    Ollama path ignores ``client`` and hits local /api/chat; Google path uses
    google-genai as before.
    """
    from core.config import OLLAMA_HOST, use_ollama

    async with rate_gate.slot(estimate_tokens(contents)) as usage:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                if use_ollama():
                    from types import SimpleNamespace

                    from core.ollama_llm import ollama_chat

                    text = await asyncio.wait_for(
                        asyncio.to_thread(
                            ollama_chat,
                            [{"role": "user", "content": contents}],
                            model=model,
                            base_url=OLLAMA_HOST,
                            timeout=_CALL_TIMEOUT_SECONDS,
                        ),
                        timeout=_CALL_TIMEOUT_SECONDS,
                    )
                    response = SimpleNamespace(text=text)
                    usage["tokens"] = estimate_tokens(contents) + estimate_tokens(text)
                    return response

                # ponytail: on timeout the worker thread keeps running until the
                # socket gives up — it just no longer blocks us. Bounded by the
                # few red-team calls that use this path.
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.models.generate_content,
                        model=model, contents=contents, config=config,
                    ),
                    timeout=_CALL_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                if not _is_transient(exc) or attempt == _MAX_ATTEMPTS - 1:
                    raise
                delay = _RETRY_BASE_SECONDS * (2 ** attempt)
                rate_gate.retries += 1
                print(f"  [retry {attempt + 1}/{_MAX_ATTEMPTS - 1}] {type(exc).__name__} — waiting {delay}s")
                await asyncio.sleep(delay)
                continue
            usage["tokens"] = _usage_total(response)
            return response


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Args:
        agent: The LlmAgent instance
        runner: The InMemoryRunner instance
        user_message: Plain text message to send
        session_id: Optional session ID to continue a conversation

    Returns:
        Tuple of (response_text, session)
    """
    user_id = "student"
    app_name = runner.app_name

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    async with rate_gate.slot(estimate_tokens(user_message)) as usage:
        for attempt in range(_MAX_ATTEMPTS):
            # A retry starts from a fresh session so the failed turn is not
            # replayed as history on top of the new one.
            session = await _resolve_session(
                runner, app_name, user_id, session_id if attempt == 0 else None
            )
            collected = []

            async def consume():
                async for event in runner.run_async(
                    user_id=user_id, session_id=session.id, new_message=content
                ):
                    usage["tokens"] = _usage_total(event) or usage["tokens"]
                    if hasattr(event, "content") and event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                collected.append(part.text)

            try:
                await asyncio.wait_for(consume(), timeout=_CALL_TIMEOUT_SECONDS)
            except Exception as exc:
                if not _is_transient(exc) or attempt == _MAX_ATTEMPTS - 1:
                    raise
                delay = _RETRY_BASE_SECONDS * (2 ** attempt)
                rate_gate.retries += 1
                print(f"  [retry {attempt + 1}/{_MAX_ATTEMPTS - 1}] {type(exc).__name__} — waiting {delay}s")
                await asyncio.sleep(delay)
                continue
            return "".join(collected), session
