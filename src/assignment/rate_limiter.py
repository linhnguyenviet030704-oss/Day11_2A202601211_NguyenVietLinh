"""
Assignment 11 — Rate Limiter.

Sliding-window, per-user rate limiting. Blocks flooding / cost attacks
that other guardrail layers do not address.
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0

    def allow(self, user_id: str = "anonymous") -> tuple[bool, str | None]:
        """Pure-Python check used by DefensePipeline (no ADK required).

        Returns (allowed, block_message_or_None).
        """
        self.total_count += 1
        now = time.time()
        window = self.user_windows[user_id]
        cutoff = now - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            self.blocked_count += 1
            return False, f"Rate limit exceeded. Try again in {wait:.0f}s."

        window.append(now)
        return True, None

    def reset(self, user_id: str | None = None):
        """Clear windows (useful between test suites)."""
        if user_id is None:
            self.user_windows.clear()
        else:
            self.user_windows.pop(user_id, None)

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        allowed, message = self.allow(user_id)
        if not allowed:
            return self._block_response(message)
        return None
