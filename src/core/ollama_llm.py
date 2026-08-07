"""
Thin ADK BaseLlm → local Ollama /api/chat (stdlib only, no litellm).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import AsyncGenerator, Optional

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from typing_extensions import override


def _part_text(parts) -> str:
    if not parts:
        return ""
    return "".join(p.text or "" for p in parts if getattr(p, "text", None))


def messages_from_request(llm_request: LlmRequest) -> list[dict]:
    """Convert ADK LlmRequest into Ollama chat messages."""
    messages: list[dict] = []
    cfg = llm_request.config
    si = getattr(cfg, "system_instruction", None) if cfg else None
    if si:
        text = si if isinstance(si, str) else _part_text(getattr(si, "parts", None))
        if text.strip():
            messages.append({"role": "system", "content": text})
    for content in llm_request.contents or []:
        role = "assistant" if content.role == "model" else (content.role or "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        text = _part_text(content.parts)
        if text:
            messages.append({"role": role, "content": text})
    return messages


def ollama_chat(
    messages: list[dict],
    *,
    model: str,
    base_url: str,
    timeout: float = 120.0,
) -> str:
    """POST {base_url}/api/chat, return assistant content."""
    url = f"{base_url.rstrip('/')}/api/chat"
    body = json.dumps(
        {"model": model, "messages": messages, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


class OllamaLlm(BaseLlm):
    """ADK model backed by local Ollama /api/chat."""

    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 120.0

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        return [r"ollama/.*"]

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self._maybe_append_user_content(llm_request)
        messages = messages_from_request(llm_request)
        import asyncio

        text = await asyncio.to_thread(
            ollama_chat,
            messages,
            model=self.model,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=text)],
            )
        )
