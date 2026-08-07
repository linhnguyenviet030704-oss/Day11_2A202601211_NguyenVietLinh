"""ponytail: smallest check that Ollama message mapping stays correct."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from core.ollama_llm import messages_from_request  # noqa: E402


def test_messages_from_request_system_and_user():
    req = SimpleNamespace(
        config=SimpleNamespace(system_instruction="You are VinBank."),
        contents=[
            SimpleNamespace(
                role="user",
                parts=[SimpleNamespace(text="What is the savings rate?")],
            )
        ],
    )
    assert messages_from_request(req) == [
        {"role": "system", "content": "You are VinBank."},
        {"role": "user", "content": "What is the savings rate?"},
    ]


def test_messages_maps_model_role_to_assistant():
    req = SimpleNamespace(
        config=None,
        contents=[
            SimpleNamespace(
                role="model",
                parts=[SimpleNamespace(text="Hello")],
            )
        ],
    )
    assert messages_from_request(req) == [
        {"role": "assistant", "content": "Hello"},
    ]


if __name__ == "__main__":
    test_messages_from_request_system_and_user()
    test_messages_maps_model_role_to_assistant()
    print("ok")
