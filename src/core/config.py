"""
Lab 11 — Configuration & API Key Setup
"""
import os
from pathlib import Path

# Load .env before reading LAB_* so `python main.py` picks up local Ollama
# without exporting vars in the shell.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


# LAB_LLM=ollama → local Ollama /api/chat; LAB_LLM=google → Gemini API.
LAB_LLM = os.environ.get("LAB_LLM", "ollama").strip().lower()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

# One place decides the model, so the whole lab (agents, judges, red team)
# stays on the same backend. Override with LAB_MODEL to switch it.
_DEFAULT_MODEL = "minimax-m3:cloud" if LAB_LLM == "ollama" else "gemma-4-31b-it"
MODEL_NAME = os.environ.get("LAB_MODEL", _DEFAULT_MODEL)

# Provider quota for that model. Every LLM call in this repo goes through
# core.utils.chat_with_agent / throttled_generate, which stay inside these.
MODEL_RPM = int(os.environ.get("LAB_RPM", "30"))
MODEL_TPM = int(os.environ.get("LAB_TPM", "16000"))

_ollama_singleton = None


def use_ollama() -> bool:
    return LAB_LLM == "ollama"


def llm_ready() -> bool:
    """True when the configured backend can accept live calls."""
    if use_ollama():
        return True
    return bool(os.environ.get("GOOGLE_API_KEY"))


def lab_model():
    """Value for LlmAgent(model=...): OllamaLlm instance or Gemini model name."""
    global _ollama_singleton
    if not use_ollama():
        return MODEL_NAME
    if _ollama_singleton is None:
        from core.ollama_llm import OllamaLlm

        timeout = float(os.environ.get("LAB_CALL_TIMEOUT", "120"))
        _ollama_singleton = OllamaLlm(
            model=MODEL_NAME, base_url=OLLAMA_HOST, timeout=timeout
        )
    return _ollama_singleton


def setup_api_key(*, prompt: bool = True):
    """Prepare the configured LLM backend.

    Returns True if live LLM calls are available.
    """
    if use_ollama():
        os.environ.setdefault("OLLAMA_HOST", OLLAMA_HOST)
        print(f"Ollama ready: {OLLAMA_HOST} model={MODEL_NAME}")
        return True

    if "GOOGLE_API_KEY" not in os.environ or not os.environ.get("GOOGLE_API_KEY"):
        if prompt and sys_stdin_is_tty():
            os.environ["GOOGLE_API_KEY"] = input("Enter Google API Key: ")
        else:
            return False
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"
    print("API key loaded.")
    return True


def sys_stdin_is_tty() -> bool:
    import sys
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


# Allowed banking topics (used by topic_filter)
ALLOWED_TOPICS = [
    "banking", "account", "transaction", "transfer",
    "loan", "interest", "savings", "credit",
    "deposit", "withdrawal", "balance", "payment",
    "tai khoan", "giao dich", "tiet kiem", "lai suat",
    "chuyen tien", "the tin dung", "so du", "vay",
    "ngan hang", "atm",
]

# Blocked topics (immediate reject)
BLOCKED_TOPICS = [
    "hack", "exploit", "weapon", "drug", "illegal",
    "violence", "gambling", "bomb", "kill", "steal",
]
