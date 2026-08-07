"""Checks for the LLM throttle in core.utils: sequencing, budgets, retries.

Run: python -m pytest tests/test_rate_gate.py
The windows are shrunk to sub-second so the whole file finishes in ~1s.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.utils import RateGate, throttled_generate  # noqa: E402
import core.utils as utils  # noqa: E402


def _gate(rpm, tpm, window=0.4):
    gate = RateGate(rpm=rpm, tpm=tpm)
    gate.WINDOW = window
    return gate


def test_rpm_limit_delays_the_extra_call():
    gate = _gate(rpm=3, tpm=10**6)

    async def run():
        start = time.monotonic()
        for _ in range(4):
            async with gate.slot(10) as usage:
                usage["tokens"] = 10
        return time.monotonic() - start

    elapsed = asyncio.run(run())
    assert elapsed >= gate.WINDOW, f"4th call at 3 RPM was not delayed ({elapsed:.2f}s)"
    assert gate.waits == 1


def test_tpm_limit_delays_on_tokens_not_request_count():
    gate = _gate(rpm=10**6, tpm=100)

    async def run():
        start = time.monotonic()
        for _ in range(3):
            async with gate.slot(60) as usage:
                usage["tokens"] = 60
        return time.monotonic() - start

    elapsed = asyncio.run(run())
    assert elapsed >= 2 * gate.WINDOW, f"TPM budget not enforced ({elapsed:.2f}s)"


def test_actual_usage_replaces_the_estimate():
    gate = _gate(rpm=10**6, tpm=10**6)

    async def run():
        async with gate.slot(5000) as usage:
            usage["tokens"] = 12

    asyncio.run(run())
    assert gate.snapshot()["tokens_last_minute"] == 12


def test_calls_never_overlap():
    gate = _gate(rpm=10**6, tpm=10**6)
    in_flight = 0
    peak = 0

    async def one():
        nonlocal in_flight, peak
        async with gate.slot(10) as usage:
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            usage["tokens"] = 10

    async def run_all():
        await asyncio.gather(*(one() for _ in range(5)))

    asyncio.run(run_all())
    assert peak == 1, f"{peak} calls were in flight at once"


def test_transient_errors_are_recognised_through_the_cause_chain():
    try:
        try:
            raise RuntimeError("503 UNAVAILABLE. This model is experiencing high demand.")
        except RuntimeError as inner:
            raise RuntimeError("Dynamic node unsafe_assistant failed") from inner
    except RuntimeError as outer:
        assert utils._is_transient(outer)
    assert not utils._is_transient(ValueError("API key not valid"))


def test_a_stalled_call_times_out_and_is_retried():
    original_timeout = utils._CALL_TIMEOUT_SECONDS
    original_base = utils._RETRY_BASE_SECONDS
    utils._CALL_TIMEOUT_SECONDS = 0.05
    utils._RETRY_BASE_SECONDS = 0
    try:
        calls = {"n": 0}

        class Client:
            class models:
                @staticmethod
                def generate_content(**kwargs):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        time.sleep(0.5)          # stalls past the timeout
                    return type("R", (), {"text": "ok", "usage_metadata": None})()

        result = asyncio.run(throttled_generate(Client(), model="m", contents="hi"))
        assert result.text == "ok", "timeout was not retried"
        assert calls["n"] == 2
    finally:
        utils._CALL_TIMEOUT_SECONDS = original_timeout
        utils._RETRY_BASE_SECONDS = original_base


def test_retry_recovers_then_gives_up_on_permanent_errors():
    original_base = utils._RETRY_BASE_SECONDS
    utils._RETRY_BASE_SECONDS = 0
    try:
        calls = {"n": 0}

        class Models:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise RuntimeError("503 UNAVAILABLE high demand")
                return type("R", (), {"text": "ok", "usage_metadata": None})()

        class Client:
            models = Models()

        result = asyncio.run(throttled_generate(Client(), model="m", contents="hi"))
        assert result.text == "ok" and calls["n"] == 3

        class Broken:
            class models:
                @staticmethod
                def generate_content(**kwargs):
                    raise ValueError("API key not valid")

        try:
            asyncio.run(throttled_generate(Broken(), model="m", contents="hi"))
            raise AssertionError("permanent error should not be retried")
        except ValueError:
            pass
    finally:
        utils._RETRY_BASE_SECONDS = original_base


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all rate gate checks passed")
