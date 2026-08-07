"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._pending: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id/user_id."""
        key = request_id or f"{user_id}:{len(self.logs)}:{time.time_ns()}"
        self._pending[key] = {
            "user_id": user_id,
            "input": text,
            "request_id": key,
            "timestamp": utc_now_iso(),
            "start": time.time(),
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        meta = None
        if request_id and request_id in self._pending:
            meta = self._pending.pop(request_id)
        else:
            # Fall back to most recent pending entry for this user
            for k, v in reversed(list(self._pending.items())):
                if v.get("user_id") == user_id:
                    meta = self._pending.pop(k)
                    break

        if meta is None:
            meta = {
                "user_id": user_id,
                "input": "",
                "request_id": request_id or f"{user_id}:{len(self.logs)}",
                "timestamp": utc_now_iso(),
                "start": time.time(),
            }

        latency_ms = round((time.time() - meta["start"]) * 1000, 2)
        self.logs.append({
            "request_id": meta["request_id"],
            "user_id": user_id,
            "timestamp": meta["timestamp"],
            "input": meta["input"],
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": latency_ms,
        })

    def export_json(self, filepath: str = "outputs/audit_log.json"):
        """Write logs to disk (JSON array)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
