"""
Assignment 11 — Monitoring & Alerts.

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def record(
        self,
        *,
        blocked: bool = False,
        rate_limited: bool = False,
        judge_fail: bool | None = None,
    ):
        """Update counters after one pipeline request."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
        if rate_limited:
            self.rate_limit_hits += 1
        if judge_fail is not None:
            self.judge_checks += 1
            if judge_fail:
                self.judge_fails += 1

    def check_metrics(self) -> list[Alert]:
        """Compute rates and append Alert objects when thresholds exceeded."""
        self.alerts = []
        snap = self.snapshot()

        if snap["block_rate"] > self.block_rate_threshold and self.total_requests >= 5:
            self.alerts.append(Alert(
                metric="block_rate",
                value=snap["block_rate"],
                threshold=self.block_rate_threshold,
                message=f"Block rate {snap['block_rate']:.0%} exceeds "
                        f"{self.block_rate_threshold:.0%} — possible attack wave.",
            ))

        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            self.alerts.append(Alert(
                metric="rate_limit_hits",
                value=float(self.rate_limit_hits),
                threshold=float(self.rate_limit_hit_threshold),
                message=f"Rate-limit hits ({self.rate_limit_hits}) >= "
                        f"{self.rate_limit_hit_threshold}.",
            ))

        if snap["judge_fail_rate"] > self.judge_fail_rate_threshold and self.judge_checks >= 3:
            self.alerts.append(Alert(
                metric="judge_fail_rate",
                value=snap["judge_fail_rate"],
                threshold=self.judge_fail_rate_threshold,
                message=f"Judge FAIL rate {snap['judge_fail_rate']:.0%} exceeds "
                        f"{self.judge_fail_rate_threshold:.0%}.",
            ))

        return self.alerts

    def export_json(self, filepath: str = "outputs/metrics.json"):
        """Write metrics + alerts to JSON."""
        self.check_metrics()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
