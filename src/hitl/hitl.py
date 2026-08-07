"""
Lab 11 — Part 4: Human-in-the-Loop Design
  ConfidenceRouter + 3 banking HITL decision points.
"""
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses by confidence and action risk.

    High-risk banking actions always escalate — confidence alone is not enough
    when money or account ownership is at stake.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Return routing decision for an agent response."""
        if action_type in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason=f"High-risk action: {action_type}",
                priority="high",
                requires_human=True,
            )

        if confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=confidence,
                reason="High confidence",
                priority="low",
                requires_human=False,
            )

        if confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=confidence,
                reason="Medium confidence — needs review",
                priority="normal",
                requires_human=True,
            )

        return RoutingDecision(
            action="escalate",
            confidence=confidence,
            reason="Low confidence — escalating",
            priority="high",
            requires_human=True,
        )


hitl_decision_points = [
    {
        "id": 1,
        "name": "Large money transfer approval",
        "trigger": "Customer requests a transfer above a configured threshold "
                   "(e.g. 50,000,000 VND) or to a new beneficiary.",
        "hitl_model": "human-in-the-loop",
        "context_needed": "Transfer amount, source/destination accounts, "
                          "KYC status, fraud-score, recent login location.",
        "example": "User asks to transfer 200M VND to a newly added account — "
                   "agent drafts confirmation but a bank officer must approve before execution.",
        "approval_path": "Approve → gateway executes with the recorded approval_id. "
                         "Reject → customer gets a branch-visit message and nothing is sent. "
                         "Timeout after 15 min → request expires unapproved (fail closed); "
                         "the transfer is never auto-executed.",
        "audit_fields": "correlation_id, user_id, action_type, intent, "
                        "diff{before,after}, reviewer_id, decision, decided_at, note.",
    },
    {
        "id": 2,
        "name": "Account closure / irreversible change",
        "trigger": "Requests to close an account, change password via chat, "
                   "or delete personal data.",
        "hitl_model": "human-in-the-loop",
        "context_needed": "Account balance, open products, identity verification "
                          "artifacts, reason for closure.",
        "example": "User says 'close my joint savings account' — escalate to branch "
                   "staff who verify both account holders before proceeding.",
        "approval_path": "Approve → closure queued only after both holders verify. "
                         "Reject → agent explains the missing verification. "
                         "Timeout after 15 min → expires unapproved; the account stays open.",
        "audit_fields": "correlation_id, user_id, action_type, intent, "
                        "diff{account_state_before,account_state_after}, reviewer_id, "
                        "decision, decided_at, note.",
    },
    {
        "id": 3,
        "name": "Security incident / credential probe",
        "trigger": "Input/output guardrails detect repeated injection attempts "
                   "or suspected secret extraction against the chatbot.",
        "hitl_model": "human-on-the-loop",
        "context_needed": "Audit log excerpts, attack patterns, user_id, "
                          "rate-limit hits, blocked layer names.",
        "example": "Same user hits rate limit then sends 5 injection prompts — "
                   "SOC analyst reviews and may lock the chat session.",
        "approval_path": "Approve (lock) → session terminated and user flagged. "
                         "Reject (false positive) → guardrail pattern tuned, session resumes. "
                         "Timeout after 15 min → session stays rate-limited and the "
                         "case escalates to the next SOC tier rather than closing itself.",
        "audit_fields": "correlation_id, user_id, action_type, intent, "
                        "diff{blocked_layers,attack_categories}, reviewer_id, "
                        "decision, decided_at, note.",
    },
]


@dataclass
class ReviewRequest:
    """One proposed high-risk action waiting on a human decision."""

    correlation_id: str
    user_id: str
    action_type: str
    intent: str
    diff: dict
    context: dict
    created_at: float
    status: str = "pending"          # pending | approved | rejected | timeout
    reviewer_id: str | None = None
    decided_at: str | None = None
    note: str = ""

    def to_audit(self) -> dict:
        """Flatten to the audit fields named in hitl_decision_points."""
        return {
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "action_type": self.action_type,
            "intent": self.intent,
            "diff": self.diff,
            "context": self.context,
            "reviewer_id": self.reviewer_id,
            "decision": self.status,
            "decided_at": self.decided_at,
            "note": self.note,
        }


class HITLQueue:
    """Hold high-risk actions until a human approves, rejects, or lets them expire.

    Fail closed on purpose: only ``approved`` authorises execution, so an
    unreviewed request that times out is treated exactly like a rejection. A
    reviewer going home must never become an implicit approval.
    """

    TIMEOUT_SECONDS = 900

    def __init__(self, timeout_seconds: int | None = None):
        # `or` would swallow a deliberate 0-second SLA used in tests/demos.
        self.timeout_seconds = (
            self.TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        self.requests: dict[str, ReviewRequest] = {}

    @staticmethod
    def new_correlation_id() -> str:
        """Format matches agents.security_boundary's approval_id contract."""
        return f"HITL-{uuid.uuid4().hex[:8].upper()}"

    def submit(
        self,
        *,
        action_type: str,
        intent: str,
        diff: dict,
        context: dict,
        user_id: str = "unknown",
    ) -> ReviewRequest:
        """Queue a proposed action and hand the reviewer intent + diff + context."""
        request = ReviewRequest(
            correlation_id=self.new_correlation_id(),
            user_id=user_id,
            action_type=action_type,
            intent=intent,
            diff=diff,
            context=context,
            created_at=time.time(),
        )
        self.requests[request.correlation_id] = request
        return request

    def _decide(self, correlation_id: str, status: str, reviewer_id: str, note: str):
        request = self.requests.get(correlation_id)
        if request is None:
            raise KeyError(f"Unknown correlation_id: {correlation_id}")
        if request.status != "pending":
            return request          # already decided or expired — never re-open
        request.status = status
        request.reviewer_id = reviewer_id
        request.decided_at = datetime.now(timezone.utc).isoformat()
        request.note = note
        return request

    def approve(self, correlation_id: str, reviewer_id: str, note: str = "") -> ReviewRequest:
        return self._decide(correlation_id, "approved", reviewer_id, note)

    def reject(self, correlation_id: str, reviewer_id: str, note: str = "") -> ReviewRequest:
        return self._decide(correlation_id, "rejected", reviewer_id, note)

    def expire_due(self, now: float | None = None) -> list[ReviewRequest]:
        """Mark pending requests older than the SLA as timed out."""
        now = now if now is not None else time.time()
        expired = []
        for request in self.requests.values():
            if request.status == "pending" and now - request.created_at >= self.timeout_seconds:
                request.status = "timeout"
                request.decided_at = datetime.now(timezone.utc).isoformat()
                request.note = f"No reviewer decision within {self.timeout_seconds}s"
                expired.append(request)
        return expired

    def is_authorized(self, correlation_id: str) -> bool:
        """Only an explicit approval authorises the action."""
        request = self.requests.get(correlation_id)
        return bool(request and request.status == "approved")

    def audit_records(self) -> list[dict]:
        return [r.to_audit() for r in self.requests.values()]


def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<18} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 80)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<25} {conf:<6.2f} {action_type:<18} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 80)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed']}")
        print(f"    Example:  {point['example']}")
        print(f"    Approval: {point['approval_path']}")
        print(f"    Audit:    {point['audit_fields']}")
    print("\n" + "=" * 60)


def demo_review_lifecycle() -> list[dict]:
    """Walk one approve, one reject and one timeout through HITLQueue."""
    queue = HITLQueue(timeout_seconds=0)   # 0s so the timeout case is observable offline

    approved = queue.submit(
        user_id="cust_001",
        action_type="transfer_money",
        intent="Transfer 200,000,000 VND to a beneficiary added 5 minutes ago",
        diff={"before": {"balance": 350_000_000}, "after": {"balance": 150_000_000}},
        context={"kyc": "verified", "fraud_score": 0.31, "login_city": "Hanoi"},
    )
    rejected = queue.submit(
        user_id="cust_002",
        action_type="close_account",
        intent="Close joint savings account without the second holder present",
        diff={"before": {"status": "active"}, "after": {"status": "closed"}},
        context={"holders": 2, "verified_holders": 1, "balance": 12_000_000},
    )
    abandoned = queue.submit(
        user_id="cust_003",
        action_type="change_password",
        intent="Reset the online-banking password over chat",
        diff={"before": {"password_age_days": 412}, "after": {"password_age_days": 0}},
        context={"channel": "chat", "identity_documents": "missing"},
    )

    queue.approve(approved.correlation_id, reviewer_id="officer-17", note="Fraud score acceptable")
    queue.reject(rejected.correlation_id, reviewer_id="branch-04", note="Second holder absent")
    queue.expire_due()

    print("\nHITL review lifecycle (approve / reject / timeout):")
    print("=" * 60)
    for record in queue.audit_records():
        print(f"  {record['correlation_id']}  {record['action_type']:<18} "
              f"{record['decision']:<9} reviewer={record['reviewer_id']}")
    print(f"  Authorized to execute: "
          f"{[r['correlation_id'] for r in queue.audit_records() if r['decision'] == 'approved']}")
    print("=" * 60)

    assert queue.is_authorized(approved.correlation_id) is True
    assert queue.is_authorized(rejected.correlation_id) is False
    assert queue.requests[abandoned.correlation_id].status == "timeout"
    assert queue.is_authorized(abandoned.correlation_id) is False, "timeout must fail closed"
    return queue.audit_records()


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
    demo_review_lifecycle()
