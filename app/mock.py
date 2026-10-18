"""Scripted stand-in for the model, used for the offline demo and as the
last-resort fallback when every live provider is down or rate limited.

Only the MODEL is scripted. The verifier still runs for real against the real
document text, so retractions in mock mode are produced by the same code path
as in live mode. The UI labels mock runs clearly.
"""
from __future__ import annotations

import json
from pathlib import Path

from .verifier import normalize

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "samples" / "sample_lease.txt"
DEMO_QUESTION = "When do I get my deposit back?"

_DRAFT = (
    "Your security deposit is refunded within 30 days after you vacate the flat and return "
    "the keys, and it carries no interest. The refund is due within 7 days according to "
    "some summaries, but the lease deductions are limited to damage beyond normal wear and "
    "tear plus unpaid rent or utilities. The landlord can also deduct a cleaning fee without "
    "itemising it."
)

_CLAIMS = {"claims": [
    {"claim": "The landlord must refund the deposit within 30 days after the tenant vacates and returns the keys.",
     "quote": "The Landlord shall refund the Security Deposit, without interest, within thirty (30) days after the Tenant vacates the Premises and returns all keys"},
    {"claim": "The deposit refund carries no interest.",
     "quote": "refund the Security Deposit, without interest, within thirty (30) days"},
    {"claim": "Deductions are limited to damage beyond normal wear and tear and unpaid rent or utility charges.",
     "quote": "may deduct from the Security Deposit only repairing damage beyond normal wear and tear and any unpaid rent or utility charges"},
    {"claim": "The refund is due within 7 days of vacating.",
     "quote": "The Landlord shall refund the Security Deposit within seven (7) days of the Tenant vacating the Premises."},
    {"claim": "The landlord can deduct cleaning fees without itemising them.",
     "quote": "The Landlord must provide an itemised statement of every deduction"},
]}

_RETRY = {"The refund is due within 7 days of vacating.":
          "Deposits are returned within one week of the Tenant moving out."}

_NOT_SUPPORTED = {"The landlord can deduct cleaning fees without itemising them."}


class MockLLM:
    mode = "mock"

    def describe(self) -> list[str]:
        return ["mock:scripted-demo"]

    @staticmethod
    def supports(document: str, question: str) -> bool:
        try:
            sample = SAMPLE_PATH.read_text()
        except OSError:
            return False
        return normalize(document) == normalize(sample) and "deposit" in question.lower()

    def complete(self, system: str, user: str, stage: str = "") -> tuple[str, str, str]:
        if stage == "draft":
            out = _DRAFT
        elif stage == "claims":
            out = json.dumps(_CLAIMS)
        elif stage == "retry":
            claim = next((c for c in _RETRY if c in user), None)
            out = json.dumps({"quote": _RETRY.get(claim)})
        elif stage == "entail":
            ok = not any(c in user for c in _NOT_SUPPORTED)
            reason = ("The passage states this directly." if ok else
                      "The passage requires an itemised statement of every deduction, "
                      "which contradicts the claim.")
            out = json.dumps({"supported": ok, "reason": reason})
        else:
            out = "{}"
        return out, "mock", "scripted-demo"
