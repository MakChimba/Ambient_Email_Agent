"""Sender reputation and reminder risk helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple


PROFILE_NAMESPACE = ("email_assistant", "sender_reputation")
PROFILE_KEY = "profile"

_MONEY_KEYWORDS = [
    "invoice",
    "payment",
    "wire",
    "transfer",
    "bank",
    "crypto",
    "bitcoin",
    "paypal",
    "urgent",
    "overdue",
    "bill",
    "due",
]


@dataclass
class SenderAssessment:
    email: str
    status: str
    risk_level: str
    reason: str


def _load_profile(store) -> Dict[str, Dict[str, Dict[str, str]]]:
    try:
        entry = store.get(PROFILE_NAMESPACE, PROFILE_KEY)
        if entry and getattr(entry, "value", None):
            return json.loads(entry.value)
    except Exception:
        pass
    return {"known": {}, "flagged": {}}


def _save_profile(store, profile: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    try:
        store.put(PROFILE_NAMESPACE, PROFILE_KEY, json.dumps(profile))
    except Exception:
        pass


def _extract_email(address: str | None) -> str:
    if not address:
        return ""
    match = re.search(r"<([^>]+)>", address)
    if match:
        return match.group(1).strip().lower()
    return address.strip().lower()


def assess_sender(store, author: str | None, subject: str, body: str) -> SenderAssessment:
    profile = _load_profile(store)
    email = _extract_email(author)
    if not email:
        return SenderAssessment(email="", status="unknown", risk_level="high", reason="Missing sender address")

    now_iso = datetime.now(timezone.utc).isoformat()
    record = profile["known"].get(email) or profile["flagged"].get(email)
    status = record.get("status") if record else "new"

    lower_subject = (subject or "").lower()
    lower_body = (body or "").lower()
    text = f"{lower_subject}\n{lower_body}"

    risk_level = "low"
    reason = "Known sender"

    money_hit = any(keyword in text for keyword in _MONEY_KEYWORDS)
    if status == "new":
        if money_hit:
            risk_level = "high"
            reason = "New sender requesting financial action"
        else:
            risk_level = "medium"
            reason = "New sender"
    elif status == "flagged":
        risk_level = "high"
        reason = record.get("reason", "Previously flagged sender")
    else:
        reason = record.get("reason", "Known sender")

    profile.setdefault("last_seen", {})[email] = now_iso
    _save_profile(store, profile)

    return SenderAssessment(email=email, status=status, risk_level=risk_level, reason=reason)


def note_sender(store, email: str, status: str, reason: str | None = None) -> None:
    if not email:
        return
    profile = _load_profile(store)
    entry = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    if reason:
        entry["reason"] = reason
    if status in {"trusted", "known"}:
        profile.setdefault("known", {})[email] = entry
        profile.get("flagged", {}).pop(email, None)
    elif status == "flagged":
        profile.setdefault("flagged", {})[email] = entry
        profile.get("known", {}).pop(email, None)
    _save_profile(store, profile)


def sender_exists(store, email: str) -> bool:
    profile = _load_profile(store)
    return email in profile.get("known", {}) or email in profile.get("flagged", {})


def sender_profile_snapshot(store) -> Dict[str, Dict[str, Dict[str, str]]]:
    return _load_profile(store)


def judge_disabled() -> bool:
    forced = os.getenv("REMINDER_JUDGE_FORCE_DECISION")
    if forced:
        return False
    use_llm = os.getenv("EMAIL_ASSISTANT_LLM_JUDGE", "").lower() in ("1", "true", "yes")
    return not use_llm
