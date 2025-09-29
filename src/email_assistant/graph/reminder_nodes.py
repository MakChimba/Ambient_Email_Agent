"""Reminder graph nodes for LangGraph workflows."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Sequence

from langgraph.types import Command

from email_assistant.tools.reminders import ReminderStore, get_default_store


logger = logging.getLogger(__name__)

_DEFAULT_THREAD_KEY = "__default__"
_REMINDER_STORE: ReminderStore | None = None


def set_reminder_store(store: ReminderStore) -> None:
    """Override the reminder store used by the graph nodes."""

    global _REMINDER_STORE
    _REMINDER_STORE = store


def _get_store() -> ReminderStore:
    global _REMINDER_STORE
    if _REMINDER_STORE is None:
        _REMINDER_STORE = get_default_store()
    return _REMINDER_STORE


def _normalise_thread_id(thread_id: object) -> str:
    if not thread_id:
        return _DEFAULT_THREAD_KEY
    return str(thread_id)


def resolve_thread_key(state: Dict[str, Any]) -> str:
    """Derive a stable reminder key from the current state."""

    email_input = state.get("email_input") or {}
    candidates = [
        email_input.get("thread_id"),
        email_input.get("threadId"),
        email_input.get("gmail_thread_id"),
        email_input.get("gmailThreadId"),
        state.get("reminder_thread_id"),
        email_input.get("id"),
        state.get("thread_id"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return _DEFAULT_THREAD_KEY


def stage_reminder_actions(
    thread_id: object,
    actions: Sequence[Dict[str, Any]],
    next_node: str,
    *,
    origin: str | None = None,
) -> Dict[str, Any]:
    """Normalise reminder actions prior to dispatch."""

    canonical_thread = _normalise_thread_id(thread_id)
    staged: List[Dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        kind = str(action.get("action", "")).lower()
        if kind not in {"cancel", "create"}:
            continue
        payload = dict(action)
        payload["action"] = kind
        payload["thread_id"] = _normalise_thread_id(payload.get("thread_id") or canonical_thread)
        staged.append(payload)

    return {
        "reminder_actions": staged,
        "reminder_thread_id": canonical_thread,
        "reminder_next_node": next_node,
        "reminder_dispatch_origin": (origin or "unknown").lower(),
    }


def register_reminder_actions(
    thread_id: object,
    actions: Sequence[Dict[str, Any]],
    next_node: str,
    *,
    origin: str | None = None,
) -> Dict[str, Any]:
    """Backward-compatible alias for staging reminder actions."""

    return stage_reminder_actions(thread_id, actions, next_node, origin=origin)


def apply_reminder_actions_node(
    state: Dict[str, Any],
    store: Any = None,
    runtime: Any = None,
) -> Command[Literal["response_agent", "triage_interrupt_handler", "__end__"]]:
    """Apply pending reminder actions atomically and route to the next node."""

    next_node = state.get("reminder_next_node")
    actions = list(state.get("reminder_actions") or [])
    thread_key = _normalise_thread_id(state.get("reminder_thread_id") or resolve_thread_key(state))
    origin = (state.get("reminder_dispatch_origin") or "unknown").lower()
    classification = (state.get("classification_decision") or "respond").lower()
    triage_done = bool(state.get("triage_interrupt_completed"))

    def _fallback_target() -> str:
        if next_node and next_node != "apply_reminder_actions_node":
            return next_node
        if classification == "notify" and not triage_done:
            return "triage_interrupt_handler"
        if classification == "ignore":
            return "__end__"
        return "response_agent"

    if not actions:
        update = {
            "reminder_actions": [],
            "reminder_thread_id": None,
            "reminder_next_node": None,
            "reminder_dispatch_origin": None,
        }
        safe_next = _fallback_target()
        return Command(goto=safe_next, update=update)

    reminder_store = _get_store()
    try:
        outcome = reminder_store.apply_actions(actions)
        logger.info(
            "Reminder dispatcher: applied %d action(s) for thread %s (cancelled=%s created=%s)",
            len(actions),
            thread_key,
            outcome.get("cancelled") if isinstance(outcome, dict) else None,
            outcome.get("created") if isinstance(outcome, dict) else None,
        )
        if store is not None and thread_key:
            try:
                store.delete(("reminders", "pending_actions"), thread_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Reminder dispatcher: failed clearing pending actions for %s: %s",
                    thread_key,
                    exc,
                )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reminder dispatcher: failed to apply actions for %s", thread_key)
        raise

    target = _fallback_target()
    has_create = any(isinstance(a, dict) and str(a.get("action", "")).lower() == "create" for a in actions)
    if origin == "triage_hitl_response" and has_create:
        target = "response_agent"
    update = {
        "reminder_actions": [],
        "reminder_thread_id": None,
        "reminder_next_node": None,
        "reminder_dispatch_outcome": outcome,
        "reminder_dispatch_origin": None,
    }
    return Command(goto=target, update=update)
