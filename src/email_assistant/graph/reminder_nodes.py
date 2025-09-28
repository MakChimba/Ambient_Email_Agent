"""Reminder graph nodes for LangGraph workflows."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Tuple

from langgraph.types import Command

from email_assistant.tools.reminders import ReminderStore, get_default_store


logger = logging.getLogger(__name__)

_REMINDER_STORE: ReminderStore | None = None
_PENDING_ACTIONS: Dict[str, List[Dict[str, Any]]] = {}
_PENDING_NEXT: Dict[str, str] = {}


def set_reminder_store(store: ReminderStore) -> None:
    """Override the reminder store used by the graph nodes."""

    global _REMINDER_STORE
    _REMINDER_STORE = store


def _get_store() -> ReminderStore:
    global _REMINDER_STORE
    if _REMINDER_STORE is None:
        _REMINDER_STORE = get_default_store()
    return _REMINDER_STORE


def register_reminder_actions(thread_id: str, actions: List[Dict[str, Any]], next_node: str) -> None:
    if not thread_id:
        thread_id = "__default__"
    if not actions:
        return
    existing = _PENDING_ACTIONS.get(thread_id, [])
    _PENDING_ACTIONS[thread_id] = existing + actions
    _PENDING_NEXT[thread_id] = next_node


def consume_reminder_actions(thread_id: str) -> List[Dict[str, Any]]:
    if not thread_id:
      thread_id = "__default__"
    return _PENDING_ACTIONS.pop(thread_id, [])


def peek_reminder_actions(thread_id: str) -> List[Dict[str, Any]]:
    if not thread_id:
        thread_id = "__default__"
    return list(_PENDING_ACTIONS.get(thread_id, []))


def _parse_due_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return datetime.now(timezone.utc)
        if trimmed.endswith("Z"):
            trimmed = trimmed[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(trimmed)
        except ValueError:
            logger.warning("Reminder node: invalid due_at string '%s'; using current time", value)
            return datetime.now(timezone.utc)
    else:
        logger.warning("Reminder node: unsupported due_at type %s; using current time", type(value).__name__)
        return datetime.now(timezone.utc)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _partition_actions(state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    thread_id = _thread_key(state)
    actions = peek_reminder_actions(thread_id)
    cancels: List[Dict[str, Any]] = []
    creates: List[Dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        kind = str(action.get("action"))
        if kind == "cancel":
            cancels.append(action)
        elif kind == "create":
            creates.append(action)
    return cancels, creates


def cancel_reminder_node(
    state: Dict[str, Any],
    store: Any = None,
    runtime: Any = None,
) -> Command[Literal["create_reminder_node", "response_agent", "triage_interrupt_handler", "__end__"]]:
    """Cancel active reminders requested in the state and route to the next node."""

    cancels, creates = _partition_actions(state)
    thread_id = _thread_key(state)
    next_node = _PENDING_NEXT.get(thread_id, state.get("reminder_next_node") or "response_agent")
    if not cancels:
        if creates:
            logger.info("Reminder node: routing %d create action(s) to create_reminder_node", len(creates))
        goto: Literal["create_reminder_node", "response_agent", "triage_interrupt_handler", "__end__"]
        goto = "create_reminder_node" if creates else next_node
        update: Dict[str, Any] = {
            "reminder_next_node": next_node,
        }
        return Command(goto=goto, update=update)

    reminder_store = _get_store()
    for action in cancels:
        thread_id = action.get("thread_id")
        if not thread_id:
            logger.warning("Reminder node: cancel action missing thread_id: %s", action)
            continue
        try:
            cancelled = reminder_store.cancel_reminder(thread_id)
            if cancelled:
                logger.info("Reminder node: cancelled %s reminder(s) for thread %s", cancelled, thread_id)
            else:
                logger.info("Reminder node: no active reminder to cancel for thread %s", thread_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Reminder node: failed to cancel reminder for %s: %s", thread_id, exc)

    remaining = [action for action in peek_reminder_actions(thread_id) if action.get("action") == "create"]
    if remaining:
        key = thread_id if thread_id else "__default__"
        _PENDING_ACTIONS[key] = remaining
        _PENDING_NEXT[key] = next_node
    else:
        key = thread_id if thread_id else "__default__"
        _PENDING_ACTIONS.pop(key, None)
        _PENDING_NEXT.pop(key, None)

    goto = "create_reminder_node" if remaining else next_node
    update = {
        "reminder_next_node": next_node,
    }
    return Command(goto=goto, update=update)


def create_reminder_node(
    state: Dict[str, Any],
    store: Any = None,
    runtime: Any = None,
) -> Command[Literal["response_agent", "triage_interrupt_handler", "__end__"]]:
    """Create reminders requested in the state and route to the next workflow node."""

    thread_id = _thread_key(state)
    next_node = _PENDING_NEXT.pop(thread_id, state.get("reminder_next_node") or "response_agent")
    actions = consume_reminder_actions(thread_id)
    creates = [action for action in actions if action.get("action") == "create"]
    if not creates:
        logger.info("Reminder node: no create actions; forwarding to %s", next_node)
        update = {
            "reminder_next_node": None,
        }
        return Command(goto=next_node, update=update)

    reminder_store = _get_store()
    logger.info("Reminder node: processing %d create action(s)", len(creates))
    for action in creates:
        thread_id = action.get("thread_id")
        if not thread_id:
            logger.warning("Reminder node: create action missing thread_id: %s", action)
            continue
        subject = str(action.get("subject") or "(no subject)")
        due_at = _parse_due_at(action.get("due_at"))
        reason = str(action.get("reason") or "Reminder created via graph node")
        try:
            reminder_id = reminder_store.add_reminder(
                thread_id=thread_id,
                subject=subject,
                due_at=due_at,
                reason=reason,
            )
            logger.info(
                "Reminder node: ensured reminder %s for thread %s due at %s",
                reminder_id,
                thread_id,
                due_at.isoformat(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Reminder node: failed to create reminder for %s: %s", thread_id, exc)

    update = {
        "reminder_next_node": None,
    }
    return Command(goto=next_node, update=update)


def _thread_key(state: Dict[str, Any]) -> str:
    email_input = state.get("email_input") or {}
    thread_id = (
        email_input.get("id")
        or email_input.get("thread_id")
        or email_input.get("gmail_id")
        or state.get("thread_id")
    )
    return str(thread_id or "__default__")

