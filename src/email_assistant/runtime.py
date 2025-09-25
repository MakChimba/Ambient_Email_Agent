"""Runtime context utilities shared across agents."""
from __future__ import annotations

import os
from typing import Any

from langgraph.runtime import Runtime

from email_assistant.schemas import AssistantContext

_DEFAULT_TIMEZONE = os.getenv("EMAIL_ASSISTANT_TIMEZONE", "Australia/Melbourne")


def extract_runtime_metadata(
    runtime: Runtime[AssistantContext] | None,
) -> tuple[str, bool, str | None, dict[str, Any]]:
    """Return timezone/eval flags/thread id metadata for tracing.

    Args:
        runtime: LangGraph runtime provided to the node.

    Returns:
        A tuple of (timezone, eval_mode, thread_id, metadata_dict).
    """

    context = (runtime.context if runtime and runtime.context else {})  # type: ignore[attr-defined]
    timezone = context.get("timezone", _DEFAULT_TIMEZONE)
    eval_mode = bool(context.get("eval_mode", False))
    thread_metadata = context.get("thread_metadata") or {}
    thread_id = context.get("thread_id") or thread_metadata.get("thread_id")
    metadata = {"timezone": timezone, "eval_mode": eval_mode}
    return timezone, eval_mode, thread_id, metadata


def runtime_thread_id(runtime: Runtime[AssistantContext] | None) -> str | None:
    """Convenience accessor for the current thread id."""

    if not runtime:
        return None
    context = runtime.context or {}
    thread_metadata = context.get("thread_metadata") or {}
    return context.get("thread_id") or thread_metadata.get("thread_id")
