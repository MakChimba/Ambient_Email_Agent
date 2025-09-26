"""Utility tools that emit streaming progress events for demos/tests."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.tools import tool
from langgraph.runtime import Runtime, get_runtime


def _stream_writer() -> Any:
    """Return the active stream writer when available."""

    try:
        runtime: Runtime[Any] | None = get_runtime()  # type: ignore[type-arg]
    except Exception:  # noqa: BLE001 - best-effort runtime detection
        runtime = None
    writer = getattr(runtime, "stream_writer", None) if runtime else None
    if callable(writer):
        return writer
    return None


@tool
def stream_progress(
    phase: str,
    *,
    steps: int = 3,
    delay_seconds: float = 0.0,
) -> str:
    """Emit progress updates over the custom stream channel.

    Args:
        phase: Human readable phase/label for the progress update.
        steps: Number of progress updates to emit (minimum of 1).
        delay_seconds: Optional pause between events (defaults to 0 for fast tests).

    Returns:
        Summary string describing how many events were emitted.
    """

    step_count = max(1, int(steps))
    writer = _stream_writer()
    payload_base = {
        "type": "progress",
        "phase": phase,
        "total": step_count,
    }

    for index in range(step_count):
        payload = dict(payload_base, step=index + 1)
        if writer:
            try:
                writer(payload)
            except Exception:  # noqa: BLE001 - stream subscriber may not be present
                # If the writer fails (e.g., custom stream not subscribed), swallow the error
                writer = None
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    if writer:
        return f"Progress events emitted for phase '{phase}' ({step_count} steps)."
    return (
        "Progress events recorded locally; no active stream writer was available."
    )


__all__ = ["stream_progress"]
