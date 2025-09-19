"""Helpers for configuring LangSmith tracing during tests."""

from __future__ import annotations

import os

TRACING_ENABLED_VALUES = {"1", "true", "yes", "on"}


def configure_tracing_project(default_project: str) -> None:
    """Set a default LangSmith project for traces when tracing is enabled."""

    tracing_flag = os.getenv("LANGSMITH_TRACING", "").lower()
    if tracing_flag not in TRACING_ENABLED_VALUES:
        return

    override = os.getenv("EMAIL_ASSISTANT_TRACE_PROJECT")
    project = override or default_project
    if not project:
        return

    os.environ["LANGSMITH_PROJECT"] = project
