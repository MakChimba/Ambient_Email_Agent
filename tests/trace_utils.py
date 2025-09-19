"""Helpers for configuring LangSmith tracing during tests."""

from __future__ import annotations

import os

TRACING_ENABLED_VALUES = {"1", "true", "yes", "on"}


def configure_tracing_project(default_project: str) -> None:
    """Set a default LangSmith project for traces when tracing is enabled."""

    tracing_flag = os.getenv("LANGSMITH_TRACING", "").lower()
    if tracing_flag not in TRACING_ENABLED_VALUES:
        return

    if os.getenv("LANGSMITH_PROJECT"):
        return

    override = os.getenv("EMAIL_ASSISTANT_TRACE_PROJECT")
    project = override or default_project
    if not project:
        return

    os.environ["LANGSMITH_PROJECT"] = project


def configure_judge_project(default_project: str) -> None:
    """Set default project for judge traces when LLM judge is active."""

    judge_flag = os.getenv("EMAIL_ASSISTANT_LLM_JUDGE", "").lower()
    if judge_flag not in TRACING_ENABLED_VALUES:
        return

    override = os.getenv("EMAIL_ASSISTANT_JUDGE_PROJECT_OVERRIDE")
    project = override or os.getenv("EMAIL_ASSISTANT_JUDGE_PROJECT") or default_project
    if not project:
        return

    os.environ["EMAIL_ASSISTANT_JUDGE_PROJECT"] = project
