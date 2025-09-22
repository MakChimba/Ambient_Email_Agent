"""Helpers for configuring LangSmith tracing during tests."""

from __future__ import annotations

import os

from email_assistant.tracing import AGENT_PROJECT, JUDGE_PROJECT, init_project


def configure_tracing_project(default_project: str) -> None:
    """Backwards-compatible shim for older tests; enforces shared agent project."""

    init_project(AGENT_PROJECT)
    os.environ["LANGSMITH_PROJECT"] = AGENT_PROJECT
    os.environ["LANGCHAIN_PROJECT"] = AGENT_PROJECT


def configure_judge_project(default_project: str) -> None:
    """Ensure judge runs default to the shared judge project."""

    os.environ.setdefault("EMAIL_ASSISTANT_JUDGE_PROJECT", JUDGE_PROJECT)
