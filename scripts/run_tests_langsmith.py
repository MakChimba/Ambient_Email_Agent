#!/usr/bin/env python
"""
Run pytest with LangSmith enabled so results appear in the LangSmith UI.

Usage (from repo root):
  .venv\Scripts\python scripts\run_tests_langsmith.py --agent-module=email_assistant

Relies on environment variables (prefer system env for plugin init time):
  LANGSMITH_API_KEY, LANGSMITH_TRACING=true, optional LANGSMITH_PROJECT

Falls back to loading .env for convenience, but plugin initialization
occurs before test import. For best results, set the env in the shell.
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False
    def find_dotenv(*args, **kwargs):  # type: ignore
        return ""


def main(argv: list[str]) -> int:
    # Try to load .env as a convenience, but prefer pre-set env for plugin
    load_dotenv(find_dotenv(), override=False)

    # Ensure plugin isn't disabled
    os.environ.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)

    # Prefer prod-target Gmail agent and deterministic eval mode in CI-like runs
    os.environ.setdefault("HITL_AUTO_ACCEPT", "1")
    os.environ.setdefault("EMAIL_ASSISTANT_SKIP_MARK_AS_READ", "1")
    if "EMAIL_ASSISTANT_EVAL_MODE" not in os.environ:
        os.environ["EMAIL_ASSISTANT_EVAL_MODE"] = "0"
    if os.environ.get("EMAIL_ASSISTANT_EVAL_MODE") != "1" and not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY not set; live Gemini calls will fail. Set EMAIL_ASSISTANT_EVAL_MODE=1 for offline runs.")

    # Friendly status
    project = os.getenv("LANGSMITH_PROJECT", "ambient-email-agent")
    tracing = os.getenv("LANGSMITH_TRACING", "false")
    has_key = bool(os.getenv("LANGSMITH_API_KEY"))
    print(f"LangSmith: tracing={tracing} project={project} key={'set' if has_key else 'missing'}")
    if not has_key:
        print("WARNING: LANGSMITH_API_KEY not set; UI will not receive results.")

    # Default args
    default_args = [
        "-v",
        "tests/test_response.py",
        "--agent-module=email_assistant_hitl_memory_gmail",
        # Focus on tool-calls test by default; criteria eval depends on live LLM
        "-k",
        "tool_calls",
    ]

    # Use provided args if any, else defaults
    pytest_args = argv if argv else default_args

    # Run pytest using the same interpreter
    cmd = [sys.executable, "-m", "pytest"] + pytest_args
    print("Running:", " ".join(cmd))

    # Run from repo root (folder with pyproject.toml)
    root = Path(__file__).resolve().parents[1]
    cwd = os.getcwd()
    try:
        os.chdir(root)
        return subprocess.call(cmd)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
