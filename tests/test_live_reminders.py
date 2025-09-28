import logging
import os
import warnings
from datetime import datetime, timezone, timedelta

import pytest

from tests.agent_test_utils import compile_agent, has_google_key, is_eval_mode
from tests.trace_utils import configure_tracing_project, configure_judge_project
from email_assistant.tracing import (
    invoke_with_root_run,
    summarize_email_for_grid,
)
from email_assistant.utils import format_messages_string
from email_assistant.eval.judges import (
    run_correctness_judge,
    build_tool_call_context,
    serialise_messages,
    JudgeUnavailableError,
)
from email_assistant.eval.reminder_run_judge import run_reminder_run_judge
from email_assistant.eval.composite import run_composite_judge

try:  # LangSmith logging is optional in offline runs
    from langsmith import testing as t
except Exception:  # pragma: no cover - logging disabled when LangSmith SDK missing
    t = None


def _safe_log_inputs(payload, run_id):
    if not t:
        return
    try:
        if run_id:
            t.log_inputs(payload, run_id=run_id)
        else:
            t.log_inputs(payload)
    except Exception:  # pragma: no cover
        pass


def _safe_log_outputs(payload, run_id):
    if not t:
        return
    try:
        if run_id:
            t.log_outputs(payload, run_id=run_id)
        else:
            t.log_outputs(payload)
    except Exception:  # pragma: no cover
        pass


def _extract_values(state):
    if hasattr(state, "values"):
        return state.values
    return state

@pytest.fixture(autouse=True)
def _configure_live_reminder_env(monkeypatch):
    logging.getLogger("email_assistant.graph.reminder_nodes").setLevel(logging.INFO)
    configure_tracing_project("email-assistant-live-reminders")
    configure_judge_project("email-assistant-judge-live-reminders")
    monkeypatch.setenv("EMAIL_ASSISTANT_LLM_JUDGE", "1")
    monkeypatch.setenv("EMAIL_ASSISTANT_JUDGE_STRICT", "0")
    monkeypatch.setenv("HITL_AUTO_ACCEPT", "1")
    monkeypatch.setenv("EMAIL_ASSISTANT_SKIP_MARK_AS_READ", "1")
    if not has_google_key() and not is_eval_mode():
        monkeypatch.setenv("EMAIL_ASSISTANT_EVAL_MODE", "1")


def test_live_reminder_create_and_cancel(agent_module_name, monkeypatch, gmail_service, tmp_path_factory):
    if "gmail" not in agent_module_name:
        pytest.skip("Live reminder flow is specific to the Gmail agent")

    reminder_db = tmp_path_factory.mktemp("reminder-db") / "reminders.sqlite"
    monkeypatch.setenv("REMINDER_DB_PATH", str(reminder_db))
    monkeypatch.setenv("REMINDER_NOTIFY_EMAIL", "assistant@example.com")
    monkeypatch.setenv("REMINDER_DEFAULT_HOURS", "24")
    monkeypatch.setenv("REMINDER_JUDGE_FORCE_DECISION", "hitl")

    email_assistant, thread_config, _, module = compile_agent(agent_module_name)
    run_id = thread_config.get("run_id")

    first_email = {
        "from": "Utility Billing <billing@example.com>",
        "to": "Assistant <assistant@example.com>",
        "subject": "Bill due October 1",
        "body": (
            "Hello,\n\nYour September electricity invoice ($120.45) is due on 1 October at 6:00 PM. "
            "Avoid late fees by paying before the deadline."
        ),
        "id": "thread-reminder-invoice",
    }

    payload = {"email_input": first_email}
    summary = summarize_email_for_grid(first_email)
    _safe_log_inputs({"case": "reminder_create", "email": first_email}, run_id)

    def _invoke_initial():
        return email_assistant.invoke(payload, config=thread_config, durability="sync")

    invoke_with_root_run(_invoke_initial, root_name="agent:reminder:create", input_summary=summary)
    initial_state = _extract_values(email_assistant.get_state(thread_config))

    reminders = module.reminder_store.iter_active_reminders()
    assert not reminders, "High-risk reminder should defer to HITL reviewer"
    creation_snapshot = [
        {
            "thread_id": r.thread_id,
            "subject": r.subject,
            "due_at": getattr(r.due_at, "isoformat", lambda: "")(),
            "reason": r.reason,
            "status": r.status,
        }
        for r in reminders
    ]

    _safe_log_outputs(
        {
            "case": "reminder_hitl",
            "assistant_reply": initial_state.get("assistant_reply"),
            "tool_trace": format_messages_string(initial_state.get("messages", [])),
            "reminders": creation_snapshot,
        },
        run_id,
    )

    # Judge approves on second attempt
    monkeypatch.setenv("REMINDER_JUDGE_FORCE_DECISION", "approve")

    followup_email = dict(first_email)
    followup_email["id"] = "thread-reminder-invoice-approval"

    followup_payload = {"email_input": followup_email}
    followup_summary = summarize_email_for_grid(followup_email)
    _safe_log_inputs({"case": "reminder_create", "email": followup_email}, run_id)

    def _invoke_followup():
        return email_assistant.invoke(followup_payload, config=thread_config, durability="sync")

    invoke_with_root_run(_invoke_followup, root_name="agent:reminder:create:approve", input_summary=followup_summary)
    followup_state = _extract_values(email_assistant.get_state(thread_config))

    reminders = module.reminder_store.iter_active_reminders()
    assert any(r.thread_id == followup_email["id"] for r in reminders)
    due_delta = reminders[0].due_at - datetime.now(timezone.utc)
    assert timedelta(hours=23) <= due_delta <= timedelta(hours=25)
    creation_snapshot = [
        {
            "thread_id": r.thread_id,
            "subject": r.subject,
            "due_at": r.due_at.isoformat(),
            "reason": r.reason,
            "status": r.status,
        }
        for r in reminders
    ]

    _safe_log_outputs(
        {
            "case": "reminder_create",
            "assistant_reply": followup_state.get("assistant_reply"),
            "tool_trace": format_messages_string(followup_state.get("messages", [])),
            "reminders": creation_snapshot,
        },
        run_id,
    )

    # Prepare forced decision for reminder judge evaluation
    monkeypatch.setenv("REMINDER_JUDGE_FORCE_DECISION", "approve")

    reply_email = {
        "from": "Assistant <assistant@example.com>",
        "to": "Utility Billing <billing@example.com>",
        "subject": "Re: Bill due October 1",
        "body": "Payment processed today. Thanks for the reminder!",
        "id": "thread-reminder-invoice-approval",
    }

    reply_payload = {"email_input": reply_email}
    reply_summary = summarize_email_for_grid(reply_email)
    _safe_log_inputs({"case": "reminder_cancel", "email": reply_email}, run_id)

    def _invoke_reply():
        return email_assistant.invoke(reply_payload, config=thread_config, durability="sync")

    invoke_with_root_run(_invoke_reply, root_name="agent:reminder:cancel", input_summary=reply_summary)
    reply_state = _extract_values(email_assistant.get_state(thread_config))

    reminders_after = module.reminder_store.iter_active_reminders()
    assert not any(r.thread_id == followup_email["id"] for r in reminders_after)
    _safe_log_outputs(
        {
            "case": "reminder_cancel",
            "assistant_reply": reply_state.get("assistant_reply"),
            "tool_trace": format_messages_string(reply_state.get("messages", [])),
            "reminders": [],
        },
        run_id,
    )

    # Secondary correctness judge for the approval run
    messages = followup_state.get("messages", [])
    tool_trace = format_messages_string(messages)
    tool_calls_summary, tool_calls_json = build_tool_call_context(messages)
    raw_payload = serialise_messages(messages)
    reminder_cleared = [
        {
            "thread_id": r.thread_id,
            "subject": r.subject,
            "due_at": getattr(r.due_at, "isoformat", lambda: "")(),
            "reason": r.reason,
            "status": r.status,
        }
        for r in reminders_after
    ]

    judge_project_override = os.getenv("EMAIL_ASSISTANT_JUDGE_PROJECT_OVERRIDE")
    if judge_project_override:
        os.environ["EMAIL_ASSISTANT_JUDGE_PROJECT"] = judge_project_override

    reminder_judge_project_override = os.getenv("EMAIL_ASSISTANT_REMINDER_JUDGE_PROJECT_OVERRIDE")
    if reminder_judge_project_override:
        os.environ["EMAIL_ASSISTANT_REMINDER_JUDGE_PROJECT"] = reminder_judge_project_override

    try:
        correctness_verdict = run_correctness_judge(
            email_markdown=followup_state.get("email_markdown", ""),
            assistant_reply=followup_state.get("assistant_reply", ""),
            tool_trace=tool_trace,
            tool_calls_summary=tool_calls_summary,
            tool_calls_json=tool_calls_json,
            raw_output_optional=raw_payload,
            parent_run_id=run_id,
        )
    except JudgeUnavailableError as exc:
        warnings.warn(f"Reminder judge unavailable: {exc}")
        monkeypatch.setenv("REMINDER_JUDGE_FORCE_DECISION", "")
        return

    try:
        reminder_verdict = run_reminder_run_judge(
            email_markdown=followup_state.get("email_markdown", ""),
            assistant_reply=followup_state.get("assistant_reply", ""),
            reminder_created=creation_snapshot,
            reminder_cleared=reminder_cleared,
            sender_email=followup_email.get("from", ""),
            parent_run_id=run_id,
        )
    except JudgeUnavailableError as exc:
        warnings.warn(f"Reminder judge unavailable: {exc}")
        monkeypatch.setenv("REMINDER_JUDGE_FORCE_DECISION", "")
        return

    composite = run_composite_judge(
        correctness=correctness_verdict,
        reminder=reminder_verdict,
        parent_run_id=run_id,
        sender_email=followup_email.get("from", ""),
        email_markdown=followup_state.get("email_markdown", ""),
        email_input=followup_email,
        reminder_created=creation_snapshot,
        reminder_cleared=reminder_cleared,
    )

    _safe_log_outputs(
        {
            "case": "reminder_judge",
            "correctness_judge": correctness_verdict.model_dump(),
            "reminder_judge": reminder_verdict.model_dump(),
            "composite": composite.model_dump(),
        },
        run_id,
    )

    monkeypatch.setenv("REMINDER_JUDGE_FORCE_DECISION", "")
