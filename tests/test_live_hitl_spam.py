import json
import os
from pathlib import Path

import pytest

from email_assistant.utils import extract_tool_calls, format_gmail_markdown, parse_gmail
from langgraph.store.memory import InMemoryStore

from tests.agent_test_utils import has_google_key, is_eval_mode
from tests.trace_utils import configure_tracing_project, configure_judge_project
from email_assistant.email_assistant_hitl_memory_gmail import interrupt_handler, llm_call


SPAM_CASE = "spam_flow"


def _load_spam_case() -> dict:
    dataset = Path("datasets/experiment_gmail.jsonl")
    with dataset.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("metadata", {}).get("case") == SPAM_CASE:
                return record["inputs"]["email_input"]
    raise RuntimeError("Spam case not found in experiment dataset")


def _extract_values(state):
    if hasattr(state, "values"):
        return state.values
    return state


@pytest.fixture(autouse=True)
def _ensure_eval_when_key_missing(monkeypatch):
    configure_tracing_project("email-assistant-test-live-hitl-spam")
    configure_judge_project("email-assistant-judge-test-live-hitl-spam")
    if not has_google_key() and not is_eval_mode():
        monkeypatch.setenv("EMAIL_ASSISTANT_EVAL_MODE", "1")
    os.environ.setdefault("EMAIL_ASSISTANT_SKIP_MARK_AS_READ", "1")


def test_hitl_spam_flow(agent_module_name, monkeypatch, gmail_service):
    if "gmail" not in agent_module_name:
        pytest.skip("Spam HITL flow only applies to the Gmail agent")

    spam_input = _load_spam_case()

    import email_assistant.email_assistant_hitl_memory_gmail as gmail_agent

    monkeypatch.setenv("HITL_AUTO_ACCEPT", "0")

    def scripted_interrupt(requests):
        if not requests:
            return [{"type": "ignore", "args": {}}]
        action = ((requests[0] or {}).get("action_request", {}) or {}).get("action", "")
        if action == "Question":
            return [{"type": "response", "args": "This is spam"}]
        if action == "mark_as_spam_tool":
            return [{"type": "accept", "args": {}}]
        return [{"type": "accept", "args": {}}]

    monkeypatch.setattr(gmail_agent, "_maybe_interrupt", scripted_interrupt)

    author, to, subject, body, email_id = parse_gmail(spam_input)
    email_markdown = format_gmail_markdown(subject, author, to, body, email_id)
    store = InMemoryStore()
    state = {
        "messages": [{"role": "user", "content": f"Respond to the email: {email_markdown}"}],
        "email_input": spam_input,
        "classification_decision": "respond",
    }

    llm_result = llm_call(state, store)
    state["messages"] = state["messages"] + llm_result["messages"]
    cmd = interrupt_handler(state, store)
    tool_order = extract_tool_calls(cmd.update.get("messages", []))
    assert any(name == "question" for name in tool_order), "Expected Question tool in spam flow"
    assert any(name == "mark_as_spam_tool" for name in tool_order), "Expected mark_as_spam_tool to run"

    email_id = spam_input.get("id") or spam_input.get("message_id")
    assert any(action == ("mark_as_spam", email_id) for action in getattr(gmail_service, "actions", [])), (
        "Spam flow should call mark_as_spam on the Gmail service"
    )

    if is_eval_mode():
        # Deterministic run should end without drafting a reply and produce an empty assistant reply.
        assert cmd.update.get("messages", [])[-1].get("role") == "tool"
