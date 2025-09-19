import pytest
from langgraph.store.memory import InMemoryStore

import email_assistant.email_assistant_hitl_memory_gmail as gmail_agent


def test_update_memory_skips_llm_when_eval_mode(monkeypatch):
    """Offline eval mode should avoid invoking live LLMs for memory updates."""
    monkeypatch.setenv("EMAIL_ASSISTANT_EVAL_MODE", "1")

    called = {"value": False}

    def _raise_if_called(*args, **kwargs):
        called["value"] = True
        raise AssertionError("get_llm should not be called during eval mode")

    monkeypatch.setattr(gmail_agent, "get_llm", _raise_if_called)

    store = InMemoryStore()
    namespace = ("test", "user")
    messages = [{"role": "user", "content": "Sample feedback"}]

    gmail_agent.update_memory(store, namespace, messages)

    assert called["value"] is False
    assert store.get(namespace, "user_preferences") is None

    monkeypatch.delenv("EMAIL_ASSISTANT_EVAL_MODE", raising=False)
