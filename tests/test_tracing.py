import sys
import types

import pytest

from email_assistant import tracing


def test_grid_text_fallback():
    assert tracing._grid_text("hello") == "hello"
    assert tracing._grid_text("  ") == "[n/a]"
    assert tracing._grid_text(None) == "[n/a]"


def test_strip_markdown_to_text_basic():
    src = """## Heading\n\n- item one\n- item two\n\n**Bold** text with a [link](https://example.com)."""
    out = tracing.strip_markdown_to_text(src)
    assert "Heading" in out
    assert "item one" in out
    assert "**" not in out
    assert "link" in out
    assert "https://" not in out


def test_summarize_email_for_grid_truncates_body():
    email = {
        "from": "Alice <alice@example.com>",
        "to": "Bob <bob@example.com>",
        "subject": "Quarterly planning meeting",
        "body": "Lorem ipsum " * 200,
    }
    summary = tracing.summarize_email_for_grid(email)
    assert summary.startswith("Quarterly planning meeting")
    assert "->" in summary
    lines = summary.splitlines()
    assert len(lines) == 2
    assert len(lines[1]) <= 800
    assert "{" not in summary and "}" not in summary


def test_summarize_tool_call_for_grid_email():
    args = {
        "to": "alice@example.com",
        "subject": "Quarterly Results",
        "content": "Body text here",
        "attachments": ["a", "b"],
    }
    summary = tracing.summarize_tool_call_for_grid("send_email_tool", args)
    assert summary.startswith("[tool] send_email_tool")
    assert "to=" in summary
    assert "attachments=2" in summary
    assert "{" not in summary and "}" not in summary


def test_summarize_llm_for_grid_user_focus():
    class Dummy:
        type = "human"

        def __init__(self, content):
            self.content = content

    payload = [Dummy("hello there"), Dummy("more info"), {"role": "assistant", "content": "ok"}]
    summary = tracing.summarize_llm_for_grid(payload)
    assert summary.startswith("3 msgs")
    assert "last user" in summary


@pytest.mark.parametrize("length", [10, 100, 5000])
def test_truncate_markdown(length):
    text = "x" * length
    result = tracing.truncate_markdown(text, max_chars=100)
    if length <= 100:
        assert result == text
    else:
        assert result.endswith("...")
        assert len(result) <= 100


def test_maybe_update_run_io_graceful_without_client(monkeypatch):
    # Simulate langsmith import failure so helper bails out cleanly
    fake_module = types.ModuleType("langsmith")
    monkeypatch.setitem(sys.modules, "langsmith", fake_module)
    assert tracing.maybe_update_run_io(run_id="run-123") is False


def test_format_final_output_reply_snippet():
    state = {
        "classification_decision": "respond",
        "messages": [
            {
                "tool_calls": [
                    {
                        "name": "send_email_tool",
                        "args": {"response_text": "Thanks for the update"},
                    }
                ]
            }
        ],
    }
    output = tracing.format_final_output(state)
    assert output.startswith("[reply]")
    assert "Thanks" in output
    assert "{" not in output


def test_format_final_output_ignore():
    state = {"classification_decision": "ignore", "messages": []}
    output = tracing.format_final_output(state)
    assert output.startswith("[no_action]")
