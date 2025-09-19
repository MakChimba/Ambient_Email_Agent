import importlib

import pytest

from email_assistant.tools.gmail import gmail_tools


@pytest.fixture(autouse=True)
def reload_gmail_tools(monkeypatch):
    """Reload gmail_tools for each test so monkeypatching globals resets cleanly."""
    # Reloading ensures patched globals from previous tests don't leak.
    module = importlib.reload(gmail_tools)
    yield module
    importlib.reload(gmail_tools)


def test_send_calendar_invite_simulates_without_api(monkeypatch):
    monkeypatch.setattr(gmail_tools, "GMAIL_API_AVAILABLE", False)

    success, message = gmail_tools.send_calendar_invite(
        attendees=["alice@example.com"],
        title="Project Sync",
        start_time="2025-09-23T15:00:00",
        end_time="2025-09-23T15:45:00",
        organizer_email="me@example.com",
    )

    assert success is True
    assert "Simulated meeting scheduling" in message


def test_send_calendar_invite_reports_missing_credentials(monkeypatch):
    monkeypatch.setattr(gmail_tools, "GMAIL_API_AVAILABLE", True)
    monkeypatch.setattr(gmail_tools, "get_credentials", lambda *args, **kwargs: None)

    success, message = gmail_tools.send_calendar_invite(
        attendees=["alice@example.com"],
        title="Project Sync",
        start_time="2025-09-23T15:00:00",
        end_time="2025-09-23T15:45:00",
        organizer_email="me@example.com",
    )

    assert success is False
    assert "credentials missing" in message.lower()


def test_schedule_meeting_tool_surfaces_failure(monkeypatch):
    monkeypatch.setattr(gmail_tools, "send_calendar_invite", lambda *args, **kwargs: (False, "calendar offline"))

    result = gmail_tools.schedule_meeting_tool.invoke(
        {
            "attendees": ["me@example.com", "alice@example.com"],
            "title": "Tax Planning",
            "start_time": "2025-09-23T15:00:00",
            "end_time": "2025-09-23T15:45:00",
            "organizer_email": "me@example.com",
        }
    )

    assert result == "calendar offline"
