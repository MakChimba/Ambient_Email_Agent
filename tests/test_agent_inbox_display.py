import logging

from email_assistant.utils import format_for_display


def test_format_for_display_handles_none_args(caplog):
    caplog.set_level(logging.WARNING)
    call = {"name": "send_email_tool", "args": None}

    output = format_for_display(call)

    assert "# Email Draft" in output
    assert "_Note: Original tool args were discarded" in output
    assert "send_email_tool emitted None args" in caplog.text


def test_format_for_display_normalises_schedule_list():
    call = {
        "name": "schedule_meeting_tool",
        "args": {
            "attendees": [None, "alex@example.com", ""],
            "duration_minutes": None,
            "subject": "Sync",
            "preferred_day": None,
        },
    }

    output = format_for_display(call)

    assert "alex@example.com" in output
    assert "N/A" in output
    assert "# Calendar Invite" in output


def test_format_for_display_logs_on_string_args(caplog):
    caplog.set_level(logging.WARNING)
    call = {"name": "write_email", "args": "to=alex"}

    output = format_for_display(call)

    assert "Original tool args were discarded" in output
    assert "non-dict args" in caplog.text
