import json

import pytest

from email_assistant.eval import judges


def test_build_tool_call_context_returns_ordered_summary():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "check_calendar_tool",
                    "args": {"date": "2025-05-01", "duration_minutes": 60},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": {"availability": ["2025-05-01T09:00"]},
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "send_email_tool",
                    "args": {
                        "email_id": "abc",
                        "email_address": "alex@example.com",
                        "content": "Booked for 10am",
                    },
                }
            ],
        },
    ]

    summary, tool_calls_json = judges._build_tool_call_context(messages)

    assert "1. check_calendar_tool" in summary
    assert "result={" in summary
    payload = json.loads(tool_calls_json)
    assert payload[0]["name"] == "check_calendar_tool"
    assert payload[1]["name"] == "send_email_tool"
    assert payload[1]["args"].startswith("{\"email_id\"")


@pytest.mark.parametrize(
    "missing_tools, incorrect_tool_uses, initial_tool_usage, expected_tool_usage, content_alignment",
    [
        (["schedule_meeting_tool"], [], 5, 2, 4),
        ([], [{"tool": "send_email_tool", "why": "wrong recipient"}], None, 1, 3),
    ],
)
def test_normalise_result_dict_clamps_tool_score(
    missing_tools, incorrect_tool_uses, initial_tool_usage, expected_tool_usage, content_alignment
):
    raw = {
        "overall_correctness": 0.9,
        "verdict": "pass",
        "content_alignment": content_alignment,
        "tool_usage": initial_tool_usage,
        "missing_tools": missing_tools,
        "incorrect_tool_uses": incorrect_tool_uses,
        "evidence": ["Example evidence"],
        "notes": "",
    }

    normalised = judges._normalise_result_dict(raw)

    assert normalised["tool_usage"] == expected_tool_usage
    assert normalised["verdict"] == "fail"
    expected_overall = 0.6 * (content_alignment / 5) + 0.4 * (expected_tool_usage / 5)
    assert normalised["overall_correctness"] == pytest.approx(expected_overall)
