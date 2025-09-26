"""Tool prompt templates for Gmail integration."""

# Gmail tools prompt for insertion into agent system prompts
GMAIL_TOOLS_PROMPT = """
- fetch_emails_tool: Fetch recent emails for an address.
  Inputs: email_address (str), minutes_since (int, default 30)
  Output: text summary of found emails
  Use when: you need recent threads; do not use during reply drafting.

- send_email_tool: Draft/reply in an existing thread (Gmail-aware).
  Inputs: email_id (str; use 'NEW_EMAIL' if composing new), response_text (str), email_address (str; sender), additional_recipients (list[str], optional)
  Output: confirmation text
  Use when: you have final reply text; call Done immediately after.

- check_calendar_tool: Check availability on specific dates.
  Inputs: dates (list[str]; DD-MM-YYYY)
  Output: availability summary
  Use when: the email requests times; do not schedule here.

- schedule_meeting_tool: Create a calendar event and send invites.
  Inputs: attendees (list[str]), title (str), start_time (ISO), end_time (ISO), organizer_email (str), timezone (str)
  Output: confirmation text
  Use when: time is decided; require consent/HITL before sending invites.

- mark_as_spam_tool: Move a Gmail message to Spam.
  Inputs: email_id (str)
  Output: status string
  Use when: user confirms spam in HITL; do not use without confirmation.

- Done: Mark workflow complete after sending the email.
"""

# Combined tools prompt (default + Gmail) for full integration
COMBINED_TOOLS_PROMPT = """
- fetch_emails_tool(email_address, minutes_since)
- send_email_tool(email_id, response_text, email_address, additional_recipients)
- check_calendar_tool(dates)
- schedule_meeting_tool(attendees, title, start_time, end_time, organizer_email, timezone)
- write_email(to, subject, content)
- check_calendar_availability(day)
- Done
"""
