# Notebooks: Project Updates and Usage Notes

This file highlights recent project changes that affect the notebooks and how to run them.

## Environment Flags

- `HITL_AUTO_ACCEPT=1`: Auto-accept HITL tool interrupts in demos/tests.
- `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1`: Skip Gmail `mark_as_read` in demos/tests.
- `EMAIL_ASSISTANT_EVAL_MODE=1`: Synthesize deterministic tool calls (no live LLM required).
- Optional: `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS=1`
  - Compatibility toggle for some evaluators that expect `send_email_tool.email_address` to be the reply recipient (the original sender), not your address.
  - Leave unset for live Gmail usage (default behavior is correct for sending).

## Gmail Agent HITL Display

For `email_assistant_hitl_memory_gmail`, the Agent Inbox card for `send_email_tool` is rendered with Gmail semantics:
- `To`: original sender (reply recipient)
- `From`: your account (sender)
- `Subject`: normalized with `Re:` if missing
- Draft body content

This is a display-only change to make approvals unambiguous and does not alter the sending logic.

## Structured Outputs (Gmail)

The Gmail agent now appends the following fields to the final state to support external evaluators (e.g., StructuredPrompt templates):
- `assistant_reply`: concise summary of the sent reply
- `tool_trace`: normalized conversation + tool-call trace
- `email_markdown`: canonical email context block

These are orthogonal to sending and safe for live runs.

## Notebook Tips

- Most notebooks assume the default agent is the Gmail HITL+memory agent: `email_assistant_hitl_memory_gmail`.
- If you run notebooks without credentials, keep the CI-style toggles set. With real Gmail, unset the eval/demo flags and configure credentials per `src/email_assistant/tools/gmail/README.md`.
- Reminders: The `agent.ipynb` notebook demonstrates the reminder flow. Ensure `REMINDER_DB_PATH` points to a writeable location (defaults to `../.local/reminders.db`).

## Live Gmail Runs

- Unset `EMAIL_ASSISTANT_EVAL_MODE` and `HITL_AUTO_ACCEPT` to review drafts in Agent Inbox.
- Keep `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS` unset for correct live sending semantics.
- Ensure Gmail credentials are configured as described in `src/email_assistant/tools/gmail/README.md`.
