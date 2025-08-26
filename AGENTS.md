# Agent Architecture

This project demonstrates an evolving AI email assistant built with LangGraph and Google's Gemini models. The architecture is designed to be modular, with functionality increasing across four distinct agents.

## Core Concepts

- Triage: The first step in the graph. A classification agent decides if an email should be ignored (e.g., spam), or responded to.
- Response Agent: The main worker. It uses tools to gather information and draft responses.
- Tools: The agent has access to tools like `write_email`, `check_calendar_availability`, etc., to perform actions.
- Human-in-the-Loop (HITL): Later agents introduce approval steps where a human must confirm or edit the AI's proposed actions before execution.
- Memory: The most advanced agents use a SQLite database (`SqliteSaver`) to maintain persistent memory across interactions, allowing them to recall past conversations.

## Agent Implementations

1. `email_assistant`:
   - The foundational agent.
   - Implements the core `triage` -> `respond` graph.
   - Uses the Gemini Pro model and basic tools.

2. `email_assistant_hitl`:
   - Builds on the first agent by adding a human-in-the-loop (HITL) step.
   - Before executing a tool (like sending an email), the graph pauses and waits for human approval.

3. `email_assistant_hitl_memory`:
   - Adds persistent memory to the HITL agent.
   - The agent's state is saved and loaded using `SqliteSaver`, allowing for contextually aware conversations over time.

4. `email_assistant_hitl_memory_gmail`:
   - An optional, advanced agent that integrates directly with the Gmail API.
   - Requires separate authentication and setup (`setup_gmail.py`).
   - Can read and process emails directly from a user's inbox.

## Recent Changes and Improvements

- Gemini 2.5 compatibility: tool binding now uses `tool_choice="any"` for HITL agents (`email_assistant_hitl`, `email_assistant_hitl_memory`) to avoid the Gemini 400 error about `allowed_function_names`.
- Robust triage routing: If the router’s structured output is missing/invalid, defaults to `respond` to prevent stalls. The HITL + memory agent also incorporates `response_preferences` into triage so preference rules (e.g., “don’t reply to direct action”) influence notify vs respond.
- HITL auto-accept: Both memory agents support `HITL_AUTO_ACCEPT=1` to accept interrupt actions automatically (useful for demos/tests). Unset the env var to use Agent Inbox interactively.
- Gmail agent safety: Tool invocation wrapped with try/except, fallback to `Done` when the model fails to emit a call, and safer classification handling. Tests mock `mark_as_read` to avoid auth in CI.
- Gmail completion toggle: `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1` optionally skips the final Gmail `mark_as_read` call for demos without credentials (default is disabled).
 - No‑reply/system notifications: If the email comes from a no‑reply address or explicitly says “do not reply,” the agent may finalize with `Done` without drafting an email, preventing loops and matching expected policy.
 - Auto‑HITL Question handling: In auto‑accept demos/tests, `Question` prompts receive a minimal synthetic response so flows proceed without manual input. In live HITL, true interrupts are preserved.

### Defaults and Test Modes

- Default agent (tests/runners): The project’s default test target is now the Gmail HITL+memory agent `email_assistant_hitl_memory_gmail` since this is the intended production agent.
- Offline eval mode: Set `EMAIL_ASSISTANT_EVAL_MODE=1` to synthesize deterministic tool calls without a live LLM. Useful for CI and tool-call tests; disable to exercise full model behavior.
- HITL auto-accept: `HITL_AUTO_ACCEPT=1` auto-accepts tool interrupts during tests/demos.
- Skip mark-as-read: `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1` avoids calling Gmail in tests/demos.
- Notebook test mode: `NB_TEST_MODE=1` makes notebooks skip long-running or online-only cells.

Environment summary for CI-like runs:
- `HITL_AUTO_ACCEPT=1`
- `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1`
- `EMAIL_ASSISTANT_EVAL_MODE=1`

## Spam Flow (Gmail)

- Trigger: During tool HITL, if the human responds to a `Question` with feedback containing spam-indicative keywords (e.g., “spam”, “phish”, “junk”).
- Flow:
  - The agent presents a confirmation card in Agent Inbox: “Move this thread to Spam?”
  - On Accept: calls a Gmail helper to apply the `SPAM` label and remove `INBOX`, records the action in the tool messages, updates triage memory to bias similar emails toward ignore/notify, and ends the workflow.
  - On decline: records that the user declined and continues the workflow.
- Rationale: keeps risky mailbox actions under explicit HITL control while still learning from feedback.
- Model name normalization: Accepts either bare names (`gemini-2.5-pro`) or prefixed (`google_genai:gemini-2.5-pro`, `models/gemini-2.5-pro`) and normalizes internally.

## Running in LangGraph Studio

- Start: `langgraph up`
- Graphs: choose one of `email_assistant`, `email_assistant_hitl`, `email_assistant_hitl_memory`, or `email_assistant_hitl_memory_gmail`.
- Env:
  - `GOOGLE_API_KEY=...`
  - `GEMINI_MODEL=gemini-2.5-pro`
  - Optional: `HITL_AUTO_ACCEPT=1` (auto-accept HITL)
 - Tips:
   - For long tool sequences, set `recursion_limit` to `100` in the request config to avoid premature termination.
   - No‑reply/system notifications can end with `Done` (no drafted email) by design.

## Agent Inbox (HITL) Resume Payloads

- Triage (triage_interrupt_handler):
  - Respond: `[{"type":"response","args":"Short approval; proceed to reply."}]`
  - Ignore: `[{"type":"ignore","args":{}}]`
- Tool interrupt (interrupt_handler):
  - Accept: `[{"type":"accept","args":{}}]`
  - Edit write_email: `[{"type":"edit","args":{"args":{"to":"alex@example.com","subject":"Re: ...","content":"Final draft here."}}}]`
  - Feedback only: `[{"type":"response","args":"Shorten and confirm time clearly."}]`

In Studio, paste the array as-is into the Resume field. In Python, wrap with `Command(resume=[...])`.

## Running Tests

### Quick smoke (tool calls only)

Runs the production-target Gmail agent with stable, offline-friendly settings.

- LangSmith-enabled runner (records to LangSmith if tracing/env is configured):
  - `python scripts/run_tests_langsmith.py`
    - Defaults: `--agent-module=email_assistant_hitl_memory_gmail`, `-k tool_calls`, and sets the CI env toggles above.

- Plain pytest:
  - `pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`

### Quality Evaluation

We rely on the LangStudio/LangSmith UI judge to evaluate reply quality and tool usage policy compliance. Local pytest focuses on tool-call presence and order only; there is no LLM-as-judge test in the repository.

## S-Class: Reminders & Follow-ups

### Overview
This feature automatically creates reminders for important emails that require a reply. A background worker checks for due reminders and sends notifications.

### Configuration
The following environment variables control the reminder behavior:
- `REMINDER_DEFAULT_HOURS`: Default time in hours to wait before a reminder is due.
- `REMINDER_POLL_INTERVAL_MIN`: How often the worker checks for due reminders.
- `REMINDER_NOTIFY_EMAIL`: Email address to send notifications to.
- `REMINDER_DB_PATH`: Path to the SQLite database file.
- `REMINDER_LABEL_PENDING`: Gmail label for a pending reminder.
- `REMINDER_LABEL_DUE`: Gmail label for a due reminder.
- `REMINDER_LABEL_DONE`: Gmail label for a completed/cancelled reminder.

### Worker Usage
The reminder worker can be run from the command line:

`# Run once and exit`
`python scripts/reminder_worker.py --once`

`# Run continuously`
`python scripts/reminder_worker.py --loop`
