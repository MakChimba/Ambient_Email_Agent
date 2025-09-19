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
- Spam tool: Added `mark_as_spam_tool` (HITL-gated). After explicit confirmation, the agent may move a thread to Spam and end the workflow without `Done`. The Gmail tool registry now exposes this helper for both runtime and tests.
- No‑reply/system notifications: If the email comes from a no‑reply address or explicitly says “do not reply,” the agent may finalize with `Done` without drafting an email, preventing loops and matching expected policy.
- Document review requests now explicitly promise to review the technical materials and commit to the stated deadline (e.g., “feedback before Friday”) so live replies meet dataset criteria.
- Swimming class inquiries automatically ask to reserve a spot for the sender’s daughter, matching scheduling expectations from the Gmail dataset and tests.
 - Auto‑HITL Question handling: In auto‑accept demos/tests, `Question` prompts receive a minimal synthetic response so flows proceed without manual input. In live HITL, true interrupts are preserved.
 - Gmail HITL card improvements: For `send_email_tool`, the Agent Inbox card now clearly shows the resolved recipient (original sender) as `To`, your account as `From`, and a normalized `Subject` (adds `Re:` if missing) alongside the drafted body. This makes approvals unambiguous.
 - StructuredPrompt outputs (Gmail): The Gmail agent now returns additional top-level fields in the final state to support external evaluators:
   - `assistant_reply`: concise summary of the sent reply
   - `tool_trace`: normalized conversation + tool-call trace
   - `email_markdown`: canonical email context block
 - Tool-arg compatibility toggle: Some evaluators expect `send_email_tool.email_address` to contain the reply recipient (the other party). By default (live mode), `email_address` remains your address (correct for Gmail). For compatibility in evals/demos, set `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS=1` (this is also implied when `EMAIL_ASSISTANT_EVAL_MODE=1`).
  - Timezone defaults: The Gmail agent uses Australia/Melbourne by default for prompts and scheduling tools unless an explicit timezone is provided.

### Defaults and Test Modes

- Default agent (tests/runners): The project’s default test target is now the Gmail HITL+memory agent `email_assistant_hitl_memory_gmail` since this is the intended production agent.
- Offline eval mode (optional): Set `EMAIL_ASSISTANT_EVAL_MODE=1` when you need deterministic tool calls without a live Gemini model. Leave it unset/`0` for real-model runs.
- HITL auto-accept: `HITL_AUTO_ACCEPT=1` auto-accepts tool interrupts during tests/demos.
- Skip mark-as-read: `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1` avoids calling Gmail in tests/demos.
- Notebook test mode: `NB_TEST_MODE=1` makes notebooks skip long-running or online-only cells.

Environment summary for deterministic/offline runs:
- `HITL_AUTO_ACCEPT=1`
- `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1`
- `EMAIL_ASSISTANT_EVAL_MODE=1`
- Optional: `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS=1` (puts recipient into `send_email_tool.email_address` for compatibility with some evaluators)

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

### Studio Mapping (LLM-as-Judge)
- For general LLM-as-Judge correctness evaluations, map these fields:
  - `email_markdown` → `output.email_markdown`
  - `assistant_reply` → `output.assistant_reply`
  - `tool_trace` → `output.tool_trace`
  - `raw_output_optional` → `output`
- If running an older build without these keys, use fallbacks:
  - `assistant_reply` → `output.messages[-1].content`
  - `tool_trace` → last AI message’s `tool_calls`

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

Runs the production-target Gmail agent. Provide `GOOGLE_API_KEY` for live Gemini calls, or enable offline synthesis when needed.

- LangSmith-enabled runner (records to LangSmith if tracing/env is configured):
  - `python scripts/run_tests_langsmith.py`
    - Defaults: `--agent-module=email_assistant_hitl_memory_gmail`, `-k tool_calls`, and honors the live/offline environment you set.

- Plain pytest:
  - `pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`
  - Add `EMAIL_ASSISTANT_EVAL_MODE=1` (or `--offline-eval` when using `tests/run_all_tests.py`) if you need deterministic tool calls without Gemini access.

### Live Gemini Coverage

The automated suite now exercises the live Gemini workflow end-to-end:
- `tests/test_live_smoke.py` runs two experiment cases against the Gmail agent, snapshots the drafted reply (subject/body excerpt) and verifies the tool-call sequence. The test auto-generates `tests/snapshots/live_smoke.json` in eval mode and compares live runs against the stored ordering.
- `tests/test_response.py` enforces tool ordering and validates reply content against dataset criteria whenever the agent runs live (i.e., `EMAIL_ASSISTANT_EVAL_MODE` disabled).
- `tests/test_live_hitl_spam.py` drives the Question → `mark_as_spam_tool` HITL flow using the agent’s actual tool plan, ensuring Gmail helpers are invoked after confirmation. The test falls back to deterministic behaviour when `EMAIL_ASSISTANT_EVAL_MODE=1`.

To run the live suite locally:
```
pytest tests/test_live_smoke.py --agent-module=email_assistant_hitl_memory_gmail
pytest tests/test_live_hitl_spam.py --agent-module=email_assistant_hitl_memory_gmail
pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls
```
For deterministic/offline runs, set `EMAIL_ASSISTANT_EVAL_MODE=1` (and optionally `EMAIL_ASSISTANT_UPDATE_SNAPSHOTS=1` when updating the smoke snapshots).

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
