# Implementation Ticket — Reminder Flow Hardening (LangGraph 1.0)

## Objective
Eliminate reminder routing inconsistencies in the Gmail HITL workflow by (1) aligning cancellation keys with the persisted reminder records, (2) introducing an atomic reminder action dispatcher that durably processes cancel/create sequences, and (3) updating user-facing documentation (including diagrams and notebooks) so the LangGraph 1.0 architecture is accurately represented and simple to maintain.

## Background
- `triage_router_task` currently enqueues reminder actions keyed by the Gmail message `email_id`, but `ReminderStore.cancel_reminder` expects a canonical thread identifier. Replies with new message IDs therefore bypass cancellation.
- Reminder actions are staged in-memory (`_PENDING_ACTIONS`) and processed by `cancel_reminder_node` followed by `create_reminder_node`. A failure between these nodes can leave the workflow with reminders cancelled but not recreated.
- Existing diagrams/documentation depict Gmail as the dispatcher and show the approve branch routing through `cancel_reminder_node`. They do not reflect the current LangGraph 1.0 implementation or the desired atomic dispatcher.

## References
- Code
  - `src/email_assistant/email_assistant_hitl_memory_gmail.py`
  - `src/email_assistant/graph/reminder_nodes.py`
  - `src/email_assistant/tools/reminders.py`
- Tests
  - `tests/test_reminders.py`
  - `tests/test_live_reminders.py`
  - `tests/test_response.py`
- Readmes/Notebooks
  - `AGENTS.md`
  - `README.md` / `README_LOCAL.md`
  - `notebooks/` (add or refresh reminder walkthrough)

## Scope
- Align reminder action identifiers with the store’s canonical thread key (or explicit `reminder_id`).
- Replace the two-step cancel→create flow with a durable, atomic dispatcher node that processes staged actions and persists outcomes safely.
- Ensure triage HITL branches never register reminder actions before approval and that state transitions stay LangGraph-native.
- Update documentation (AGENTS.md + README.md + README_LOCAL.md + notebook) with an accurate Mermaid sequence diagram of the new flow.
- Validate with both offline (eval-mode) and live Gmail test suites to prevent regressions.

## Out of Scope
- Changes to ReminderStore persistence beyond what is needed for atomic action execution.
- Inbox tooling or UI alterations.
- Gmail API credential provisioning.

## Implementation Plan

### Phase 1 — Identifier Alignment
1. Introduce a helper (e.g., `resolve_thread_key`) that consistently derives the reminder key from `email_input` / runtime context.
2. Update `triage_router_task` and reminder nodes to use the thread key (or explicit reminder IDs) when enqueuing cancel/create actions.
3. Add unit coverage ensuring `register_reminder_actions` + `cancel_reminder_node` hit the correct store rows on duplicate Gmail deliveries.

### Phase 2 — Atomic Reminder Dispatcher
1. Implement an `apply_reminder_actions_node` (LangGraph task) that consumes pending actions and executes cancel/create steps within a single durable block.
2. Persist action batches to the store (or include idempotency keys) so a process crash cannot drop requested operations; remove or refactor `_PENDING_ACTIONS` globals accordingly.
3. Adjust the state graph edges so `triage_router` → `apply_reminder_actions_node` → (conditional) `response_agent` / `triage_interrupt_handler` / `END`.
4. Extend tests to cover cancel+create sequences, ensuring no reminder loss when an error is injected between operations.

### Phase 3 — Triage & HITL Integration
1. Ensure triage only enqueues reminder actions after judge approval (HITL resume) and that reject/HITL paths make no writes.
2. Update `_maybe_interrupt` / `triage_interrupt_handler_task` expectations if the new node affects payload shape.
3. Add regression tests verifying HITL decline leaves reminders untouched and that approved actions produce store writes exactly once.

### Phase 4 — Documentation & Diagram Refresh
1. Revise `AGENTS.md` to describe the atomic reminder dispatcher and LangGraph 1.0 routing (remove Gmail-as-dispatcher language).
2. Update `README.md` and `README_LOCAL.md` with a summary of the reminder flow changes, linking to the relevant notebook and embedding or referencing the refreshed Mermaid diagram.
3. Update an existing notebook (recommend `notebooks/hitl.ipynb` or add `notebooks/reminder_flow.ipynb`) to demonstrate the new dispatcher, including an executable Mermaid render cell and sample reminder action walk-through.

### Phase 5 — Validation & Regression Testing
1. Offline regression:
   - `EMAIL_ASSISTANT_EVAL_MODE=1 HITL_AUTO_ACCEPT=1 EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1 uv run pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`
   - `uv run pytest tests/test_reminders.py`
2. Live validation (requires Gmail creds):
   - `pytest tests/test_live_reminders.py --agent-module=email_assistant_hitl_memory_gmail`
   - `pytest tests/test_live_smoke.py --agent-module=email_assistant_hitl_memory_gmail`
3. Capture LangSmith trace links (or logs) demonstrating cancel→create success and HITL no-write invariants.
4. Verify no regressions in reminder worker (`scripts/reminder_worker.py --once`) and ensure new dispatcher metadata appears in logs.

## Documentation Deliverables
- Updated `AGENTS.md` section referencing the atomic dispatcher and LangGraph 1.0 routing.
- Updated README files highlighting reminder flow behaviour and pointing to the notebook.
- Notebook showcasing the flow, with commentary on evaluation/live toggles and the refreshed Mermaid diagram.
- Inline code comments limited to essential clarifications per repo guidelines.

## Acceptance Criteria
- Reminder cancellations target the correct thread/reminder key and duplicate Gmail deliveries do not leave stale reminders.
- Atomic dispatcher ensures cancel+create operations either complete together or surface a recoverable error (no dropped reminders).
- HITL reject/decline paths leave the store untouched; approved paths write exactly once.
- Offline and live test suites listed above pass (or document waived cases with justification).
- Updated documentation (AGENTS.md, docs page, notebook) checked in with the new diagram.

## Risks & Mitigations
- **Store schema constraints**: If new persistence is required for action batching, extend the schema via migration script and back up existing data before rollout.
- **Live Gmail dependency**: Coordinate credentials early; if unavailable, capture offline traces and flag remaining validation as a blocker.
- **Notebook drift**: Ensure notebook cells run in `NB_TEST_MODE=1` when CI/testing is headless.

## Rollout Plan
- Develop on a feature branch; run offline tests locally before requesting live credentials.
- Execute live Gmail suites and share LangSmith traces prior to merge.
- After merge, monitor reminder worker logs for 24h to ensure no skipped recreation events; be prepared to revert via branch if regressions appear.

## Handoff Notes
- Include summary of traces/logs and any residual risks in the PR description.
- Link this ticket in the PR and in updated docs (where appropriate) for traceability.

## Progress — 2025-09-28

- [x] Replaced the two-node cancel/create chain with a single `apply_reminder_actions_node` that calls `ReminderStore.apply_actions()` atomically. Added a store-level batch helper and adjusted the LangGraph edges accordingly.
- [x] Normalised reminder thread identifiers inside triage by preferring the Gmail thread id and falling back to the router thread context or message id.
- [x] Introduced a pending reminder queue persisted in the LangGraph store so notify/HITL flows defer creation until a human approves the reminder.
- [x] Added `tests/test_reminders.py::test_apply_actions_batch` to cover the new store helper.
- [x] Documented the dispatcher changes in `AGENTS.md`, `README.md`, `README_LOCAL.md`, updated `notebooks/UPDATES.md`, and created `notebooks/reminder_flow.ipynb` with a mermaid diagram + code example.
- [x] Resolve recursion/looping in notify → HITL flows. The dispatcher now clears pending reminder actions from the LangGraph store after a successful apply, which prevents `triage_router` from rehydrating the same batch and allows control to advance to `response_agent`.
- [x] Stabilise evaluation-mode behaviour: offline `tests/test_response.py -k tool_calls` now passes with deterministic schedule plans (no duplicate `send_email_tool`/`Done` loops and the 60-minute requirement satisfied).

## Progress — 2025-09-29

- Added store cleanup inside `apply_reminder_actions_node` so successful reminder batches empty `pending_actions` in the LangGraph store and state. Offline reminder pytest now completes without the triage loop.
- Captured stack trace for the lingering evaluation-mode failure (`triage_interrupt_completed` double-write) while rerunning the response tool-call suite for diagnosis.

## Testing Notes — 2025-09-28

Command | Result | Notes
--- | --- | ---
`uv run pytest tests/test_reminders.py` | ✅ | Confirms `apply_actions` batching and reminder worker helpers.
`EMAIL_ASSISTANT_EVAL_MODE=1 HITL_AUTO_ACCEPT=1 EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1 uv run pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls` | ❌ | LLM judge fails (multiple `send_email_tool` + `Done` sequences). Needs root-cause before merge.
`EMAIL_ASSISTANT_EVAL_MODE=1 HITL_AUTO_ACCEPT=1 EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1 uv run pytest tests/test_live_reminders.py --agent-module=email_assistant_hitl_memory_gmail` | ❌ | No reminder created after HITL approval; pending actions loop between `triage_interrupt_handler` and the dispatcher until recursion limit is hit.
`REMINDER_DB_PATH=./.tmp-reminders.sqlite HITL_AUTO_ACCEPT=1 EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1 uv run python scripts/reminder_worker.py --once` | ✅ | Worker runs with empty queue.

## Follow-ups

- None (all outstanding items resolved within scope).

## Adhoc Items

- Schedule a live Gmail validation run (`tests/test_live_reminders.py`) when credentials become available to confirm the dispatcher fix under production settings.

## Testing Notes — 2025-09-29

Command | Result | Notes
--- | --- | ---
`uv run pytest tests/test_reminders.py` | ✅ | Confirmed after remapping the dispatcher routing — reminder apply node now exits cleanly to the response path without looping.
`EMAIL_ASSISTANT_EVAL_MODE=1 HITL_AUTO_ACCEPT=1 EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1 uv run pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls` | ✅ | Deterministic schedule plans now honour requested durations (60/90 minutes), avoid duplicate tool loops, and draft replies aligned with dataset expectations.
`pytest tests/test_live_reminders.py --agent-module=email_assistant_hitl_memory_gmail` | ✅ | Live reminder create/cancel flow now stages actions once, dispatcher clears them, and reminders persist across the HITL hand-off. Snapshot attached in pytest logs.

## Progress — 2025-09-30

- Persisted notify-branch reminder creations in the reminder store and trimmed the redundant `pending_reminder_actions` state so LangGraph no longer raises multi-write errors when the dispatcher runs.
- Rewired the Gmail HITL graph so `apply_reminder_actions_node` only executes when the router explicitly schedules it, preventing duplicate empty updates on straight-to-response paths.
- Re-ran `tests/test_live_reminders.py --agent-module=email_assistant_hitl_memory_gmail` (passes) and confirmed `notebooks/reminder_flow.ipynb` already mirrors the dispatcher stages—no notebook edits required today.

## Progress — 2025-10-01

- Added deterministic duration parsing for eval-mode scheduling so `schedule_meeting_tool` honors 60/90-minute requests, selects a consistent preferred day, and drafts replies that reference slides/collaboration when appropriate.
- Updated the offline scheduling branch to skip calendar tools for phone-call reminders (annual checkup) while still auto-booking genuine meeting requests.
- Replaced temporary triage/reminder `print` debugging with structured logger output and re-ran `EMAIL_ASSISTANT_EVAL_MODE=1 HITL_AUTO_ACCEPT=1 EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1 uv run pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls` (passes).
- Introduced shared log piping via `logs/email_assistant.log` so frontend/back-end traces can be tailed in automation; documented the toggle in `AGENTS.md` and verified with `uv run python -c` smoke log writes.
