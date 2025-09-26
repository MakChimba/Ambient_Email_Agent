# Follow-up Ticket — Ambient Email Agent: LangChain/LangGraph Phase 5 Completion

## Objective
Wrap up the outstanding Phase 5 tasks for the LangGraph 1.0 rollout, focusing on live validation, streaming demo artefacts, and release coordination.

## Outstanding Work
- Capture before/after streaming demo artefacts (screenshots or short clip) showing multi-mode (`updates`, `messages`, `custom`) output.
- Execute the live Gmail validation suite once credentials are available:
  - `pytest tests/test_live_smoke.py --agent-module=email_assistant_hitl_memory_gmail`
  - `pytest tests/test_live_hitl_spam.py --agent-module=email_assistant_hitl_memory_gmail`
  - `pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`
  - Record LangSmith URLs if tracing enabled and confirm `tests/snapshots/live_smoke.json` still matches.
- Collect final pass/fail evidence with links to artefacts, logs, and snapshots; update the implementation ticket once complete.
- Coordinate final merge strategy (feature branch, toggles) and document rollback approach.

## Current Status
- Offline coverage complete through 2025-09-30; LLM judge failures limited to expected scheduling-tool gaps when Gmail tools are mocked.
- Awaiting window with live Gmail credentials + demo time slot before commencing pass/fail evidence gathering and merge coordination.

## Acceptance Criteria
- Streaming demo artefacts stored under `notebooks/img/` (or shared location) and referenced in `notebooks/UPDATES.md` plus the implementation ticket’s “Demo Artifacts” section.
- Live Gmail suites run with accessible traces/logs, any expected judge failures documented.
- Phase 5 checklist in the primary implementation ticket updated with outcomes and release notes.

## Suggested Steps
1. Prepare environment with live credentials; rerun the live pytest commands and capture output.
2. Follow `demo_artifacts.md` checklist to record the streaming session; save assets and update documentation.
3. Revisit `dev_tickets/LangChain-LangGraph-v1-implementation-ticket.md` to tick remaining checkboxes and append rollout/rollback notes.
4. Summarise results in this follow-up ticket for historical tracking.

## Risks & Mitigations
- Live Gmail access may introduce variability → capture LangSmith traces and note any expected judge warnings.
- Streaming recordings may drift with UI updates → timestamp assets and include scenario details to ease future refreshes.
