# Ticket: Build AG-UI/CopilotKit Frontend for Gmail HITL Agent

## Summary
Create an operator-facing web frontend for the `email_assistant_hitl_memory_gmail` LangGraph agent using the AG-UI protocol and CopilotKit SDK. The interface must stream agent runs (triage, response drafting, tool usage), surface SqliteSaver-backed memory, and provide human-in-the-loop controls for approval/edits. Backend integration should expose an AG-UI-compliant endpoint that wraps the existing graph while preserving HITL semantics and environment toggles.

## Goals
- [ ] Deliver a working FastAPI (or equivalent) endpoint exposing the Gmail HITL agent over AG-UI.
- [ ] Implement a React-based CopilotKit client that authenticates to the endpoint and renders:
  - [ ] Inbox/email context viewer (original email, triage summary, assistant draft)
  - [ ] Run timeline with lifecycle/tool events and status indicators
  - [ ] HITL approval modals mirroring Agent Inbox behavior
  - [ ] Memory panel allowing inspection and limited edits of persistent state
- [ ] Ensure destructive tools (`send_email_tool`, `mark_as_spam_tool`, `schedule_meeting_tool`) require explicit human approval with optional auto-accept via `HITL_AUTO_ACCEPT`.
- [ ] Respect evaluation/deterministic mode env vars (`EMAIL_ASSISTANT_EVAL_MODE`, `EMAIL_ASSISTANT_SKIP_MARK_AS_READ`, `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS`) in backend and UI flows.
- [ ] Provide observability for tool errors (calendar fallback, Gmail auth issues) within the UI.

## Deliverables
- [ ] **Backend endpoint**
  - [ ] Wrap `email_assistant_hitl_memory_gmail` with `LangGraphAGUIAgent` (or equivalent) using AsyncSqliteSaver persistence.
  - [ ] Expose SSE/WebSocket stream aligned with AG-UI lifecycle, text, tool, and state-delta events.
  - [ ] Document launch steps (`langgraph up` or FastAPI command), required env vars, and auth assumptions.
- [ ] **Frontend application**
  - [ ] Scaffold via `npx copilotkit@latest init` or `npx create-ag-ui-app` (React).
  - [ ] Connect using CopilotKit LangGraph hooks (`useCoAgent`, `useLangGraphInterrupt`, `useCopilotAction`, etc.).
  - [ ] Implement primary layout showing email context, agent draft, and tool/action feed.
  - [ ] Build HITL modal displaying tool args (To, From, Subject, body, spam confirmation) and resolving interrupts.
  - [ ] Add memory view reflecting SqliteSaver state with controlled edits (e.g., update response preferences) via frontend actions.
  - [ ] Surface eval/auto-accept indicators and toggles where allowed.
- [ ] **Testing & validation**
  - [ ] Document manual walkthrough: connect, ingest sample email, triage, respond, approve draft, observe state change.
  - [ ] Provide offline/eval verification steps using `EMAIL_ASSISTANT_EVAL_MODE=1`.
  - [ ] Supply smoke test instructions or script for running key pytest cases against the endpoint.
- [ ] **Docs/README updates**
  - [ ] Add setup instructions (backend + frontend) to repo docs (e.g., `README_LOCAL.md` or new `docs/AGUI_UI.md`).
  - [ ] Document auth/security requirements and Gmail credential configuration.

## Constraints / Notes
- Reference research: `dev_tickets/ag-ui-copilotkit-research.md` for design rationale and next steps.
- Maintain parity with existing Agent Inbox semantics (question handling, tool trace summaries, auto-accept flag).
- Ensure memory edits respect existing schema; avoid exposing sensitive tool arguments not meant for operators.
- Coordinate with existing `.env`/config conventions; avoid new env vars unless necessary.
- Keep implementation ASCII-only and include focused comments only for complex logic.

## Acceptance Criteria
- [ ] Operator can load the UI, watch a run progress in real time, and approve a `send_email_tool` draft.
- [ ] Memory panel reflects changes after a run (e.g., updated response preferences) and allows safe modifications.
- [ ] Tool errors and spam flow confirmations surface in the UI with actionable messaging.
- [ ] Backend honors HITL auto-accept toggle and eval mode environment variables.
- [ ] Documentation enables another developer to set up and run the system end-to-end.
