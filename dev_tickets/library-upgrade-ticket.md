# Ticket: LangSmith / LangGraph / LangChain Dependency Upgrade

## Summary
- Upgrade `langsmith` to 0.4.30, `langgraph` to 0.6.x, and `langchain` to 0.3.27.
- Preserve existing Gmail HITL + memory agent behaviour, tooling, and reminder worker stability.
- Document new env/runtime knobs introduced by the upgraded libraries.

## Background
- Research notes in `dev_tickets/library-upgrade-research.md` highlight API changes (LangSmith telemetry queue tuning, LangGraph durability/interrupt payload updates, LangChain schema fixes).
- Current pins (langsmith 0.3.45, langgraph 0.4.8, langchain 0.3.25) block adoption of new queue controls, durability hooks, and structured output patches needed for upcoming features.

## Scope
- Dependency bumps and lockfile regeneration (no unrelated package upgrades).
- Code updates required to satisfy new LangGraph durability and interrupt payload schemas.
- Test + tracing updates to ensure LangSmith queue flushing and judge integrations stay deterministic.
- Documentation updates for env vars and operational guidance.

## Out of Scope
- Migrating existing graphs to typed runtime contexts (`context_schema`) beyond compatibility fixes.
- Reworking reminder worker architecture beyond ensuring compatibility.
- Broad refactors to agent logic.

## Implementation Plan
- [x] **Phase 1 – Library pin bump**
  - [x] Update `pyproject.toml` / `uv.lock` to `langsmith==0.4.30`, `langchain==0.3.27` (keep LangGraph at 0.4.8 temporarily).
  - [x] Run `pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls` with live Gemini configuration.
  - [x] Verify LangSmith queue env vars (`LANGSMITH_MAX_BATCH_SIZE`, `run_ops_buffer_size`, `run_ops_buffer_timeout_ms`, `max_batch_size_bytes`) interact correctly and that `_record_feedback` flushes in staging when `LANGSMITH_RUN_ENDPOINTS` is overridden.
- [ ] **Phase 2 – LangGraph 0.6 upgrade**
  - [ ] Bump LangGraph to 0.6.x and replace deprecated `checkpoint_during` usage with explicit `durability` arguments where persistence before HITL is required.
  - [ ] Confirm interrupt payload consumers ignore removed `ns` / `resumable` fields (update Agent Inbox serializers/tests as needed).
  - [ ] Validate `SqliteSaver` integration (memory agent + reminder worker) following 0.6 guidance on context manager usage.
  - [ ] Exercise HITL flows via `pytest tests/test_live_hitl_spam.py --agent-module=email_assistant_hitl_memory_gmail` (live) and deterministic eval mode (`EMAIL_ASSISTANT_EVAL_MODE=1`, `HITL_AUTO_ACCEPT=1`).
- [ ] **Phase 3 – Optional enhancements**
  - [ ] Assess adoption of `langsmith.Client.get_experiment_results()` for judge pagination in `src/email_assistant/eval/judges.py`.
  - [ ] Evaluate replacing manual HITL payload dicts with `langgraph.prebuilt.interrupt.HumanInterrupt`.
  - [ ] Consider wrapping high side-effect nodes with `@task` decorators for better durable replay semantics.
- [ ] **Documentation & Communication**
  - [ ] Update README / agent docs with new env toggles (`OTEL_ENABLED`, `OTEL_ONLY`, durability usage, org-scoped LangSmith keys, reminder worker notes).
  - [ ] Capture testing matrix (live vs eval modes, env var combinations) for regression reference.
  - [ ] Announce migration steps + rollback plan to the team once Phase 2 is validated.

## Acceptance Criteria
- All automated suites (`tests/test_response.py -k tool_calls`, `tests/test_live_smoke.py`, `tests/test_live_hitl_spam.py`) pass in live mode post-upgrade.
- Deterministic offline runs succeed with `EMAIL_ASSISTANT_EVAL_MODE=1`, `HITL_AUTO_ACCEPT=1`, `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1`.
- LangSmith feedback uploads flush correctly with custom `LANGSMITH_RUN_ENDPOINTS` values; no regressions in `_record_feedback`.
- Agent Inbox resumes function without relying on removed LangGraph interrupt fields.
- Reminder worker (`python scripts/reminder_worker.py --once`) runs without SQLite or durability regressions.
- Documentation reflects new env vars and operational expectations.

## Testing Notes
- Live Gemini verification requires valid `GOOGLE_API_KEY`, `GEMINI_MODEL=gemini-2.5-pro`, and tracing env vars when applicable.
- For offline smoke checks, set `EMAIL_ASSISTANT_EVAL_MODE=1`, `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS=1` (if evaluator compatibility is required).
- Monitor LangSmith logs for queue back-pressure warnings after pin bumps; adjust `run_ops_buffer_size` / `max_batch_size_bytes` as needed.

## Risks & Mitigations
- **Feedback loss when using custom `LANGSMITH_RUN_ENDPOINTS`:** Validate on staging before production rollout; keep rollback plan to 0.3.45.
- **Pytest dependency conflicts via `langsmith[pytest]`:** Reconcile pins in `pyproject.toml` if `pytest-asyncio` or `pytest-xdist` versions diverge.
- **Durability misconfiguration causing state loss:** Explicitly set `durability="sync"` for nodes executing irreversible tool calls; add regression tests around HITL pauses.
- **Reminder worker thread issues:** Document `SqliteSaver` thread-safety guidance; explore `AsyncSqliteSaver` if sync locking becomes problematic.

## Dependencies
- Merge behind any outstanding LangGraph DSL refactors touching durability.
- Coordinate with DevOps on LangSmith staging endpoints for feedback testing.
- Ensure LangSmith tracing credentials support org-scoped keys before rollout.

## Rollback Plan
- Revert dependency pins to previous versions (langsmith 0.3.45, langgraph 0.4.8, langchain 0.3.25).
- Restore any removed compatibility shims for `checkpoint_during` or interrupt payload fields.
- Disable new env toggles introduced during upgrade.

## References
- Research notes: `dev_tickets/library-upgrade-research.md`
- LangSmith release notes 0.4.0–0.4.30
- LangGraph 0.5/0.6 migration guide (`docs/docs/concepts/durable_execution.md`)
- LangChain 0.3.27 changelog
