# Implementation Ticket — Ambient Email Agent: LangChain/LangGraph v1.0 Upgrade

## Objective
Ship the LangChain/LangGraph v1.0 upgrade across the production Gmail HITL agent, shared tooling, and developer workflows using the research captured in `research/LangChain-LangGraph-v1-research-report.md`. The implementation should preserve current behaviour while enabling durability, Runtime/context plumbing, multi-mode streaming, and Gemini v1 conventions by default.

## References
- Research findings: `research/LangChain-LangGraph-v1-research-report.md`
- Research ticket (completed): `dev_tickets/LangChain-LangGraph-v-1-research-ticket.md`
- Proof artifacts: `research/_artifacts/`

## Scope
- Upgrade dependency pins to LangChain v1.x, LangChain Core v1.x, LangGraph v1.x, and langchain-google-genai >=2.1.12 (metadata already allows `langchain-core>=0.3.75`, so 1.x is supported) while keeping `langgraph-checkpoint-sqlite` pinned separately.
- Propagate `durability="sync"` to compiled graphs, runners, tests, scripts, and notebooks.
- Adopt `Runtime[Context]` usage (timezone, eval_mode, thread metadata) for the triage/router node and any shared helpers currently reading from `config["configurable"]`.
- Standardise multi-mode streaming (`["updates","messages","custom"]`) and surface progress events via `get_stream_writer()` where applicable.
- Align Gemini integration with v1 expectations: provider routing, structured output, tool calling, and handling of `ToolException` defaults.
- Refresh notebooks, docs, and tracing helpers to use v1 imports (`langchain.agents.create_agent`) and `.content_blocks` aware summaries.
- Ensure HITL flows continue to use `HumanInterrupt` helpers and resume cleanly after upgrade.

## Implementation Plan
### Phase 1 — Dependencies & Environment
- [x] Update `pyproject.toml` / lockfile to LangChain v1.x, LangGraph v1.x, langchain-google-genai ≥2.1, langgraph-checkpoint-sqlite latest proved in research. *(2025-09-25: bumped to langchain==1.0.0a9 / langgraph==1.0.0a3, regenerated `uv.lock`, validated install via `uv pip install --prerelease allow .`)*
- [x] Document new minimum Python version expectations if required by upstream packages. *(No change: upstream continues to target Python ≥3.11; requirement already documented in `pyproject.toml`.)*
- [x] Verify local `uv pip install` / CI install paths remain valid. *(Confirmed `uv pip install --prerelease allow .` completes locally; no CI adjustments required.)*

### Phase 2 — Runtime & Durability Defaults
- **Durability & Streaming Mechanics**
  - [x] Apply `durability="sync"` to all graph compilation helpers and high-level `invoke/stream` entrypoints (tests, scripts, notebooks).
    - 2025-09-26 — Tests/scripts now compile agents with `durability="sync"`; notebooks updated to mirror the same pattern in showcased examples.
  - [x] Update streaming loops to unpack `(mode, chunk)` and handle multi-mode lists.
    - 2025-09-26 — Tests and notebooks now use `for mode, chunk ...` with `if mode == "custom": continue`. CLI demos still rely on default stdout; revisit alongside progress logging follow-ups.
  - [x] Confirm multi-mode stream consumers (CLI, scripts) surface progress via `get_stream_writer()` without regressions. *(CLI script now supports `--stream` and prints `updates`/`messages` from multi-mode streaming; manual dry-run with eval mode passes.)*
- **Runtime Context & Tracing Updates**
  - [x] Implement the `Runtime[Context]` refactor for triage/router (and related nodes) and pass context from runners/tests. *(Phase 2 now covers memory + Gmail agents; runtime helpers/tests/scripts updated to seed metadata + multi-mode streaming.)*
  - [x] Ensure `prime_parent_run` / tracing metadata accepts the new context payloads. *(Runtime helpers now feed timezone/eval/thread metadata through every triage/tool node; scripts/tests verified via offline smoke.)*
  - [x] Add focused regression coverage so triage/router nodes exercise the new context plumbing offline and live. *(New `tests/test_runtime_context.py` validates runtime metadata extraction and triage routing with patched LLM.)*

### Phase 3 — Gemini & Tooling Updates
- **Model Initialisation & Routing**
  - [x] Normalize Gemini provider routing via `init_chat_model(..., model_provider="google_genai")` or prefixed IDs; update docs/examples accordingly.
- **Structured Output & Tool Calls**
  - [x] Ensure structured output + tool calling continue to pass type checks; update wrappers/tests as needed.
  - [x] Capture any schema or retry adjustments required for Gemini v1 exceptions.
- **Tool Error Handling & Telemetry**
  - [x] Audit tool nodes for `handle_tool_errors` / `ToolException` policy and adjust retries or messaging.
  - [x] Add/maintain sample tool emitting `custom` streaming events for progress logging.

### Phase 4 — UI, Docs, and Notebooks
- **Notebook Refresh**
  - [x] Update notebooks (`notebooks/*.ipynb`) to use the new streaming API, context schema, and `create_agent` import path.
- **Documentation Updates**
  - [x] Update README / README_LOCAL / developer docs with new env toggles (`EMAIL_ASSISTANT_TIMEZONE`, streaming expectations) and upgrade notes.
  - [x] Link docs back to the research ticket and this implementation ticket where relevant.
- **Demo Artifacts**
  - [ ] Capture before/after screenshots or short clips demonstrating multi-mode streaming in Studio or CLI.

### Phase 5 — Validation & Rollout
- [ ] Run targeted pytest suite (see Testing Notes) against live Gemini where possible; capture logs and LangSmith links when tracing enabled.
  - 2025-09-26 — Offline smoke (`pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`) passes tool assertions; remaining failure from LLM judge expected while Gmail API creds unavailable (auto-flag for scheduling behaviour).
  - 2025-09-26 — Added sanity run for base agent (`pytest tests/test_response.py --agent-module=email_assistant -k tool_calls --maxfail=1`) to confirm durability refactor; all cases pass offline.
  - 2025-09-29 — Re-ran offline Gmail smoke with eval toggles (same command as above, `--maxfail=1`). LLM judge still flags `email_input_1` for not auto-scheduling (expected in eval mode); no additional regressions observed.
- [ ] Regenerate or verify `tests/snapshots/live_smoke.json` and other fixtures if deterministic outputs change.
- [ ] Collect final pass/fail checklist evidence and link in this ticket (artifacts, logs, screenshots).
- [ ] Coordinate merge strategy (feature branch, rollout toggles) and note any fallback procedure.

## Acceptance Criteria
- [ ] Dependency bump merged with passing CI (unit + integration) on both offline eval and live Gemini runs.
- [ ] Email triage/router node uses `Runtime[Context]` and downstream code reads context via `runtime` APIs.
- [ ] All documented `.stream()` entrypoints request `["updates","messages","custom"]` and produce at least one `custom` event in demos/tests.
- [ ] HITL pause/resume works with sync durability on upgraded stack (log or LangSmith trace attached).
- [ ] Gemini structured output, tool-calling, and token streaming confirmed post-upgrade (artifact or test log).
- [ ] Tracing or logging utilities consume `.content_blocks` where available and fall back gracefully.
- [ ] README/docs updated to describe new defaults, env toggles, and migration notes; links back to research + this ticket included.

## Testing Notes
- Primary: `pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`
- Live coverage: `pytest tests/test_live_smoke.py --agent-module=email_assistant_hitl_memory_gmail`
- HITL spam flow: `pytest tests/test_live_hitl_spam.py --agent-module=email_assistant_hitl_memory_gmail`
- Optional full runner: `python scripts/run_tests_langsmith.py` (honours live vs offline env)
- When running offline/deterministic: set `EMAIL_ASSISTANT_EVAL_MODE=1`, `HITL_AUTO_ACCEPT=1`, `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1`, optionally `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS=1`.

## Risks & Mitigations
- Provider routing regressions → add unit tests around `init_chat_model` helper and document env toggles.
- Streaming fan-out affecting CLI consumers → verify progress logging/backpressure in notebooks and CLI.
- Tool exception behaviour changes → ensure HITL prompts and retry logic surface actionable errors to operators.
- Notebook re-execution drift → capture updated snapshots or note cells that require live Gemini access.

## Rollout & Follow-up
- Stage rollout behind a feature branch; merge once acceptance checklist satisfied.
- After deployment, monitor Agent Inbox for duplicate sends or missed checkpoints.
- Plan subsequent work to expand `Runtime[Context]` usage across remaining nodes (memory updates, Gmail tool wrappers) as follow-up tasks if needed.

## Progress Log
- 2025-09-25 — Phase 1 dependency upgrades landed (`langchain==1.0.0a9`, `langgraph==1.0.0a3`, regenerated `uv.lock`, validated install via `uv pip install --prerelease allow .`). Introduced shared `extract_runtime_metadata` helper and moved `email_assistant` + `email_assistant_hitl` triage nodes onto `Runtime[AssistantContext]` while preserving sync durability.
- 2025-09-26 — Extended runtime-context plumbing to memory + Gmail agents; scripts/tests emit context metadata and request multi-mode streams. Offline Gmail smoke suite passes apart from expected LLM judge failure (no Gmail API).
- 2025-09-26 — Updated tests/scripts to compile graphs with `durability="sync"`; validated base agent tool-call smoke locally.
- 2025-09-27 — Added CLI streaming progress (`--stream`) and runtime metadata regression tests (`tests/test_runtime_context.py`).
- 2025-09-28 — Phase 3 Gemini/tooling pass: standardized `get_llm` on `init_chat_model`, added `stream_progress` demo tool with custom-stream logging, and wrapped tool execution in resilient telemetry. Tests: `uv run pytest tests/test_configuration.py` and `EMAIL_ASSISTANT_EVAL_MODE=1 HITL_AUTO_ACCEPT=1 uv run pytest tests/test_response.py --agent-module=email_assistant -k tool_calls --maxfail=1`.
- 2025-09-29 — Phase 4 docs/notebooks: README, README_LOCAL, and AGENTS.md now document the new `init_chat_model` defaults, streaming env toggles (`EMAIL_ASSISTANT_TRACE_DEBUG`, provider overrides), and link back to this ticket. Updated `hitl.ipynb`, `memory.ipynb`, and `langgraph_101.ipynb` to compile graphs with `.with_config(durability="sync")`, refreshed LangGraph 1.0 narration, and kept `stream_mode=["updates","messages","custom"]` walkthroughs intact. Demo artefacts (screenshots/video) still pending.
