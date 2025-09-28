# Standalone Email Assistant

The Standalone Email Assistant is a LangGraph-powered workflow that triages incoming email, coordinates tools, and drafts high-quality replies using Google's Gemini models. The project showcases how to layer human-in-the-loop checkpoints, durable memory, and Gmail automation on top of a modular graph so you can evolve an assistant from a simple responder into a production-ready agent.

## Highlights
- LangGraph 1.0 graph lineup with sync durability baked in: baseline responder, HITL, persistent memory, and Gmail-native automation.
- Multi-mode streaming (`updates`/`messages`/`custom`) wired through `get_stream_writer()` plus the `stream_progress` helper for live progress events.
- Gemini 2.5 tool orchestration (`tool_choice="any"`) with guardrails for spam handling, no-reply detection, and deterministic evaluation modes.
- Rich tool belt including calendar scheduling, email drafting, spam labeling, and reminder automation.
- Durable execution via SQLite-backed checkpoints/stores so interrupts and memory survive process restarts.
- Optional S-Class reminder worker that escalates follow-ups via Gmail labels and background polling.

## Agent Lineup
Name | What it adds | Typical use
--- | --- | ---
`email_assistant` | Core triage → respond loop with Gemini Pro and basic tools. | Playground runs, unit demos.
`email_assistant_hitl` | Human-in-the-loop approvals before tools execute. | Agent Inbox, supervised pilots.
`email_assistant_hitl_memory` | Persistent memory with `SqliteSaver` plus response preference routing. | Long-running or personalized assistants.
`email_assistant_hitl_memory_gmail` | Gmail API integration, spam tooling, and structured outputs for evaluators. | Default production/test target.

See `AGENTS.md` for a deep dive into routing, tool behavior, and the latest feature upgrades.

## Architecture at a Glance
1. **Triage** – Classifies each email (respond, ignore/notify, spam). Fallback logic defaults to respond so flows never stall.
2. **Respond** – Drafts replies, schedules meetings, or escalates reminders using LangGraph tools.
3. **Tools & Integrations** – `send_email_tool`, `schedule_meeting_tool`, Gmail spam helper, and more. HITL wraps sensitive actions.
4. **Memory & Checkpoints** – SQLite checkpoints and stores persist agent state and preference history across runs.
5. **Human Interrupts** – HITL agents pause on tool calls using `HumanInterrupt`. Auto-accept can be enabled for demos/tests.

## Getting Started
### Prerequisites
- Python 3.11+
- `uv` (recommended) or `pip`
- Google Gemini API access via `GOOGLE_API_KEY`
- Optional: Gmail API credentials for the Gmail agent (`setup_gmail.py`)

### Installation
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
# or: pip install -e .
```

### Configure Environment
Create a `.env` file (referenced by `langgraph.json`) with at least:
```
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.5-pro
```

Model helper defaults: `email_assistant.configuration.get_llm` now normalises provider/model pairs via `init_chat_model`, defaulting to `google_genai:gemini-2.5-pro`. Override the provider by setting `EMAIL_ASSISTANT_MODEL_PROVIDER` or prefixing the model value (e.g., `google_genai:gemini-2.0-pro-exp`).

Helpful toggles (leave unset for live runs):
- `HITL_AUTO_ACCEPT=1` – auto-accept tool interrupts.
- `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1` – skip Gmail `mark_as_read` call.
- `EMAIL_ASSISTANT_EVAL_MODE=1` – deterministic, offline tool calls.
- `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS=1` – evaluator compatibility mode.
- `EMAIL_ASSISTANT_MODEL_PROVIDER=google_genai` – explicit provider override for `init_chat_model` (defaults to `google_genai`).
- `EMAIL_ASSISTANT_SQLITE_TIMEOUT=60` – optional override (seconds) for SQLite busy timeouts when running LangSmith traces or parallel judges; defaults to 30.
- `EMAIL_ASSISTANT_TRACE_TIMEZONE=Australia/Sydney` – override the timezone used when auto-grouping LangSmith projects (`email-assistant-AGENT-YYYYMMDD`). Defaults to Australia/Sydney.
- `EMAIL_ASSISTANT_TRACE_DEBUG=1` – log LangGraph stream events and tracing metadata to stdout (useful when validating custom streaming progress).
- `EMAIL_ASSISTANT_TRACE_STAGE` / `EMAIL_ASSISTANT_TRACE_TAGS` – append rollout metadata to LangSmith runs for multi-stage deploys.
- `EMAIL_ASSISTANT_TIMEZONE=Australia/Melbourne` – default runtime timezone used by scripts and reminders when no explicit timezone is provided.
- `EMAIL_ASSISTANT_JUDGE_PROJECT=email-assistant:judge` – base LangSmith project name for Gemini judge evaluations (date suffix appended automatically).

### Streaming & Tracing
- All notebooks, scripts, and tests request `stream_mode=["updates","messages","custom"]`; the `custom` channel surfaces `stream_progress` events for live demos and automated logs.
- `scripts/run_real_outputs.py --stream` mirrors the production agent stream and prints each channel as it arrives. Combine with `EMAIL_ASSISTANT_TRACE_DEBUG=1` when you need verbose instrumentation.
- LangSmith project helpers respect `EMAIL_ASSISTANT_TRACE_PROJECT`, `EMAIL_ASSISTANT_TRACE_STAGE`, and `EMAIL_ASSISTANT_TRACE_TAGS`, keeping traces grouped when replaying multi-mode streaming runs (`EMAIL_ASSISTANT_TRACE_PROJECT` changes the base name; the daily suffix still applies).
- See `dev_tickets/LangChain-LangGraph-v1-implementation-ticket.md` for the upgrade log covering the LangGraph 1.0 migration and streaming instrumentation decisions.

### Launch LangGraph Studio or CLI
```bash
langgraph up                    # Start LangGraph Studio
# or run a graph directly
langgraph run email_assistant_hitl_memory_gmail
```
Select the graph that matches your use case (`langgraph.json` lists all available graphs). Increase `recursion_limit` (e.g., 100) in Studio for long tool sequences.

## Working with HITL
- HITL agents surface approvals through LangGraph Agent Inbox (`HumanInterrupt`).
- Resume payloads can accept, edit, or provide feedback (see README_LOCAL.md for samples).
- Set `HITL_AUTO_ACCEPT=1` for automated Acceptance during tests or demos.

## Testing & Quality Assurance
- **Live Gemini suites:**
  - `pytest tests/test_live_smoke.py --agent-module=email_assistant_hitl_memory_gmail`
  - `pytest tests/test_live_hitl_spam.py --agent-module=email_assistant_hitl_memory_gmail`
  - `pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`
- **Offline/deterministic runs:** set `EMAIL_ASSISTANT_EVAL_MODE=1` (and optionally `EMAIL_ASSISTANT_UPDATE_SNAPSHOTS=1`).
- **SQLite lock avoidance:** tests now auto-configure unique `EMAIL_ASSISTANT_CHECKPOINT_PATH` / `EMAIL_ASSISTANT_STORE_PATH` values, but when running scripts manually set them to fresh locations (e.g. `/tmp/checkpoints.sqlite`) to avoid `OperationalError: database is locked` from earlier runs.
- `python scripts/run_tests_langsmith.py` mirrors the tool-call suite and records traces when LangSmith is configured.
- Enable Gemini 2.5 Flash judging (`EMAIL_ASSISTANT_LLM_JUDGE=1`) to score correctness/tool usage; add `EMAIL_ASSISTANT_JUDGE_STRICT=1` to fail on judge verdicts.

## Reminders & Follow-ups
The reminder stack now includes a LangGraph dispatcher plus a background worker:
- `triage_router` batches cancel/create intents and defers HITL-created reminders via `pending_reminder_actions`.
- `apply_reminder_actions_node` executes batched operations atomically through `ReminderStore.apply_actions()` and exposes the outcome as `reminder_dispatch_outcome`.
- `triage_interrupt_handler` replay pending reminder actions only after a reviewer chooses to respond, keeping notify flows HITL-first.
- `scripts/reminder_worker.py` promotes due reminders:
  - Configure labels/timers with `REMINDER_*` environment variables.
  - Run once: `python scripts/reminder_worker.py --once`
  - Continuous loop: `python scripts/reminder_worker.py --loop` (tmux/cron examples live in README_LOCAL.md)
- See `notebooks/reminder_flow.ipynb` for a diagram + code sample showing the dispatcher in action.

## Repository Layout
Path | Description
--- | ---
`src/email_assistant/` | Agent graphs, tools, prompts, checkpointing, and server helpers.
`scripts/` | Utilities including dataset registration and reminder worker.
`tests/` | Live/eval pytest suites, spam flow coverage, notebooks smoke tests.
`datasets/` | LangSmith experiment datasets (`register_dataset_from_jsonl.py`).
`docs/` | Placeholder for extended documentation.
`README_LOCAL.md` | Local development recipes, dependency pin notes, testing nuances.

## Additional Resources
- `README_LOCAL.md` – local dev/test details, tmux/cron recipes, notebook tips.
- `AGENTS.md` – agent evolution, feature changelog, evaluation modes.
- `dev_tickets/LangChain-LangGraph-v1-implementation-ticket.md` – end-to-end implementation log with acceptance checklist and rollout notes.
- `dev_tickets/LangChain-LangGraph-v1-follow-up-ticket.md` – Phase 5 validation/demo follow-ups, risks, and merge coordination tasks.
- `notebooks/UPDATES.md` – notebook refresh log, live-first checklists, reminder/HITL env toggles.
- `CONTRIBUTING.md` – branching, review, and testing expectations.
- `system_prompt.md` – canonical assistant instructions.

## CodeRabbit Reviews
- CodeRabbit CI still gates PRs automatically, but contributors must run a local review after finishing edits: `coderabbit review --plain` from the repo root (or `--prompt-only` for a lighter summary).
- Share the exact command you ran in handoff notes so others can reproduce or compare results (`coderabbit auth status` is handy when double-checking login).
- Need a refresher? `coderabbit --help` lists every subcommand, `coderabbit <subcommand> --help` dives into its flags, and `coderabbit watch --help` covers auto-review workflows.

## Contributing
Pull requests are welcome; follow the workflow in `CONTRIBUTING.md`. Expect to run the live Gemini suites unless credentials are unavailable.

## License
Distributed under the MIT License. See `LICENSE` for details.
