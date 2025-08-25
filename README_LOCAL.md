# Local Development & Deployment Notes

This document contains recipes and notes for running the Email Assistant components in a local development environment, particularly under WSL.

## Running the Reminder Worker

The reminder worker is a standalone script that should run as a persistent background process to check for due reminders.

### 1. Using tmux (interactive sessions)

tmux lets you run the worker in a session you can detach from and leave running.

- Start a new session:
  ```bash
  tmux new -s reminders
  ```
- Inside tmux, activate the env and start the worker in loop mode:
  ```bash
  source .venv/bin/activate
  python scripts/reminder_worker.py --loop
  ```
- Detach with Ctrl+b then d. Reattach later:
  ```bash
    tmux attach -t reminders
  ```

### 2. Using cron (automated execution)

Ideal for a production-like setup. Run the worker periodically with your project’s venv.

- Open your crontab:
  ```bash
  crontab -e
  ```
- Add this line to run every 15 minutes. Replace `/path/to/project` with the absolute path to this repo:
  ```cron
  */15 * * * * cd /path/to/project && source .venv/bin/activate && python scripts/reminder_worker.py --once >> _artifacts/reminders.log 2>&1
  ```

Notes
- Use absolute paths in cron; relative paths may not resolve.
- Ensure `.venv` exists and dependencies are installed before enabling the cron job.
- Logs are appended to `_artifacts/reminders.log` for inspection.

## Test & Evaluation Modes

This repo supports both offline-friendly tests and live model evaluation.

### Defaults

- The default agent for test runs is the Gmail HITL+memory agent: `email_assistant_hitl_memory_gmail`.
- For stable CI-style runs, the following env toggles are commonly set:
  - `HITL_AUTO_ACCEPT=1`
  - `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1`
  - `EMAIL_ASSISTANT_EVAL_MODE=1` (synthesize tool calls without a live LLM)

### Notebooks

- Notebook tests set `NB_TEST_MODE=1` to skip long/online cells. Run notebooks interactively without this env to exercise full behavior.

### Running Tests

- Tool-call smoke tests (stable, offline-friendly):
  - `pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`
  - or `python scripts/run_tests_langsmith.py` (records to LangSmith if configured)

- Full tests with LLM-as-judge (live model):
  1. Install `langchain-google-genai` and set `GOOGLE_API_KEY`.
  2. Unset `EMAIL_ASSISTANT_EVAL_MODE` (or set to `0`).
  3. Run:
     - `pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail`
     - or `python scripts/run_tests_langsmith.py tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail`

Notes
- Gmail tools return mock results on missing credentials; tests assert tool-call presence, not delivery.
- Live model runs incur real LLM cost and may retry on transient provider errors.
