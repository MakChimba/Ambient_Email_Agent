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

