#!/usr/bin/env bash
set -e

ARTIFACTS_DIR="_artifacts"
LOG_FILE="$ARTIFACTS_DIR/s3_smoke_test.txt"

echo "INFO: Running S3 smoke test..."

# Ensure artifacts directory exists for log output
mkdir -p "$ARTIFACTS_DIR"

# Prefer a local virtualenv python if available
PYTHON_BIN="python3"
if [ -x ".venv_wsl/bin/python" ]; then
  PYTHON_BIN=".venv_wsl/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

# This command imports the main agent executor and attempts to instantiate it.
# Ensure Python can find the package under ./src by extending PYTHONPATH.
PYTHONPATH="${PYTHONPATH}:src" "$PYTHON_BIN" -c "from email_assistant.email_assistant import get_agent_executor; get_agent_executor(); print('Smoke test passed: Agent executor created successfully.')" 2>&1 | tee "$LOG_FILE"

echo -e "\nINFO: Smoke test output saved to $LOG_FILE"
