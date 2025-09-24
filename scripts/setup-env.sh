#!/usr/bin/env bash
set -euo pipefail

GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
EMAIL_ASSISTANT_EVAL_MODE="${EMAIL_ASSISTANT_EVAL_MODE:-0}"

if [ "$EMAIL_ASSISTANT_EVAL_MODE" = "1" ]; then
  if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "offline-test-key" ]; then
    export GOOGLE_API_KEY="offline-test-key"
  fi
  echo "INFO: EMAIL_ASSISTANT_EVAL_MODE=1 set; running in offline eval mode"
else
  if [ -z "$GOOGLE_API_KEY" ]; then
    echo "ERROR: GOOGLE_API_KEY not found in env" >&2
    echo "Set a real Gemini key or enable EMAIL_ASSISTANT_EVAL_MODE=1 for offline runs." >&2
    exit 1
  fi

  if ! [[ "$GOOGLE_API_KEY" =~ ^AIza[0-9A-Za-z_-]{20,}$ ]]; then
    echo "ERROR: GOOGLE_API_KEY does not look like a valid Gemini API key" >&2
    echo "Provide a key starting with 'AIza' or enable EMAIL_ASSISTANT_EVAL_MODE=1 for offline runs." >&2
    exit 1
  fi

  echo "OK: Valid-looking GOOGLE_API_KEY detected"
fi

export GOOGLE_API_KEY
exec "$@"
