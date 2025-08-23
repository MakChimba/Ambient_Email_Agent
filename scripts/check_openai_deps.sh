#!/usr/bin/env bash
set -e

echo "INFO: Searching for OpenAI dependencies in src/ and tests/..."

# The -r flag searches recursively, -i ignores case.
# If no matches are found, grep exits 1; treat that as non-fatal.
if ! grep -r -n -i --include='*.py' 'openai' src/ tests/; then
  echo "INFO: No matches found."
fi

echo "\nINFO: Search complete."

