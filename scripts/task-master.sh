#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run Task Master CLI via npx, using project-local config
exec npx -y --package=task-master-ai task-master "$@"

