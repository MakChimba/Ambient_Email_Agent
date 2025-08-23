#!/usr/bin/env bash
set -e

ARTIFACTS_DIR="_artifacts"
ENV_CHECK_FILE="$ARTIFACTS_DIR/env_check.txt"

# --- S1: Environment Bootstrap ---
echo "INFO: Starting Step S1: WSL Environment Bootstrap..."

# 1. WSL & Path Safety Check
echo "INFO: Checking current path..."
if pwd | grep -q '^/mnt/'; then
  echo "WARNING: Your project is in /mnt/c/... which can cause performance and permission issues."
  echo "WARNING: Consider moving it to your WSL home directory (~/) for a better experience."
fi

# 2. System Dependencies
echo "INFO: Updating package lists and installing system dependencies..."
echo "INFO: This may require your password for 'sudo'."
sudo apt-get update
sudo apt-get install -y git curl build-essential pkg-config python3 python3-venv python3-dev python3-pip dos2unix

# 3. Git Configuration for WSL
echo "INFO: Configuring Git for safe CRLF handling..."
git config --global core.autocrlf input

# 4. Install 'uv' if not present
if ! command -v uv &> /dev/null; then
  echo "INFO: 'uv' not found. Installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.cargo/env"
else
  echo "INFO: 'uv' is already installed."
fi

# 5. Create/Normalize .env from example
echo "INFO: Setting up .env for Gemini API..."
if [ -f ".env.example" ]; then
  # Remove OpenAI key and copy to .env
  grep -v 'OPENAI_API_KEY' .env.example > .env
  echo "INFO: Created .env from .env.example, excluding OpenAI keys."
else
  touch .env
  echo "INFO: Created empty .env file."
fi

# Ensure Gemini and LangSmith vars are present
{
  echo -e "\n# LangGraph Project Configuration"
  echo "GOOGLE_API_KEY="
  echo "GEMINI_MODEL=gemini-2.5-pro"
  echo "# Optional: LangSmith for tracing"
  echo "LANGSMITH_TRACING=true"
  echo "LANGSMITH_API_KEY="
  echo "LANGSMITH_PROJECT=ambient-email-agent-gemini"
} >> .env

# 6. Install Python dependencies
echo "INFO: Installing Python dependencies with 'uv sync'..."
uv sync --extra dev

# 7. Generate Environment Report
echo "INFO: Generating environment report..."
mkdir -p "$ARTIFACTS_DIR"
(
  echo "--- Environment Check Report ---"
  echo "Date: $(date -u)"
  echo "\n--- OS/Kernel ---"
  cat /etc/os-release
  uname -a
  echo "\n--- Path ---"
  pwd
  echo "\n--- Git Config ---"
  echo "core.autocrlf = $(git config --get core.autocrlf)"
  echo "\n--- Tool Versions ---"
  python3 --version
  uv --version
  echo "\n--- Python Dependencies (uv pip list) ---"
  uv pip list
) > "$ENV_CHECK_FILE"

# --- Final Instructions ---
echo ""
echo "SUCCESS: Bootstrap complete. Report saved to '$ENV_CHECK_FILE'."
echo "ACTION REQUIRED: Please open the '.env' file and add your GOOGLE_API_KEY."

