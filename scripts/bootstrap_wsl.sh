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

# Ensure Gemini and LangSmith vars are present (idempotent)
ENV_SENTINEL="# LangGraph Project Configuration"
if ! grep -Fq "$ENV_SENTINEL" .env; then
  cat <<'EOF' >> .env

# LangGraph Project Configuration
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-2.5-pro
# Optional: LangSmith for tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=ambient-email-agent-gemini
EOF
  echo "INFO: Added LangGraph defaults to .env."
fi

ensure_env_var() {
  local key="$1"
  local default_value="$2"
  if ! grep -Eq "^${key}=" .env; then
    printf '%s=%s\n' "$key" "$default_value" >> .env
    echo "INFO: Added missing $key to .env."
  fi
}

ensure_env_var "GEMINI_MODEL" "gemini-2.5-pro"
ensure_env_var "LANGSMITH_TRACING" "true"

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

# 8. Shell auto-activation hook
echo "INFO: Configuring shell auto-activation..."
BASHRC="$HOME/.bashrc"
AUTO_SNIPPET_START="# >>> standalone-email-agent auto env >>>"
AUTO_SNIPPET_END="# <<< standalone-email-agent auto env <<<"
REPO_PATH="$(pwd)"

if [ ! -f "$BASHRC" ]; then
  touch "$BASHRC"
fi

python3 - "$BASHRC" "$AUTO_SNIPPET_START" "$AUTO_SNIPPET_END" <<'PY'
import pathlib
import sys
start = sys.argv[2]
end = sys.argv[3]
path = pathlib.Path(sys.argv[1])
if not path.exists():
    sys.exit(0)
text = path.read_text()
start_idx = text.find(start)
end_idx = text.find(end)
if start_idx != -1 and end_idx != -1:
    end_idx += len(end)
    if end_idx < len(text) and text[end_idx] == '\n':
        end_idx += 1
    text = text[:start_idx] + text[end_idx:]

# Remove legacy auto-activate function
legacy_func_pattern = "\nauto_activate_email_assistant()"  # cheap sentinel
if legacy_func_pattern in text:
    start = text.find(legacy_func_pattern)
    end = text.find('\n}\n\ncase ":${PROMPT_COMMAND:-}:"', start)
    if end != -1:
        # include the case/esac block and trailing newline
        end_block = text.find("esac", end)
        if end_block != -1:
            end_block = text.find('\n', end_block)
            if end_block == -1:
                end_block = len(text)
            else:
                end_block += 1
            text = text[:start] + text[end_block:]

# Remove legacy env snippet
legacy_start = "# >>> email-assistant autoenv >>>"
legacy_end = "# <<< email-assistant autoenv <<<"
ls_idx = text.find(legacy_start)
le_idx = text.find(legacy_end)
if ls_idx != -1 and le_idx != -1:
    le_idx += len(legacy_end)
    if le_idx < len(text) and text[le_idx] == '\n':
        le_idx += 1
    text = text[:ls_idx] + text[le_idx:]

path.write_text(text)
PY

cat <<EOF >> "$BASHRC"
$AUTO_SNIPPET_START
_standalone_email_agent_autoenv_apply_prompt_command() {
  case ":\${PROMPT_COMMAND:-}:" in
    *"_standalone_email_agent_autoenv;"*) ;;
    *)
      if [ -n "\${PROMPT_COMMAND:-}" ]; then
        PROMPT_COMMAND="_standalone_email_agent_autoenv;\${PROMPT_COMMAND}"
      else
        PROMPT_COMMAND="_standalone_email_agent_autoenv"
      fi
      ;;
  esac
}

_standalone_email_agent_autoenv() {
  local repo_path="$REPO_PATH"
  local venv_path="\${repo_path}/.venv"
  local env_file="\${repo_path}/.env"

  case "\$PWD" in
    "\$repo_path"|"\$repo_path"/*)
      if [ -f "\$venv_path/bin/activate" ] && [ "\${VIRTUAL_ENV:-}" != "\$venv_path" ]; then
        source "\$venv_path/bin/activate"
        _standalone_email_agent_autoenv_apply_prompt_command
      fi
      if [ -f "\$env_file" ]; then
        local env_mtime
        env_mtime=\$(stat -c %Y "\$env_file" 2>/dev/null || stat -f %m "\$env_file" 2>/dev/null || echo 0)
        if [ "\${_STANDALONE_EMAIL_AGENT_ENV_MTIME:-}" != "\$env_mtime" ]; then
          set -a
          source "\$env_file"
          set +a
          export _STANDALONE_EMAIL_AGENT_ENV_MTIME="\$env_mtime"
        fi
      fi
      ;;
    *)
      if [ "\${VIRTUAL_ENV:-}" = "\$venv_path" ] && command -v deactivate >/dev/null 2>&1; then
        deactivate >/dev/null 2>&1 || true
      fi
      ;;
  esac
}

_standalone_email_agent_autoenv_apply_prompt_command
_standalone_email_agent_autoenv
$AUTO_SNIPPET_END
EOF

echo "INFO: Ensured auto-activation hook is present in '$BASHRC'."

# --- Final Instructions ---
echo ""
echo "SUCCESS: Bootstrap complete. Report saved to '$ENV_CHECK_FILE'."
echo "ACTION REQUIRED: Please open the '.env' file and add your GOOGLE_API_KEY."
