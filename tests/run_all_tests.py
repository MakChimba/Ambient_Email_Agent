#!/usr/bin/env python
import os
import subprocess
import sys
import argparse
from pathlib import Path

from email_assistant.tracing import AGENT_PROJECT, init_project

def main():
    init_project(AGENT_PROJECT)
    # LangSmith suite / project name
    langsmith_project = "E-mail Tool Calling and Response Evaluation"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run tests for email assistant implementations")
    parser.add_argument("--rich-output", action="store_true", help="[DEPRECATED] LangSmith output is now enabled by default")
    parser.add_argument("--experiment-name", help="Name for the LangSmith experiment")
    parser.add_argument("--implementation", help="Run tests for a specific implementation")
    parser.add_argument("--all", action="store_true", help="Run tests for all implementations")
    parser.add_argument(
        "--offline-eval",
        action="store_true",
        help="Enable EMAIL_ASSISTANT_EVAL_MODE=1 to synthesize tool calls deterministically",
    )
    args = parser.parse_args()
    
    # Base pytest options (kept offline-friendly)
    base_pytest_options = ["-v", "--disable-warnings"]
    # The --langsmith-output flag is intentionally omitted to avoid requiring
    # the LangSmith pytest plugin and native zstandard dependency during local runs
    
    # Define available implementations
    # Note: email_assistant_hitl and email_assistant_hitl_memory (non-Gmail) remain excluded by default because:
    # 1. They include the Question tool not covered by current non-Gmail dataset ground truth
    # 2. They would require updated datasets and resume handling logic to be reliably testable
    # The Gmail HITL+memory agent is supported via a Gmail-specific dataset and tool name mapping in tests.
    implementations = [
        "email_assistant",                      # baseline workflow agent
        "email_assistant_hitl_memory_gmail",    # production-target Gmail HITL+memory agent
    ]
    
    # Determine which implementations to test
    if args.implementation:
        if args.implementation in implementations:
            implementations_to_test = [args.implementation]
        else:
            print(f"Error: Unknown implementation '{args.implementation}'")
            print(f"Available implementations: {', '.join(implementations)}")
            return 1
    elif args.all:
        implementations_to_test = implementations
    else:
        # Default to testing the production-target Gmail agent
        # Use --implementation to override, or --all to include baseline too
        implementations_to_test = ["email_assistant_hitl_memory_gmail"]
    
    # Environment keys are expected via .env; no OpenAI key required
    # Stabilize agent behavior for CI-like runs
    os.environ.setdefault("HITL_AUTO_ACCEPT", "1")
    os.environ.setdefault("EMAIL_ASSISTANT_SKIP_MARK_AS_READ", "1")

    if args.offline_eval:
        os.environ["EMAIL_ASSISTANT_EVAL_MODE"] = "1"
        # Provide a placeholder key so Google clients avoid looking up ADC credentials.
        os.environ.setdefault("GOOGLE_API_KEY", "offline-test-key")
    else:
        os.environ.setdefault("EMAIL_ASSISTANT_EVAL_MODE", "0")
        if not os.getenv("GOOGLE_API_KEY"):
            print(
                "WARNING: GOOGLE_API_KEY not set. Tests depending on live Gemini calls will skip or fail."
            )

    # Keep notebook execution short/deterministic by default.
    os.environ.setdefault("NB_TEST_MODE", "1")

    gmail_core_tests = [
        ("test_response.py", ["-k", "tool_calls"]),
        ("test_spam_flow.py", []),
        ("test_reminders.py", []),
    ]

    # LangSmith-backed reminder suite is optional; include when credentials exist.
    if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
        gmail_core_tests.append(("test_reminders_langsmith.py", []))

    gmail_notebook_tests = [("test_notebooks.py", [])]

    implementation_test_matrix = {
        "email_assistant": {
            "core": [("test_response.py", ["-k", "tool_calls"])],
            "notebook": [],
        },
        "email_assistant_hitl_memory_gmail": {
            "core": gmail_core_tests,
            "notebook": gmail_notebook_tests,
        },
    }

    # Run tests for each implementation
    for implementation in implementations_to_test:
        print(f"\nRunning tests for {implementation}...")

        # Create a fresh copy of the pytest options for this run
        pytest_options = base_pytest_options.copy()

        # Add the module parameter for this specific implementation
        module_param = f"--agent-module={implementation}"
        pytest_options.append(module_param)

        configured_tests = implementation_test_matrix.get(implementation, {})
        core_tests = configured_tests.get("core", [])
        notebook_tests = configured_tests.get("notebook", [])

        if not (core_tests or notebook_tests):
            print(f"WARNING: No tests configured for implementation '{implementation}'.")
            continue

        def run_suite(label: str, tests: list[tuple[str, list[str]]]) -> None:
            if not tests:
                return
            print(f"\n-- {label} --")
            print(f"   Project: {langsmith_project}")
            for test_file, extra_args in tests:
                print(f"\nRunning {test_file} for {implementation}...")
                experiment_name = f"Test: {test_file.split('/')[-1]} | Agent: {implementation}"
                print(f"   Experiment: {experiment_name}")
                # Not setting LANGSMITH_EXPERIMENT to keep tests offline

                # Ensure third-party pytest plugins are not auto-loaded
                os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

                # Run pytest from the tests directory using the current Python interpreter
                python_executable = sys.executable
                cmd = [python_executable, "-m", "pytest", test_file] + extra_args + pytest_options

                # Change to the script's directory to ensure correct imports
                script_dir = Path(__file__).parent
                cwd = os.getcwd()
                os.chdir(script_dir)
                result = subprocess.run(cmd, capture_output=True, text=True)
                os.chdir(cwd)  # Restore original working directory

                # Print test output
                print(result.stdout)
                if result.stderr:
                    print(result.stderr)

        run_suite("Core coverage", core_tests)
        run_suite("Notebook coverage", notebook_tests)
                
if __name__ == "__main__":
    sys.exit(main() or 0)
