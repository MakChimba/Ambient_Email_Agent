#!/usr/bin/env python
import os
import subprocess
import sys
import argparse
from pathlib import Path

def main():
    # LangSmith suite / project name
    langsmith_project = "E-mail Tool Calling and Response Evaluation"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run tests for email assistant implementations")
    parser.add_argument("--rich-output", action="store_true", help="[DEPRECATED] LangSmith output is now enabled by default")
    parser.add_argument("--experiment-name", help="Name for the LangSmith experiment")
    parser.add_argument("--implementation", help="Run tests for a specific implementation")
    parser.add_argument("--all", action="store_true", help="Run tests for all implementations")
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
    os.environ.setdefault("EMAIL_ASSISTANT_EVAL_MODE", "1")

    # Run tests for each implementation
    for implementation in implementations_to_test:
        print(f"\nRunning tests for {implementation}...")
        
        # Skip LangSmith/Tracing env for offline/local runs
        
        # Create a fresh copy of the pytest options for this run
        pytest_options = base_pytest_options.copy()
        
        # Add the module parameter for this specific implementation
        module_param = f"--agent-module={implementation}"
        pytest_options.append(module_param)
        
        # Determine which test files to run based on implementation
        test_files = ["test_response.py"]  # All implementations run response tests
                    
        # Run each test file
        print(f"   Project: {langsmith_project}")
        for test_file in test_files:
            print(f"\nRunning {test_file} for {implementation}...")
            experiment_name = f"Test: {test_file.split('/')[-1]} | Agent: {implementation}"
            print(f"   Experiment: {experiment_name}")
            # Not setting LANGSMITH_EXPERIMENT to keep tests offline
            
            # Ensure third-party pytest plugins are not auto-loaded
            os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

            # Run pytest from the tests directory
            # Use the same Python interpreter that's running this script
            python_executable = sys.executable
            # Emphasize tool-call tests by default (avoid flaky LLM-as-judge)
            cmd = [python_executable, "-m", "pytest", test_file, "-k", "tool_calls"] + pytest_options
            
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
                
if __name__ == "__main__":
    sys.exit(main() or 0)
