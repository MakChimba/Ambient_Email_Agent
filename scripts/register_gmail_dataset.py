#!/usr/bin/env python
"""
Register a Gmail-specific dataset in LangSmith Studio from the local dataset file.

Reads src/email_assistant/eval/email_gmail_dataset.py and creates (or updates)
examples in a LangSmith dataset so it can be used in Studio experiments.

Usage:
  PYTHONPATH=src python scripts/register_gmail_dataset.py \
    --dataset-name agents-from-scratch.gmail_test_response

Environment:
  - LANGSMITH_API_KEY: Required for authentication
  - Optionally source .env before running
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv, find_dotenv


def main() -> int:
    # Do not override pre-set env vars (e.g., LANGSMITH_API_KEY) with .env values
    load_dotenv(find_dotenv(), override=False)

    try:
        from langsmith import Client
    except Exception as e:
        print(f"ERROR: langsmith client not available: {e}")
        return 1

    # Ensure we can import the dataset module
    try:
        from email_assistant.eval import email_gmail_dataset as ds
    except Exception as e:
        print(
            "ERROR: Failed to import email_gmail_dataset. Ensure PYTHONPATH includes 'src'.\n"
            f"Details: {e}"
        )
        return 1

    ap = argparse.ArgumentParser(description="Register Gmail dataset in LangSmith")
    ap.add_argument(
        "--dataset-name",
        default="agents-from-scratch.gmail_test_response",
        help="LangSmith dataset name to create or update",
    )
    ap.add_argument(
        "--include-non-respond",
        action="store_true",
        help="Include all emails (not only triage == respond)",
    )
    args = ap.parse_args()

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("ERROR: LANGSMITH_API_KEY is not set. Aborting.")
        return 1

    client = Client()

    # Upsert dataset
    try:
        dataset = client.create_dataset(args.dataset_name, description="Gmail test dataset for agents-from-scratch")
        print(f"Created dataset: {dataset.id} ({dataset.name})")
    except Exception:
        dataset = client.read_dataset(dataset_name=args.dataset_name)
        print(f"Using existing dataset: {dataset.id} ({dataset.name})")

    # Build examples
    inputs_list = getattr(ds, "email_inputs")
    names_list = getattr(ds, "email_names")
    criteria_list = getattr(ds, "response_criteria_list")
    triage_list = getattr(ds, "triage_outputs_list")
    expected_calls_list = getattr(ds, "expected_tool_calls")

    created = 0
    for i, (email_input, name, criteria, triage, calls) in enumerate(
        zip(inputs_list, names_list, criteria_list, triage_list, expected_calls_list)
    ):
        if not args.include_non_respond and triage != "respond":
            continue

        # Inputs for the agent
        inputs: Dict[str, Any] = {"email_input": email_input}

        # Store expected tool calls and other metadata as reference
        metadata: Dict[str, Any] = {
            "response_criteria": criteria,
            "triage_output": triage,
            "expected_tool_calls": calls,
            "source": "email_gmail_dataset.py",
        }

        try:
            client.create_example(
                inputs=inputs,
                outputs={},  # reference tasks; model will produce outputs
                dataset_id=dataset.id,
                metadata=metadata,
            )
            created += 1
        except Exception as e:
            # If example exists or error occurs, report and continue
            print(f"Skipping example '{name}': {e}")

    print(f"Done. Created {created} new example(s) in dataset '{args.dataset_name}'.")
    print("Tip: Set HITL_AUTO_ACCEPT=true in Experiment env to avoid stalls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
