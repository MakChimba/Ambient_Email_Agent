#!/usr/bin/env python
"""
Register a dataset in LangSmith from a local JSONL file.

Each line must be a JSON object with keys:
- inputs: dict (e.g., {"email_input": {...}})
- metadata: dict (optional)
- outputs: dict (optional; usually empty for reference datasets)

Usage:
  PYTHONPATH=src python scripts/register_dataset_from_jsonl.py \
    --jsonl datasets/experiment_gmail.jsonl \
    --dataset-name standalone.email_assistant.experiments

Environment:
  - LANGSMITH_API_KEY (required)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv, find_dotenv


def main() -> int:
    load_dotenv(find_dotenv(), override=False)
    try:
        from langsmith import Client
    except Exception as e:
        print(f"ERROR: langsmith client not available: {e}")
        return 1

    ap = argparse.ArgumentParser(description="Register dataset from JSONL")
    ap.add_argument("--jsonl", required=True, help="Path to JSONL file with examples")
    ap.add_argument("--dataset-name", required=True, help="LangSmith dataset name")
    args = ap.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        print("ERROR: LANGSMITH_API_KEY is not set.")
        return 1

    path = args.jsonl
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return 1

    client = Client()

    # Upsert dataset
    try:
        dataset = client.create_dataset(args.dataset_name, description=f"Registered from {os.path.basename(path)}")
        print(f"Created dataset: {dataset.id} ({dataset.name})")
    except Exception:
        dataset = client.read_dataset(dataset_name=args.dataset_name)
        print(f"Using existing dataset: {dataset.id} ({dataset.name})")

    created = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj: Dict[str, Any] = json.loads(line)
            except Exception as e:
                print(f"Skipping line {i}: invalid JSON: {e}")
                continue
            inputs = obj.get("inputs") or {}
            outputs = obj.get("outputs") or {}
            metadata = obj.get("metadata") or {}
            try:
                client.create_example(inputs=inputs, outputs=outputs, dataset_id=dataset.id, metadata=metadata)
                created += 1
            except Exception as e:
                print(f"Skipping example {i}: {e}")

    print(f"Done. Created {created} example(s) from {os.path.basename(path)} in dataset '{args.dataset_name}'.")
    print("Tip: In Studio, set graph=email_assistant_hitl_memory_gmail and use Agent Inbox for HITL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

