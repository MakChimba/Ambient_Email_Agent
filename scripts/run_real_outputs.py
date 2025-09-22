#!/usr/bin/env python
"""Run the email assistant on test inputs and print real outputs.

Usage:
  python scripts/run_real_outputs.py \
      --agent-module email_assistant_hitl_memory \
      --max 5 \
      --respond-only \
      --auto-accept

Notes:
  - Requires GOOGLE_API_KEY and prefers GEMINI_MODEL=gemini-2.5-pro
  - For HITL flows, use --auto-accept or set HITL_AUTO_ACCEPT=1
  - Uses the non-Gmail dataset; no Gmail auth needed.
"""
from __future__ import annotations

import argparse
import os
import importlib
from typing import Any, Dict, List, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from email_assistant.tracing import (
    AGENT_PROJECT,
    init_project,
    invoke_with_root_run,
    summarize_email_for_grid,
)


def _load_dataset(agent_module: str):
    """Load the appropriate dataset based on agent module name."""
    if "gmail" in agent_module:
        ds_mod = importlib.import_module("email_assistant.eval.email_gmail_dataset")
    else:
        ds_mod = importlib.import_module("email_assistant.eval.email_dataset")
    return (
        ds_mod.email_inputs,
        ds_mod.email_names,
        getattr(ds_mod, "triage_outputs_list", ["respond"] * len(ds_mod.email_inputs)),
    )


def _compile_agent(agent_module: str):
    """Compile the selected agent with memory + checkpointer when applicable."""
    module = importlib.import_module(f"email_assistant.{agent_module}")
    checkpointer = MemorySaver()
    store = InMemoryStore()
    try:
        graph = module.overall_workflow.compile(checkpointer=checkpointer, store=store)
    except Exception:
        # Fallback for agents that don't take store
        graph = module.overall_workflow.compile(checkpointer=checkpointer)
        store = None
    return graph, store


def _extract_tool_calls(messages: List[Any]) -> List[str]:
    names: List[str] = []
    for m in messages:
        tc = getattr(m, "tool_calls", None)
        if tc:
            for t in tc:
                names.append(str(t.get("name")).lower())
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-module", default="email_assistant_hitl_memory", help="Agent module under email_assistant.*")
    ap.add_argument("--max", type=int, default=5, help="Max examples to run")
    ap.add_argument("--respond-only", action="store_true", help="Only run examples with triage=='respond'")
    ap.add_argument("--auto-accept", action="store_true", help="Auto-accept HITL prompts")
    args = ap.parse_args()

    init_project(AGENT_PROJECT)

    if args.auto_accept:
        os.environ.setdefault("HITL_AUTO_ACCEPT", "1")

    # Ensure model is 2.5 series with normalized name
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    if ":" in model:
        model = model.split(":", 1)[1]
    if model.startswith("models/"):
        model = model.split("/", 1)[1]
    os.environ["GEMINI_MODEL"] = model

    # Load dataset and compile agent
    email_inputs, email_names, triage_list = _load_dataset(args.agent_module)
    agent, store = _compile_agent(args.agent_module)

    # Thread config
    import uuid
    thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

    count = 0
    for i, (inp, name, triage) in enumerate(zip(email_inputs, email_names, triage_list)):
        if args.respond_only and str(triage).lower() != "respond":
            continue
        print(f"\n=== [{i}] {name} | triage={triage} ===")
        payload = {"email_input": inp}
        summary = summarize_email_for_grid(inp)

        def _invoke_agent():
            return agent.invoke(payload, config=thread_config)

        try:
            result = invoke_with_root_run(
                _invoke_agent,
                root_name=f"agent:{args.agent_module}",
                input_summary=summary,
            )
        except Exception as e:
            print(f"ERROR during invoke: {e}")
            continue

        # Fetch final state messages
        state = agent.get_state(thread_config)
        values: Dict[str, Any] = state.values if hasattr(state, "values") else state
        messages = values.get("messages", [])
        tools = _extract_tool_calls(messages)
        print(f"Tool calls: {tools}")

        # Print last assistant message tool calls and tool observations
        for m in messages:
            role = getattr(m, "type", getattr(m, "role", ""))
            if role == "tool" or getattr(m, "role", "") == "tool":
                print(f"[tool] {getattr(m, 'content', '')}")

        # Best-effort: surface write_email content if present in tool call args
        try:
            from langchain_core.messages import AIMessage
            for m in reversed(messages):
                if isinstance(m, AIMessage) and m.tool_calls:
                    for tc in m.tool_calls:
                        if str(tc.get("name")).lower() in {"write_email", "send_email_tool"}:
                            args = tc.get("args", {})
                            to = args.get("to") or args.get("email_address")
                            subject = args.get("subject")
                            content = args.get("content") or args.get("response_text")
                            print("Draft summary:")
                            print(f"  To: {to}")
                            print(f"  Subject: {subject}")
                            if content:
                                preview = str(content).strip().splitlines()
                                preview = " ".join(preview[:5])
                                print(f"  Body (preview): {preview[:300]}")
                            break
                    break
        except Exception:
            pass

        count += 1
        if count >= args.max:
            break


if __name__ == "__main__":
    main()
