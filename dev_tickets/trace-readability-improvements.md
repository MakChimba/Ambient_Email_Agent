# Ticket: Improve LangSmith Trace Readability

## Overview
LangSmith traces for the Gmail HITL + memory workflow remain difficult to scan. Runs surface raw email JSON in the input column and a lowercase `ai:` summary line in the output column, making it hard for reviewers to understand a case without expanding nested messages. Our current wiring follows default tracing behaviour, but it skips a few LangGraph best practices for metadata and summary presentation.

## Proposed Work
- **Config metadata & naming**: Before `email_assistant.invoke`, enrich the LangGraph `config` with `run_name`, `tags`, and metadata. Use `run_name = f"{AGENT_MODULE}:{email_name}"` whenever the dataset provides `email_name`; otherwise fall back to the Gmail subject (from `parse_gmail`) and, if that’s missing, `f"{AGENT_MODULE}:{thread_uuid}"` to avoid collisions. Tags should follow `["pytest", AGENT_MODULE]`, append the dataset case when available, and leave room to include triage outcomes later.
- **Metadata payload**: Attach `{"dataset_case": email_name or None, "email_subject": subject or None}`. `parse_gmail` already coerces missing fields to `""`; convert empty strings back to `None` so LangSmith filters stay useful.
- **Readable inputs**: When calling `_safe_log_inputs`, keep the existing minimal dict but also add `email_markdown` generated via `format_gmail_markdown(...)`. Since this agent is Gmail-only today, we don’t need cross-provider branching yet—note in comments that `format_email_markdown` is the future fallback if new providers land.
- **Summary via assistant_reply**: Update `mark_as_read_node` in `email_assistant_hitl_memory_gmail.py` to stop appending a synthetic `AIMessage` and emit the final summary solely through the `assistant_reply` field. This removes the lowercase `ai:` role prefix without losing evaluator signals.
- **Documentation**: Capture the new tracing defaults and LangGraph best practices in `README_LOCAL.md` (or a dedicated tracing guide). Reference LangGraph’s “Enable LangSmith Tracing” guide, document the run-name/tags convention plus metadata schema, and call out that Gmail is the only supported provider today.

## Acceptance Criteria
- New LangSmith runs (manual or pytest-driven) display the configured `run_name`, tags, and metadata with subject/case details following the convention above.
- The Input column renders the formatted markdown email context; raw dicts are no longer exposed by default.
- The Output column shows the summary text without an `ai:` prefix, and the existing automated suite (`pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`) still passes.
- Repository documentation reflects the tracing best practices, the Gmail-specific assumptions, and specifically mentions the assistant-reply-only summary approach.

## References
- Team guidance on run naming/tags and metadata fallbacks (internal notes, June 2025).
- Relevant code: `tests/agent_test_utils.py:116-128`, `tests/test_response.py:295-352`, `src/email_assistant/email_assistant_hitl_memory_gmail.py:1175-1214`.
- LangGraph docs: “Enable LangSmith Tracing” how-to (`/langchain-ai/langgraph` → tracing topic).
