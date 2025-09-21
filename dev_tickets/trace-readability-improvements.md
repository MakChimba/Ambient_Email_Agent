# Ticket: Improve LangSmith Trace Readability

## Overview
LangSmith traces for the Gmail HITL + memory workflow remain difficult to scan. Runs surface raw email JSON in the input column and a lowercase `ai:` summary line in the output column, making it hard for reviewers to understand a case without expanding nested messages. Our current wiring follows default tracing behaviour, but it skips a few LangGraph best practices for metadata and summary presentation.

## Proposed Work
- **Tracing helpers**: Introduce `email_assistant/tracing.py` (or similar) to expose shared `slugify_for_traces(...)` and `truncate_markdown(...)` utilities. Port the inline helper from `tests/conftest.py:33-45` into this module so pytest and runtime paths call the same implementation. The truncation helper should cap markdown produced by `format_gmail_markdown(...)` to ~4096 characters, trimming on a word boundary and appending ` …` when shortened.
- **Config metadata & naming**: Before `email_assistant.invoke`, enrich the LangGraph `config` with `run_name`, `tags`, and metadata. Use the shared slugify helper to derive run names, preferring the dataset `email_name`; if absent, fall back to the parsed subject, then the sender, and finally `thread_uuid[:8]`. Ensure every string promoted for naming is sanitized. Tags should follow `["pytest", AGENT_MODULE]`, append the dataset case when available, and leave room to include triage outcomes later. Metadata should include `{"dataset_case": email_name or None, "email_subject": subject or None, "email_sender": sender or None, "thread_uuid_prefix": thread_uuid[:8] or None}`, with blanks normalised to `None` so LangSmith filters stay useful and fallback choices remain visible in LangSmith.
- **Runtime coverage**: Mirror the same config enrichment in non-pytest launchers (`scripts/run_real_outputs.py`, `scripts/run_reminder_evaluation.py`, `src/email_assistant/eval/evaluate_triage.py`). Each entrypoint that loops over emails must clone the base config inside the loop so every run gets a fresh thread id, run name, tags, and metadata derived from the helper; avoid reusing a single thread config across examples.
- **Readable inputs**: When calling `_safe_log_inputs`, keep the existing minimal dict but also add an `email_markdown` snapshot generated via `format_gmail_markdown(...)`. Pass the markdown through the shared truncation helper before logging to LangSmith. Preserve the original dict structure alongside the new field, and note in comments that `format_email_markdown` is the future fallback if new providers land.
- **Summary via assistant_reply**: Update `mark_as_read_node` in `email_assistant_hitl_memory_gmail.py` to stop appending a synthetic `AIMessage`. Emit the final summary solely through the structured `assistant_reply` field while continuing to populate `tool_trace` and `email_markdown`.
- **Documentation**: Capture the new tracing defaults and LangGraph best practices in `README_LOCAL.md` (or a dedicated tracing guide). Reference LangGraph’s “Enable LangSmith Tracing” guide, the shared helper module, the run-name/tag convention with fallback order, the 4 KB markdown guard, and the requirement that non-pytest launchers perform per-email config cloning.

## Acceptance Criteria
- New LangSmith runs (manual or pytest-driven) display the configured `run_name`, tags, and metadata, exposing dataset/subject/sender/thread UUID details per the shared fallback order, and each email processed in a loop produces a unique run config.
- The Input column renders the formatted markdown email context capped at ~4 KB via the shared truncation helper; raw dicts are no longer exposed by default.
- The Output column shows the summary text without an `ai:` prefix, and the existing automated suite (`pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`) still passes.
- The shared tracing helper is consumed by both tests and runtime logging paths for slugification and markdown truncation so Studio, pytest, and scripts produce consistent traces.
- Repository documentation reflects the tracing best practices, the Gmail-specific assumptions, the helper module, the assistant-reply-only summary approach, and guidance for non-pytest launchers to adopt per-email metadata enrichment.

## References
- Team guidance on run naming/tags and metadata fallbacks (internal notes, June 2025).
- Relevant code: `tests/agent_test_utils.py:116-128`, `tests/test_response.py:77-141`, `tests/conftest.py:33-45`, `scripts/run_real_outputs.py:92-140`, `scripts/run_reminder_evaluation.py:84-120`, `src/email_assistant/eval/evaluate_triage.py:33-63`, `src/email_assistant/email_assistant_hitl_memory_gmail.py:1175-1231`.
- New helper location: `src/email_assistant/tracing.py` (to be introduced).
- LangGraph docs: “Enable LangSmith Tracing” how-to (`/langchain-ai/langgraph` → tracing topic).
