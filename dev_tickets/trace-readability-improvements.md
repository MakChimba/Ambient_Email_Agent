# Ticket: Improve LangSmith Trace Readability

## Overview
LangSmith traces for the Gmail HITL + memory workflow remain difficult to scan. Runs surface raw email JSON in the input column and a lowercase `ai:` summary line in the output column, making it hard for reviewers to understand a case without expanding nested messages. Our current wiring follows default tracing behaviour, but it skips a few LangGraph best practices for metadata and summary presentation.

## Proposed Work
- **Config metadata & naming**: Before `email_assistant.invoke`, enrich the LangGraph `config` with `run_name`, `tags`, and metadata. Promote the slugify helper currently living in `tests/conftest.py:33-45` into a shared runtime utility (e.g., `email_assistant/tracing.py` or `email_assistant/utils.py`) so both tests and production entrypoints call the same implementation. Use it to derive names like `f"{AGENT_MODULE}-{slugified_subject}"`; fall back to the dataset’s `email_name` or, if empty, `f"{AGENT_MODULE}-{thread_uuid[:8]}"` to avoid collisions. Tags should follow `["pytest", AGENT_MODULE]`, append the dataset case when available, and leave room to include triage outcomes later. Metadata should include `{"dataset_case": email_name or None, "email_subject": subject or None}`, converting empty strings back to `None` so LangSmith filters stay useful.
- **Runtime coverage**: Today only the pytest harness injects this metadata. Mirror the same enrichment in other launchers that invoke `overall_workflow`—notably `scripts/run_real_outputs.py:97`, `scripts/run_reminder_evaluation.py:84`, and `src/email_assistant/eval/evaluate_triage.py:33-63`. Document that any LangGraph Studio adapters or future CLI utilities must do the same so run labelling stays consistent outside tests.
- **Readable inputs**: When calling `_safe_log_inputs`, keep the existing minimal dict but also add an `email_markdown` snapshot generated via `format_gmail_markdown(...)`. Cap the stored markdown at ~4096 characters (post-formatting) by truncating on a word boundary and appending ` …` so reviewers can tell it was shortened. Preserve the original dict structure alongside the new field; note in comments that `format_email_markdown` is the future fallback if new providers land.
- **Summary via assistant_reply**: Update `mark_as_read_node` in `email_assistant_hitl_memory_gmail.py` to stop appending a synthetic `AIMessage` and emit the final summary solely through the `assistant_reply` field. This removes the lowercase `ai:` role prefix without losing evaluator signals.
- **Documentation**: Capture the new tracing defaults and LangGraph best practices in `README_LOCAL.md` (or a dedicated tracing guide). Reference LangGraph’s “Enable LangSmith Tracing” guide, document the run-name/tags convention plus metadata schema (including the shared slugify helper and fallback rules), the 4 KB markdown guard, and call out that Gmail is the only supported provider today. Mention the non-pytest launchers that must inject metadata.

## Acceptance Criteria
- New LangSmith runs (manual or pytest-driven) display the configured `run_name`, tags, and metadata with subject/case details following the convention above.
- The Input column renders the formatted markdown email context capped at ~4 KB; raw dicts are no longer exposed by default.
- The Output column shows the summary text without an `ai:` prefix, and the existing automated suite (`pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`) still passes.
- Repository documentation reflects the tracing best practices, the Gmail-specific assumptions, the slugified run naming, and the assistant-reply-only summary approach, including guidance for non-pytest launchers to adopt the same metadata enrichment.

## References
- Team guidance on run naming/tags and metadata fallbacks (internal notes, June 2025).
- Relevant code: `tests/agent_test_utils.py:116-128`, `tests/test_response.py:295-352`, `tests/conftest.py:33-45`, `scripts/run_real_outputs.py:97`, `scripts/run_reminder_evaluation.py:84`, `src/email_assistant/eval/evaluate_triage.py:33-63`, `src/email_assistant/email_assistant_hitl_memory_gmail.py:1175-1214`.
- LangGraph docs: “Enable LangSmith Tracing” how-to (`/langchain-ai/langgraph` → tracing topic).
