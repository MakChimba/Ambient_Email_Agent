# LangSmith Trace Readability — Unified Implementation Plan

**Status:** P1 • **Owners:** Eng • **Areas:** LangGraph, LangSmith, Tool wrappers, Evaluators  
**Summary:** Eliminate raw JSON and visible markdown tokens from the Runs table. Inputs and Outputs must always be short, plain‑text strings (never dicts). Keep rich markdown and raw payloads only in metadata/extra. Apply the same policy to parent and child runs (LLM + tools) across AGENT, JUDGE, Studio, tests, and live.

---

## Motivation

- AGENT runs sometimes show the original email dict in **Input**.
- JUDGE runs show `{"email_markdown": "…"}` (still JSON) and literal `**` in grid cells.
- Some **Outputs** still surface raw `function_call` JSON or markdown tokens.
- Parent Input/Output can look duplicative for devs scanning traces.

**Goal:** Fast, consistent at‑a‑glance review in the Runs list without expanding nodes.

---

## Design Decisions (non‑negotiable)

1. **First‑write‑is‑pretty:** The very first value we send to LangSmith for `run.inputs` (parent or child) is already the **human‑readable string**. No post‑hoc conversion for Inputs.
2. **String, not dict:** Grid cells show **strings only**. Dicts render as JSON—do not use them for Inputs/Outputs.
3. **Plain text in grid; markdown in metadata:** Strip formatting tokens for the grid; store the rich markdown copy in `metadata.email_markdown` (capped).
4. **Pre‑map dataset examples:** Convert AGENT/JUDGE examples to the display string **before** tracing starts.
5. **Child‑run parity:** Apply the same summarization to tool + LLM child runs; never log raw `arguments`/message arrays in child `inputs`.
6. **Action‑focused Outputs:** A two‑line, plain‑text summary: outcome token + short rationale/snippet. No `ai:` prefix, no JSON, no `**`.
7. **Fallback updater stays:** A `finally`‑path, retrying updater patches any stragglers (both inputs and outputs) but should rarely trigger.
8. **Guardrails in CI:** Hide inputs in CI while rolling out.

---

## Work Items

### 1) Shared helpers (`src/email_assistant/tracing.py`)
Create/extend a single module used **everywhere** (runners, nodes, tests, Studio):

- `strip_markdown_to_text(md: str) -> str`  
  Remove `** _ # >` etc.; preserve newlines; collapse long whitespace.
- `summarize_email_for_grid(email) -> str`  
  Returns: `Subject — From → To\n<first ~800 chars of body snippet>`.  
  - Must be **plain text** (use `strip_markdown_to_text`).  
  - Hard cap ~800 chars with ellipsis; no code fences; no braces if possible.
- `summarize_tool_call_for_grid(name: str, args: Mapping) -> str`  
  Example: `[tool] send_email(to=alice@…, subject="Quarterly…", attachments=2)`  
  - Selective fields + counts only. Never embed raw `args` JSON.
- `summarize_llm_for_grid(messages|prompt) -> str`  
  Example: `4 msgs | last user: "please draft a reply about…"`.
- `truncate_markdown(md: str, max_chars=4096) -> str`  
  Keep rich copy for the Run detail view. Unicode‑aware trim with ellipsis.
- (Keep existing slug/tag/name helpers; export them here for single‑source usage.)

> Note: The grid strings must be **type `str`**. Anything structured belongs in `metadata`/`extra`.

---

### 2) Parent run priming (every entrypoint)
- When starting the top‑level run, set **`inputs` to the string** from `summarize_email_for_grid`.
- Save **raw input** under `extra.raw_input` (or a reference) and **markdown** under `metadata.email_markdown = truncate_markdown(...)`.
- Keep run naming/tags/metadata conventions you already implemented; ensure they’re produced via the shared helpers to avoid drift.

---

### 3) Dataset input mapping (AGENT + JUDGE + evaluators + tests)
- Before any tracing, map each example into the display **string** via `summarize_email_for_grid`.
- Pass that string as `inputs` when creating/starting the run.
- Stash the original example under `metadata.example_raw` (not in `inputs`).
- Apply in:
  - `scripts/run_real_outputs.py`
  - `scripts/run_reminder_evaluation.py`
  - `src/email_assistant/eval/evaluate_triage.py`
  - Pytest harnesses (`tests/test_response.py`, fixtures)
- Ensure each loop iteration clones a fresh config (unique thread id + run name + tags + metadata).

---

### 4) Child runs (tools + LLM)
- **Tool wrappers (e.g., Gmail):** The tool run’s `inputs` must be a **single string** from `summarize_tool_call_for_grid`. Put raw payload in `metadata.tool_raw`.
- **Function/tool args emitted by LLM:** Replace raw `arguments` dict with a one‑liner string (selected fields + counts). The raw dict goes to metadata only.
- **LLM runs:** `inputs` is `summarize_llm_for_grid`; do not log full message arrays as a child run input.

---

### 5) Output policy (developer‑friendly)
- Outputs are concise, plain‑text strings:
  - Line 1: Outcome token — e.g., `[reply]`, `[no_action]`, `[tool_call] send_email`.
  - Line 2: Short justification or first ~160–240 chars of reply.
- No `ai:` prefix, no markdown tokens, no raw tool JSON.
- Keep full reply + tool trace in `outputs.details`/`metadata` as needed.

---

### 6) Fallback updater (`maybe_update_run_io`)
- Run in a `finally` block on all paths.
- If Inputs/Outputs are not strings or contain raw JSON braces, patch both to summarized strings.
- Retry 2–3 times with jitter; ensure you target the **root** run id.
- Optional: patch relevant child runs if a framework emitted early (goal is to avoid this).

---

### 7) CI guardrails
- In CI/pytest set:
  - `LANGSMITH_HIDE_INPUTS=true`
  - (Optional) `LANGSMITH_HIDE_OUTPUTS=true`
- Keep visible in dev/prod once green.

---

## Examples

**Input (grid string):**
Quick question about API documentation — Alice Smith alice@company.com
 → Lance Martin lance@company.com

Hi Lance, I was reviewing the API documentation for the new authentication service...


**Output (grid string):**
[reply] Drafted clarification email via template A
Explained /auth/refresh and /auth/validate, asked if endpoints were intentionally omitted; offered to update docs upon confirmation.


**Tool child run (grid string):**

---

## Acceptance Criteria

1. **Inputs (all environments)**  
   - Runs list shows a **plain string** (no JSON braces, no markdown tokens).  
   - ≤ ~800 chars; includes `Subject — From → To` + body snippet.
2. **Outputs**  
   - Plain‑text, 1–2 lines; includes outcome token; no `ai:`/JSON/markdown tokens.
3. **Child runs**  
   - Tool + LLM child `inputs` are plain strings; no raw `arguments` or full message arrays in grid.
4. **Metadata**  
   - `metadata.email_markdown` contains a ≤4 KB rich snapshot.  
   - Raw payloads live in `metadata`/`extra` only.
5. **Dataset runners**  
   - AGENT + JUDGE always pre‑map examples to string inputs; unique per‑email config per run.
6. **Stability**  
   - Fallback updater corrects any regressions (both inputs and outputs), with retries.

---

## Test Plan

**Unit**
- `strip_markdown_to_text` removes tokens, preserves newlines.
- `summarize_email_for_grid` returns string ≤ 800 chars, no braces, no markdown.
- `summarize_tool_call_for_grid` exposes selected fields + counts only.

**Integration (pytest)**
- AGENT + JUDGE suites: assert `type(run.inputs) is str`, **no `{`** and **no `**`** in Inputs/Outputs.
- Tool path vs. no‑tool path produce clean Outputs.
- Error paths still end with tidy Inputs/Outputs (updater verified).

**Manual**
- In LangSmith Runs (AGENT/JUDGE/Studio/live), scan that all rows meet criteria; expand a run to confirm markdown + raw are present in metadata, not grid.

---

## Rollout Steps

1. Land helpers + parent priming.
2. Convert dataset runners to **pre‑trace mapping**.
3. Wrap tool + LLM child runs.
4. Keep updater with retries; enable CI hide‑inputs during rollout.
5. Remove CI hide‑inputs once validated across suites.

---

## Risks & Mitigations

- **Loss of debug detail in grid** → Full markdown + raw payloads live in metadata/extra and in the detailed run view.  
- **Framework double‑logging/races** → First‑write‑is‑pretty + finally‑path updater with retries.  
- **Perf of markdown stripping** → Lightweight regex; early caps on string length.

---

## File/Code Touchpoints

- `src/email_assistant/tracing.py` — add/export summarizers + strip/trim helpers (single source of truth).  
- Tool wrappers (Gmail, etc.) — use `summarize_tool_call_for_grid` for child `inputs`, stash raw in metadata.  
- LLM wrappers — summarize prompts; never log full message arrays in child `inputs`.  
- Runners: `scripts/run_real_outputs.py`, `scripts/run_reminder_evaluation.py`, `src/email_assistant/eval/evaluate_triage.py` — pre‑trace mapping + fresh per‑email configs.  
- Tests/fixtures — import helpers; add assertions listed in **Test Plan**.

---

## Notes

- Keep your existing run naming/tagging conventions; just ensure all entrypoints import the same helpers.  
- The plan intentionally separates **developer‑scan strings** (grid) from **rich debugging artifacts** (metadata/extra).

