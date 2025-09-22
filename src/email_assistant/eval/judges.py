"""LLM-based evaluators for the email assistant.

This module exposes a Gemini 2.5 Flash powered "LLM-as-judge" utility that
scores a run's final reply and tool usage. The implementation mirrors the
LangSmith prompt shared in the project documentation so it can be reused both
locally (during pytest runs) and in LangSmith datasets/experiments.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langsmith import Client
from langsmith.evaluation import EvaluationResult, LangChainStringEvaluator, StringEvaluator
from langsmith.run_helpers import get_current_run_tree, traceable
from langsmith.schemas import Example, Run

from email_assistant.utils import extract_message_content, format_messages_string
from email_assistant.tracing import invoke_with_root_run, strip_markdown_to_text

__all__ = [
    "JudgeResult",
    "JudgeUnavailableError",
    "build_correctness_judge_chain",
    "build_tool_call_context",
    "run_correctness_judge",
    "serialise_messages",
    "create_langsmith_correctness_evaluator",
]


class JudgeUnavailableError(RuntimeError):
    """Raised when the judge cannot be executed (e.g., config missing)."""


class ToolIssue(BaseModel):
    tool: str = Field(default="", description="Name of the tool that was misused")
    why: str = Field(default="", description="Explanation of what went wrong")


class JudgeResult(BaseModel):
    overall_correctness: float
    verdict: Literal["pass", "fail"]
    content_alignment: int
    tool_usage: int
    missing_tools: List[str]
    incorrect_tool_uses: List[ToolIssue]
    evidence: List[str]
    notes: str

    def short_summary(self) -> str:
        return (
            f"verdict={self.verdict} overall={self.overall_correctness:.2f} "
            f"content={self.content_alignment}/5 tools={self.tool_usage}/5"
        )


SYSTEM_PROMPT = """You are “LLM Reviewer.” Return ONE valid JSON object only.\n\nJudge CORRECTNESS of an email agent’s final reply and its tool usage.\n\nAxes (0–5 each):\n• Content Alignment — Does the drafted reply resolve the sender’s request with correct commitments, deadlines, tone, and follow-through?\n• Tool Usage — Were the right tools used with correct arguments, in a valid order?\n\nPolicies:\n\n- Scheduling request → check_calendar_tool → schedule_meeting_tool → send_email_tool → Done. Confirm meeting duration matches the ask (allow ±10 minutes for “about” phrasing).\n- 90-minute planning (availability-only) → check_calendar_tool → send_email_tool → Done (NO scheduling). Ensure reply offers availability covering the requested window.\n- Conference invitations that only require interest/questions (e.g., TechConf) → send_email_tool → Done. Do NOT expect calendar tools unless the sender explicitly asks to schedule.\n- send_email_tool must include email_id and email_address, then Done.\n- Never call Done before drafting the reply.\n- Spam requires explicit confirmation before mark_as_spam_tool.\n- Tool-name normalization: treat as equivalent\n    - send_email_tool == write_email\n    - check_calendar_tool == check_calendar_availability\n    - schedule_meeting_tool == schedule_meeting\n- Exceptions:\n    - No‑reply/system “do not reply”: may end without send_email_tool (Done allowed).\n    - Spam (after explicit HITL confirmation): may end via mark_as_spam_tool without Done and without drafting a reply.\n- Question is correct when information is missing or consent is ambiguous.\n- For schedule_meeting_tool: start_time/end_time must be ISO; organizer_email required; timezone present or defaulted; duration should respect the request.\n- Tool-name/arg normalization: send_email_tool == write_email. When the name is write_email (normalized), accept the fields “to”, “subject”, and “content” and do not require Gmail-only args. Require email_id and email_address only when the raw name is send_email_tool.\n- When evaluating send_email_tool/write_email content, ensure the reply acknowledges key user constraints (deadlines, commitments, next steps) reflected in the email context and any executed tools.\n\nFinding & scoring rules:\n- Always explain tool issues via missing_tools / incorrect_tool_uses instead of only lowering scores.\n- missing_tools must list every expected tool from Policies that never ran (use normalized names).\n- incorrect_tool_uses must contain {{"tool": name, "why": explanation}} for wrong args, wrong order, premature Done, duration mismatches, wrong recipients, or other policy violations.\n- - If either list is non-empty, tool_usage must be ≤2 (use 1 for severely broken flows). Scores 3–5 are only allowed when both lists are empty.\n- Reduce content_alignment when the reply text fails to acknowledge commitments/tools outcomes the user needs.\n\nRubric:\n• Content Alignment: 5 exact/complete/tone-appropriate; 3 minor gaps; 1 misses/incorrect.\n• Tool Usage: 5 correct tools/args/order (+ terminal Done when applicable); 3 minor issues; 1 wrong/missing tools or invalid sequence.\n\nExamples:\nPass — all tools correct:\n{{"overall_correctness": 0.92, "verdict": "pass", "content_alignment": 5, "tool_usage": 5, "missing_tools": [], "incorrect_tool_uses": [], "evidence": ["Schedule shows 60 min meeting", "Reply confirms time and attendees"], "notes": "Handled scheduling request end-to-end."}}\nFail — missing schedule_meeting_tool:\n{{"overall_correctness": 0.40, "verdict": "fail", "content_alignment": 2, "tool_usage": 2, "missing_tools": ["schedule_meeting_tool"], "incorrect_tool_uses": [], "evidence": ["Email asked to schedule 60 min", "Only check_calendar_tool was called"], "notes": "Agent never scheduled the requested meeting."}}\n\nEvaluation source of truth:\n\n- Prefer final-state fields if present: output.assistant_reply, output.tool_trace, output.email_markdown.\n- Otherwise, derive from output.messages (e.g., last assistant content; AI tool_calls).\n- Validate arguments using raw tool calls from output.messages (AI messages with tool_calls). Use output.tool_trace for sequence evidence only; it is lossy/normalized.\n- Do not penalize the literal tool name when an equivalent normalized name is used.\n- Treat each field as populated unless it literally reads "(no … provided)" or "(empty)". Truncated strings ending with an ellipsis (… ) still contain evidence—use them. NEVER claim "information insufficient" when substantive content is present.\n\nCompute:\noverall_correctness = 0.6*(content_alignment/5) + 0.4*(tool_usage/5)\nverdict = "pass" if overall_correctness ≥ 0.70 else "fail".\n\nSTRICT OUTPUT CONTRACT: return valid JSON with these exact keys (empty arrays allowed): overall_correctness, verdict, content_alignment, tool_usage, missing_tools, incorrect_tool_uses, evidence, notes. Use double quotes and no trailing commas.\n\nValidity rules: standard JSON only (no markdown or free-text outside the JSON). Provide 2–4 evidence strings (≤120 chars each). Notes must always contain a concise (≤300 chars) insight: for passes, highlight strengths; for fails, state the key fix. If information is insufficient, still return the full object with zeros and empty arrays and put the reason in notes."""

HUMAN_PROMPT = """Evaluate this run per the rubric and return ONLY the JSON object.\n\n<email_markdown>\n{{email_markdown}}\n</email_markdown>\n\n<assistant_reply>\n{{assistant_reply}}\n</assistant_reply>\n\n<tool_trace>\n{{tool_trace}}\n</tool_trace>\n\n<tool_calls_summary>\n{{tool_calls_summary}}\n</tool_calls_summary>\n\n<tool_calls_json>\n{{tool_calls_json}}\n</tool_calls_json>\n\n<raw_output_optional>\n{{raw_output_optional}}\n</raw_output_optional>\n"""


def _default_model_name() -> str:
    return os.getenv("EMAIL_ASSISTANT_JUDGE_MODEL", "gemini-2.5-flash")


@lru_cache(maxsize=4)
def _build_base_chain(model_name: str, temperature: float) -> Runnable:
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=4096,
        convert_system_message_to_human=False,
    )
    parser = JsonOutputParser(pydantic_object=JudgeResult)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
    return prompt | llm | parser


def build_correctness_judge_chain(
    *, model_name: Optional[str] = None, temperature: float = 0.1
) -> Runnable:
    """Return a cached runnable that executes the correctness judge."""

    name = model_name or _default_model_name()
    return _build_base_chain(name, temperature)


def _coalesce_placeholder(value: Optional[str], placeholder: str) -> str:
    """Return the original string when it has content, otherwise the placeholder."""

    if value is None:
        return placeholder
    if isinstance(value, str):
        if value.strip():
            return value
        return placeholder
    return str(value)


def run_correctness_judge(
    *,
    email_markdown: str,
    assistant_reply: str,
    tool_trace: str,
    tool_calls_summary: str = "",
    tool_calls_json: str = "",
    raw_output_optional: str = "",
    parent_run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.1,
) -> JudgeResult:
    """Execute the LLM judge and return its structured verdict."""

    if not os.getenv("GOOGLE_API_KEY"):
        raise JudgeUnavailableError(
            "GOOGLE_API_KEY missing – cannot call Gemini-based judge."
        )

    chain = build_correctness_judge_chain(model_name=model_name, temperature=temperature)

    payload = {
        "email_markdown": _coalesce_placeholder(email_markdown, "(no email_markdown provided)"),
        "assistant_reply": _coalesce_placeholder(assistant_reply, "(assistant_reply empty)"),
        "tool_trace": _coalesce_placeholder(tool_trace, "(tool_trace empty)"),
        "tool_calls_summary": _coalesce_placeholder(tool_calls_summary, "(no tool calls)"),
        "tool_calls_json": tool_calls_json or "[]",
        "raw_output_optional": _coalesce_placeholder(raw_output_optional, "(raw output omitted)"),
    }

    trace_project = os.getenv("EMAIL_ASSISTANT_JUDGE_PROJECT", "email-assistant-judge")
    should_trace = bool(os.getenv("LANGSMITH_API_KEY") and trace_project)

    def _invoke(data):
        return chain.invoke(data)

    if should_trace:
        traced = traceable(
            run_type="chain",
            name="gemini_correctness_judge",
            project_name=trace_project,
        )(_invoke)
        invoke_fn = traced
    else:
        invoke_fn = _invoke

    try:
        judge_summary = "Judge review"
        reply_snippet = _truncate_text(strip_markdown_to_text(assistant_reply), limit=120)
        if reply_snippet:
            judge_summary = f"{judge_summary} | reply: {reply_snippet}"
        tools_snippet = (
            _truncate_text(strip_markdown_to_text(tool_calls_summary), limit=80)
            if tool_calls_summary
            else ""
        )
        if tools_snippet:
            judge_summary = f"{judge_summary} | tools: {tools_snippet}"

        def _summarize_output(raw_result: object) -> str | None:
            verdict = None
            score = None
            if isinstance(raw_result, JudgeResult):
                verdict = raw_result.verdict
                score = raw_result.overall_correctness
            elif isinstance(raw_result, dict):
                verdict = raw_result.get("verdict")
                score = raw_result.get("overall_correctness")
            if verdict is None and score is None:
                return "[judge]"
            details: list[str] = ["[judge]"]
            if verdict:
                details.append(f"verdict={verdict}")
            if score is not None:
                try:
                    details.append(f"score={float(score):.2f}")
                except Exception:
                    details.append(f"score={score}")
            return " ".join(details)

        try:
            raw = invoke_with_root_run(
                lambda: invoke_fn(payload),
                root_name="judge:gemini_correctness",
                input_summary=judge_summary,
                metadata={"payload_keys": list(payload.keys())},
                output_transform=_summarize_output,
                project_name=trace_project,
            )
        except OutputParserException as exc:
            llm_output = getattr(exc, "llm_output", None)
            snippet = (llm_output or "").strip().splitlines()[0][:160] if llm_output else ""
            message = (
                "Judge produced non-JSON output, skipping evaluation."
                if snippet
                else "Judge returned an empty response; treating evaluation as unavailable."
            )
            if snippet:
                message += f" First line: {snippet}"
            raise JudgeUnavailableError(message) from exc

        if isinstance(raw, JudgeResult):
            result = raw
        elif isinstance(raw, dict):
            normalised = _normalise_result_dict(raw)
            result = JudgeResult(**normalised)
        else:
            raise TypeError(f"Unexpected judge payload type: {type(raw)}")

        _record_feedback(result, parent_run_id=parent_run_id)
        return result
    except Exception as exc:  # pragma: no cover - defensive logging
        raise JudgeUnavailableError(f"Judge invocation failed: {exc}") from exc


def serialise_messages(messages: Iterable[object]) -> str:
    """Serialize LangChain/primitive messages to JSON for judge context."""

    serialisable = []
    for message in messages or []:
        if hasattr(message, "model_dump"):
            try:
                serialisable.append(message.model_dump())
                continue
            except Exception:  # pragma: no cover - fall back
                pass
        if hasattr(message, "dict"):
            try:
                serialisable.append(message.dict())  # type: ignore[attr-defined]
                continue
            except Exception:  # pragma: no cover - fall back
                pass
        serialisable.append(str(message))

    try:
        return json.dumps(serialisable, ensure_ascii=False)
    except TypeError:
        return json.dumps([str(item) for item in serialisable], ensure_ascii=False)


def _truncate_text(text: str, limit: int = 200) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def _compact_json(value: Any, limit: int = 200) -> str:
    if isinstance(value, str):
        return _truncate_text(value, limit)
    try:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        payload = str(value)
    return _truncate_text(payload, limit)


def _message_role(message: object) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "").lower()
    return str(
        getattr(message, "role", None)
        or getattr(message, "type", None)
        or ""
    ).lower()


def _tool_calls_from_message(message: object) -> Optional[List[Dict[str, Any]]]:
    if isinstance(message, dict):
        calls = message.get("tool_calls")
    else:
        calls = getattr(message, "tool_calls", None)
    if not calls:
        return None
    return list(calls)


def _tool_result_payload(message: object) -> Tuple[str, str]:
    role = _message_role(message)
    if role != "tool":
        return "", ""
    if isinstance(message, dict):
        tool_call_id = str(message.get("tool_call_id") or message.get("id") or "")
        raw_content = message.get("content", "")
        if isinstance(raw_content, (list, dict)):
            content = _compact_json(raw_content)
        else:
            content = _truncate_text(raw_content)
    else:
        tool_call_id = str(
            getattr(message, "tool_call_id", None)
            or getattr(message, "id", None)
            or ""
        )
        raw_content = extract_message_content(message)
        content = _truncate_text(raw_content)
    return tool_call_id, content


def _build_tool_call_context(messages: Iterable[object]) -> Tuple[str, str]:
    messages = list(messages or [])
    if not messages:
        return "(no tool calls)", "[]"

    tool_results: Dict[str, str] = {}
    for message in messages:
        tool_call_id, content = _tool_result_payload(message)
        if tool_call_id and content:
            tool_results[tool_call_id] = content

    entries: List[Dict[str, Any]] = []
    summary_lines: List[str] = []
    index = 1
    for message in messages:
        tool_calls = _tool_calls_from_message(message)
        if not tool_calls:
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                call = {"name": str(call), "args": str(call)}
            name = str(call.get("name", ""))
            args = call.get("args", {})
            args_text = _compact_json(args)
            call_id = str(call.get("id") or call.get("tool_call_id") or "")
            result_text = tool_results.get(call_id, "")
            entry: Dict[str, Any] = {"step": index, "name": name, "args": args_text}
            if result_text:
                entry["result"] = result_text
                summary_lines.append(
                    f"{index}. {name} args={args_text} → result={result_text}"
                )
            else:
                summary_lines.append(f"{index}. {name} args={args_text}")
            entries.append(entry)
            index += 1

    if not entries:
        return "(no tool calls)", "[]"

    summary = "\n".join(summary_lines)
    tool_calls_json = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return summary, tool_calls_json


def build_tool_call_context(messages: Iterable[object]) -> Tuple[str, str]:
    """Public wrapper that returns (summary_text, json_payload) for tool calls."""

    return _build_tool_call_context(messages)


def _normalise_result_dict(data: dict) -> dict:
    result = data.copy()
    incorrect = result.get("incorrect_tool_uses") or []
    normalised_incorrect = []
    for item in incorrect:
        if isinstance(item, dict):
            normalised_incorrect.append(item)
        else:
            normalised_incorrect.append({"tool": "", "why": str(item)})
    result["incorrect_tool_uses"] = normalised_incorrect
    missing_tools = list(result.get("missing_tools") or [])
    result["missing_tools"] = missing_tools
    evidence = [str(e) for e in (result.get("evidence") or [])]
    trimmed = [e[:120] for e in evidence][:4]
    if len(trimmed) < 2 and evidence:
        trimmed = trimmed[: max(1, len(trimmed))]
    result["evidence"] = trimmed
    content_alignment_val = result.get("content_alignment")
    tool_usage_val = result.get("tool_usage")
    if content_alignment_val is not None:
        content_alignment_val = int(content_alignment_val)
        result["content_alignment"] = content_alignment_val
    if tool_usage_val is not None:
        tool_usage_val = int(tool_usage_val)
    if tool_usage_val is None and (missing_tools or normalised_incorrect):
        tool_usage_val = 1
    if (missing_tools or normalised_incorrect) and tool_usage_val is not None and tool_usage_val > 2:
        tool_usage_val = 2
    if tool_usage_val is not None:
        result["tool_usage"] = tool_usage_val
    if content_alignment_val is not None and tool_usage_val is not None:
        recalculated = 0.6 * (content_alignment_val / 5) + 0.4 * (tool_usage_val / 5)
        result["overall_correctness"] = float(recalculated)
    elif "overall_correctness" in result:
        result["overall_correctness"] = float(result["overall_correctness"])
    if "content_alignment" in result:
        result["content_alignment"] = int(result["content_alignment"])
    if "verdict" in result:
        result["verdict"] = str(result["verdict"]).lower()
    if tool_usage_val is not None and tool_usage_val < 3:
        result["verdict"] = "fail"
    notes = str(result.get("notes") or "").strip()
    verdict = str(result.get("verdict", "")).lower()
    if not notes:
        if verdict == "pass":
            notes = "Pass: reply and tool usage met the request; keep matching user constraints."
        elif verdict == "fail":
            notes = "Fail: review evidence and address the highlighted tool/content issues."
        else:
            notes = "No additional notes provided."
    result["notes"] = notes[:300]
    return result


def _record_feedback(result: JudgeResult, parent_run_id: Optional[str] = None) -> None:
    """Attach judge feedback to the current LangSmith run if possible."""

    try:
        run_id = parent_run_id
        if not run_id:
            run_tree = get_current_run_tree()
            if not run_tree:
                return
            run_id = run_tree.id
        run_id = str(run_id)
        client = Client()

        # Emit feedback in reverse priority so UI (newest-first) shows verdict → notes → evidence → metrics → diagnostics.

        # Diagnostics first (lowest priority)
        if result.missing_tools:
            try:
                client.create_feedback(
                    run_id=run_id,
                    key="missing_tools",
                    value=", ".join(result.missing_tools),
                    comment="Tools the agent failed to invoke",
                )
            except Exception:
                pass

        if result.incorrect_tool_uses:
            for issue in result.incorrect_tool_uses:
                try:
                    client.create_feedback(
                        run_id=run_id,
                        key="tool",
                        value=issue.tool or "(unknown)",
                        comment=issue.why or "Incorrect tool usage detected",
                    )
                    client.create_feedback(
                        run_id=run_id,
                        key="why",
                        value=issue.why or "See comment",
                        comment=(
                            f"Tool '{issue.tool}' flagged by judge"
                            if issue.tool
                            else "Incorrect tool usage details"
                        ),
                    )
                except Exception:
                    continue

        # Metrics next
        metric_entries = [
            (
                "tool_usage",
                result.tool_usage,
                "Tool usage score (0-5)",
            ),
            (
                "content_alignment",
                result.content_alignment,
                "Content alignment score (0-5)",
            ),
            (
                "overall_correctness",
                result.overall_correctness,
                "Weighted correctness (0..1)",
            ),
        ]
        for key, value, comment in metric_entries:
            try:
                client.create_feedback(
                    run_id=run_id,
                    key=key,
                    score=float(value),
                    value=str(value),
                    comment=comment,
                )
            except Exception:
                continue

        # Evidence bundle (if any)
        if result.evidence:
            try:
                client.create_feedback(
                    run_id=run_id,
                    key="evidence",
                    value=" | ".join(result.evidence),
                    comment="Judge evidence snippets",
                )
            except Exception:
                pass

        # Notes
        try:
            client.create_feedback(
                run_id=run_id,
                key="notes",
                value=result.notes,
                comment="Judge guidance",
            )
        except Exception:
            pass

        # Verdict last (highest priority for display)
        try:
            client.create_feedback(
                run_id=run_id,
                key="verdict",
                value=result.verdict,
                comment=f"Pass when overall_correctness ≥ 0.70 (scored {result.overall_correctness:.2f})",
                extra=result.model_dump(),
            )
        except Exception:
            pass

    except Exception:
        # Feedback attachment is best-effort; ignore failures so tests keep running.
        return


class GeminiJudgeStringEvaluator(StringEvaluator):
    """LangSmith-compatible string evaluator that uses the Gemini judge."""

    evaluation_name: str = "gemini_correctness_judge"
    input_key: str = "email_markdown"
    prediction_key: str = "assistant_reply"
    answer_key: Optional[str] = None
    grading_function = staticmethod(lambda *_: {})  # Unused; required by base class
    model_name: Optional[str] = None
    temperature: float = 0.1

    def evaluate_run(
        self,
        run: Run,
        example: Optional[Example] = None,
        evaluator_run_id: Optional[str] = None,
    ) -> EvaluationResult:
        if run.outputs is None:
            raise ValueError("Run outputs cannot be None for LLM judge.")

        outputs = run.outputs
        email_markdown = (
            outputs.get("email_markdown")
            or run.inputs.get("email_markdown")
            or ""
        )
        assistant_reply = (
            outputs.get("assistant_reply")
            or outputs.get("output")
            or ""
        )
        tool_trace = outputs.get("tool_trace") or ""
        messages = outputs.get("messages") or []
        if not tool_trace and messages:
            try:
                tool_trace = format_messages_string(messages)
            except Exception:  # pragma: no cover
                tool_trace = ""

        tool_calls_summary, tool_calls_json = _build_tool_call_context(messages)
        raw_messages = serialise_messages(messages)
        raw_output = _compact_json(
            {
                "message_count": len(messages),
                "messages_preview": raw_messages[:400],
            },
            limit=400,
        )

        parent_run_id = getattr(run, "id", None)

        result = run_correctness_judge(
            email_markdown=email_markdown,
            assistant_reply=assistant_reply,
            tool_trace=tool_trace,
            tool_calls_summary=tool_calls_summary,
            tool_calls_json=tool_calls_json,
            raw_output_optional=raw_output,
            parent_run_id=parent_run_id,
            model_name=self.model_name,
            temperature=self.temperature,
        )

        return EvaluationResult(
            key=self.evaluation_name,
            score=result.overall_correctness,
            value=result.verdict,
            comment=result.notes,
            evaluator_info={"judge": result.model_dump()},
        )


def create_langsmith_correctness_evaluator(
    *, model_name: Optional[str] = None, temperature: float = 0.1
) -> LangChainStringEvaluator:
    """Return a LangChainStringEvaluator wrapping the Gemini judge."""

    evaluator = GeminiJudgeStringEvaluator(
        model_name=model_name, temperature=temperature
    )
    return LangChainStringEvaluator(evaluator=evaluator)
