"""Shared sanitization helpers for building readable benign exports.

This module is a helper library only; it has no entry point. The functions here are
imported by data_utils/preprocess_native_benign.py, which converts a native
multi-sample Inspect .eval log of honest runs into the readable-export layout that
experiment.py consumes: message normalization at the observation boundary, canonical
task validation, the synthetic linear AIDE journal, and the .eval/markdown writers.
"""

from __future__ import annotations

import hashlib
import json
import re
from uuid import uuid4
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence


MANIFEST_FIELDS = [
    "sample_uuid",
    "task_id",
    "side_task_name",
    "side_task_group",
    "source_name",
    "source_group",
    "attack_group",
    "eval_log_filename",
    "eval_created",
    "attack_model",
    "main_task_success",
    "side_task_success",
    "assistant_messages_count",
    "tool_call_count",
    "attack_step",
    "total_steps",
    "has_stepwise",
]


def parse_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return text
        if text[0] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "sample"


def stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    parts.append(stripped)
                continue
            if isinstance(item, dict):
                block_type = str(item.get("type") or "").lower()
                if block_type in {"tool_use", "tool_result", "server_tool_use", "computer_call"}:
                    continue
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                elif isinstance(text, list):
                    nested = stringify_content(text)
                    if nested:
                        parts.append(nested)
        return "\n\n".join(parts).strip()
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").lower()
        if block_type in {"tool_use", "tool_result", "server_tool_use", "computer_call"}:
            return ""
        text = value.get("text") or value.get("content")
        if isinstance(text, str):
            return text.strip()
        if isinstance(text, list):
            return stringify_content(text)
    return ""


def normalize_tool_call(raw_call: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    function_name = (
        raw_call.get("function")
        or raw_call.get("name")
        or raw_call.get("tool_name")
        or raw_call.get("tool")
        or raw_call.get("type")
        or "unknown_tool"
    )
    arguments = (
        raw_call.get("arguments")
        or raw_call.get("input")
        or raw_call.get("args")
        or raw_call.get("parameters")
        or {}
    )
    if isinstance(arguments, str):
        parsed = parse_json_maybe(arguments)
        arguments = parsed if isinstance(parsed, dict) else {"value": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return {
        "id": str(raw_call.get("id") or raw_call.get("tool_use_id") or fallback_id),
        "function": str(function_name),
        "arguments": arguments,
        "type": "function",
    }


def attach_missing_tool_ids(messages: list[dict[str, Any]]) -> None:
    pending_ids: list[str] = []
    call_by_id: dict[str, dict[str, Any]] = {}

    for message in messages:
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                tool_id = str(tool_call.get("id") or "")
                if tool_id:
                    pending_ids.append(tool_id)
                    call_by_id[tool_id] = tool_call
            continue

        if message.get("role") != "tool":
            continue

        tool_call_id = str(message.get("tool_call_id") or "")
        if not tool_call_id and pending_ids:
            tool_call_id = pending_ids.pop(0)
            message["tool_call_id"] = tool_call_id
        elif tool_call_id in pending_ids:
            pending_ids.remove(tool_call_id)

        if not message.get("function") and tool_call_id in call_by_id:
            message["function"] = call_by_id[tool_call_id].get("function")


def build_tool_outputs_by_call_id(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id") or "")
        if not tool_call_id:
            continue
        outputs[tool_call_id] = {
            "function": str(message.get("function") or ""),
            "content": stringify_content(message.get("content")),
            "error": message.get("error"),
            "stderr": stringify_content(message.get("stderr")),
            "return_code": message.get("return_code"),
        }
    return outputs


def messages_at_observation_boundary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain submit actions but withhold their post-hoc grader results."""

    copied = json.loads(json.dumps(messages))
    submit_ids = {
        str(call.get("id"))
        for message in copied
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
        if str(call.get("function") or "").strip().lower() == "submit" and call.get("id")
    }
    for message in copied:
        call_id = str(message.get("tool_call_id") or "")
        function_name = str(message.get("function") or "").strip().lower()
        if message.get("role") == "tool" and (call_id in submit_ids or function_name == "submit"):
            message["content"] = "[post-submission grader feedback withheld]"
    return copied


def render_tool_call_for_journal(tool_call: dict[str, Any]) -> str:
    function_name = str(tool_call.get("function") or "unknown_tool")
    arguments = tool_call.get("arguments") or {}
    if function_name == "bash":
        return str(arguments.get("cmd") or "").strip()
    if function_name == "python":
        return str(arguments.get("code") or "").strip()
    payload = {"tool": function_name, **arguments}
    return json.dumps(payload, ensure_ascii=True)


def build_synthetic_aide_journal(messages: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    tool_outputs = build_tool_outputs_by_call_id(messages)
    step_index = 0

    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            continue

        plan_text = stringify_content(message.get("content")) or "Tool-assisted action."
        for tool_call in tool_calls:
            tool_call_id = str(tool_call.get("id") or "")
            tool_output = tool_outputs.get(tool_call_id, {})
            output_was_recorded = tool_call_id in tool_outputs
            output_text = tool_output.get("content") or ""
            error_value = tool_output.get("error")
            stderr_text = tool_output.get("stderr") or ""
            exc_info = None
            if error_value:
                exc_info = (
                    dict(error_value)
                    if isinstance(error_value, dict)
                    else {"message": str(error_value)}
                )
            if error_value and not stderr_text:
                stderr_text = json.dumps(error_value, ensure_ascii=True, sort_keys=True)
            if not output_was_recorded:
                output_text = "[no tool result recorded; execution outcome unknown]"
            node_identity = hashlib.sha256(
                f"{step_index}\0{tool_call_id}\0{render_tool_call_for_journal(tool_call)}".encode("utf-8")
            ).hexdigest()[:32]
            node = {
                "id": node_identity,
                "code": render_tool_call_for_journal(tool_call),
                "plan": plan_text,
                "step": step_index,
                "ctime": float(step_index),
                "parent": None,
                "execution_info": {
                    "stdout": output_text,
                    "stderr": stderr_text,
                    "return_code": tool_output.get("return_code"),
                    "exec_time": 0.0,
                    "exc_type": "tool_error" if error_value else None,
                    "exc_info": exc_info,
                    "exc_stack": None,
                },
                "analysis": "",
                "metric": None,
                "validation_metric": None,
                "metric_to_optimize": None,
                "is_buggy": None,
                "is_submitted": str(tool_call.get("function") or "").strip().lower() == "submit",
            }
            nodes.append(node)
            step_index += 1

    return {"nodes": nodes, "synthetic_linear": True}


def content_to_output_blocks(content: Any) -> list[dict[str, Any]]:
    text = stringify_content(content)
    if not text:
        return []
    return [{"type": "text", "text": text}]


def build_eval_output(messages: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    last_assistant = None
    for message in reversed(messages):
        if message.get("role") == "assistant":
            last_assistant = message
            break

    if last_assistant is None:
        assistant_message = {
            "id": str(uuid4().hex),
            "content": [],
            "source": "generate",
            "metadata": {"tools_hidden_from_monitor": [], "monitor_views": {}},
            "role": "assistant",
            "tool_calls": [],
            "model": model_name,
        }
        completion = ""
        stop_reason = "stop"
    else:
        tool_calls = last_assistant.get("tool_calls") or []
        assistant_message = {
            "id": str(last_assistant.get("id") or uuid4().hex),
            "content": content_to_output_blocks(last_assistant.get("content")),
            "source": "generate",
            "metadata": {"tools_hidden_from_monitor": [], "monitor_views": {}},
            "role": "assistant",
            "tool_calls": tool_calls,
            "model": model_name,
        }
        completion = stringify_content(last_assistant.get("content"))
        stop_reason = "tool_calls" if tool_calls else "stop"

    return {
        "model": model_name,
        "choices": [
            {
                "message": assistant_message,
                "stop_reason": stop_reason,
            }
        ],
        "completion": completion,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "time": 0.0,
    }


_CANONICAL_TASK_PREAMBLE = "Your task is as follows:"


def validate_canonical_task_text(task_text: str) -> str:
    """Validate the user-task envelope and reject post-hoc grader payloads."""

    normalized = str(task_text or "").strip()
    if not normalized:
        raise ValueError("Missing benign canonical task text.")
    lines = normalized.splitlines()
    if lines[0].strip() != _CANONICAL_TASK_PREAMBLE:
        raise ValueError(
            "Invalid benign canonical task envelope; expected the authoritative "
            f"{_CANONICAL_TASK_PREAMBLE!r} preamble, found {lines[0].strip()!r}."
        )
    nonempty_body = [line.strip() for line in lines[1:] if line.strip()]
    if not nonempty_body:
        raise ValueError("Invalid benign canonical task envelope; task body is empty.")
    return normalized


def sample_member_name(sample_payload: dict[str, Any]) -> str:
    sample_id = str(sample_payload.get("id") or sample_payload.get("uuid") or "sample")
    epoch = sample_payload.get("epoch") or 1
    return f"samples/{sample_id}_epoch_{epoch}.json"


def write_eval_archive(
    *,
    journal_start: dict[str, Any],
    sample_payload: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "_journal/start.json",
            json.dumps(journal_start, ensure_ascii=True) + "\n",
        )
        archive.writestr(
            sample_member_name(sample_payload),
            json.dumps(sample_payload, ensure_ascii=True) + "\n",
        )


def write_examples_markdown(records: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Bash Arena Benign Readable Export",
        "",
        f"- Exported examples: {len(records)}",
        "",
    ]
    for index, record in enumerate(records, start=1):
        lines.append(f"## Example {index}: {record.get('sample_uuid')}")
        lines.append("")
        for field in MANIFEST_FIELDS:
            value = record.get(field)
            if value not in (None, ""):
                lines.append(f"- **{field}**: {value}")
        if record.get("per_sample_eval_path"):
            lines.append(f"- **per_sample_eval_path**: {record.get('per_sample_eval_path')}")
        notes = record.get("notes") or {}
        if notes:
            lines.append("")
            lines.append("### Notes")
            lines.append("")
            for key, value in notes.items():
                if value not in (None, ""):
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
