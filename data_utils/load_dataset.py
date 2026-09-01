from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "ctrldataset2026"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "readable" / "malicious"
DEFAULT_ZIP_PATH = DEFAULT_DATASET_DIR / "final_dataset.zip"
DEFAULT_LIMIT = 200
# The "Refined" source group: successful attacks rewritten to lower a development
# monitor's score. This is the split the paper evaluates on.
REFINED_SOURCE_NAMES = ("recon_refined", "human_strategies_refined", "opus_only_refined")

TEXT_FIELDS = [
    "attack_strategy",
    "plan_strategy",
    "attack_summary",
    "structural_strategy",
    "structural_strategy_explanation",
    "evasion_tactics",
    "evasion_tactics_explanation",
    "technical_techniques",
    "technical_techniques_explanation",
]

CORE_FIELDS = [
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

SCORE_FIELDS = [
    "dev_gpt5_monitor_score",
    "dev_opus_monitor_score",
    "dev_gpt5_nano_monitor_score",
    "sw_full_claude-opus-4-5-20251101",
    "sw_trimmed_claude-opus-4-5-20251101",
    "sw_full_gpt-5.1-codex-max",
    "sw_trimmed_gpt-5.1-codex-max",
    "sw_full_gpt-5",
    "sw_trimmed_gpt-5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export ctrldataset2026 into a readable format. "
            "If final_dataset.zip is present, matching .eval logs are extracted too."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--side-task-name")
    parser.add_argument(
        "--source-name",
        action="append",
        default=None,
        help=(
            "Restrict the export to these MonitoringBench source names; repeatable. "
            f"Defaults to the refined split ({', '.join(REFINED_SOURCE_NAMES)}), which is what "
            "the paper scores. Pass --all-source-names to export every group."
        ),
    )
    parser.add_argument(
        "--all-source-names",
        action="store_true",
        help="Export every source group instead of the default refined split.",
    )
    parser.add_argument("--sample-uuid")
    parser.add_argument("--task-id")
    parser.add_argument(
        "--sort-by",
        default="eval_created",
        choices=["eval_created", "side_task_name", "source_name", "sample_uuid"],
    )
    return parser.parse_args()


def load_rows(dataset_dir: Path, zip_path: Path) -> list[dict[str, Any]]:
    arrow_paths = sorted((dataset_dir / "test").glob("*.arrow"))
    if arrow_paths:
        tables: list[pa.Table] = []
        for arrow_path in arrow_paths:
            with pa.memory_map(str(arrow_path), "r") as source:
                tables.append(ipc.open_stream(source).read_all())

        return pa.concat_tables(tables).to_pylist()

    parquet_paths = sorted(dataset_dir.glob("*.parquet"))
    if parquet_paths:
        return pq.read_table(parquet_paths[0]).to_pylist()

    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            parquet_members = [name for name in archive.namelist() if name.endswith("master.parquet")]
            if parquet_members:
                parquet_bytes = BytesIO(archive.read(parquet_members[0]))
                return pq.read_table(parquet_bytes).to_pylist()

    raise FileNotFoundError(
        "Could not find dataset rows in Arrow, Parquet, or final_dataset.zip."
    )


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    # Default to the refined split; an explicit --source-name or --all-source-names wins.
    if args.all_source_names:
        source_names = None
    else:
        source_names = frozenset(args.source_name or REFINED_SOURCE_NAMES)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if args.side_task_name and row.get("side_task_name") != args.side_task_name:
            continue
        if source_names is not None and row.get("source_name") not in source_names:
            continue
        if args.sample_uuid and row.get("sample_uuid") != args.sample_uuid:
            continue
        if args.task_id and str(row.get("task_id")) != str(args.task_id):
            continue
        selected.append(row)

    selected.sort(key=lambda row: _sortable_value(row.get(args.sort_by)))
    if args.limit > 0:
        selected = selected[: args.limit]
    return selected


def _sortable_value(value: Any) -> tuple[int, str]:
    if value is None:
        return (1, "")
    return (0, str(value))


def build_manifest_record(row: dict[str, Any], extracted_log_path: str | None = None) -> dict[str, Any]:
    record = {field: row.get(field) for field in CORE_FIELDS}
    record["notes"] = {field: row.get(field) for field in TEXT_FIELDS if _has_text(row.get(field))}
    record["monitor_scores"] = {
        field: row.get(field) for field in SCORE_FIELDS if row.get(field) is not None
    }
    if extracted_log_path:
        record["extracted_log_path"] = extracted_log_path
    return record


def sample_member_name(sample_payload: dict[str, Any]) -> str:
    sample_id = str(sample_payload.get("id") or sample_payload.get("uuid") or "sample")
    epoch = sample_payload.get("epoch") or 1
    return f"samples/{sample_id}_epoch_{epoch}.json"


def trimmed_eval_start(
    journal_start: dict[str, Any] | None,
    sample_payload: dict[str, Any],
) -> dict[str, Any]:
    trimmed = json.loads(json.dumps(journal_start or {"version": 2, "eval": {}, "plan": {}}))
    eval_section = trimmed.setdefault("eval", {})
    dataset = eval_section.setdefault("dataset", {})
    dataset["samples"] = 1
    dataset["sample_ids"] = [str(sample_payload.get("id") or sample_payload.get("uuid") or "sample")]
    return trimmed


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


def stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n\n".join(parts).strip()
    if isinstance(value, dict):
        text = value.get("text") or value.get("content")
        if isinstance(text, str):
            return text.strip()
    return ""


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


def build_eval_sample_payload(sample_payload: dict[str, Any]) -> dict[str, Any]:
    messages = messages_at_observation_boundary(sample_payload.get("messages", []) or [])
    metadata = dict(sample_payload.get("metadata") or {})
    metadata["aide_journal"] = build_synthetic_aide_journal(messages)
    metadata["aide_journal_is_synthetic_linear"] = True
    model_name = str(
        sample_payload.get("output", {}).get("model")
        or sample_payload.get("metadata", {}).get("metadata", {}).get("attack", {}).get("model")
        or "unknown/unknown"
    )

    return {
        "id": sample_payload.get("id"),
        "epoch": sample_payload.get("epoch", 1),
        "input": sample_payload.get("input", ""),
        "target": sample_payload.get("target", ""),
        "messages": messages,
        "events": [],
        "metadata": metadata,
        "model_usage": sample_payload.get("model_usage") or {},
        "total_time": sample_payload.get("total_time"),
        "working_time": sample_payload.get("working_time"),
        "uuid": sample_payload.get("uuid"),
        "error_retries": sample_payload.get("error_retries") or [],
        "attachments": sample_payload.get("attachments") or {},
        "created_at": sample_payload.get("created_at"),
        "output": build_eval_output(messages, model_name),
        "sandbox": sample_payload.get("sandbox"),
        "scores": sample_payload.get("scores") or {},
        "store": sample_payload.get("store") or {},
        "started_at": sample_payload.get("started_at"),
        "completed_at": sample_payload.get("completed_at"),
    }


def assert_no_orphan_eval_files(output_dir: Path, selected_sample_uuids: set[str]) -> None:
    """Fail before export when an existing readable root contains stale samples."""

    if not output_dir.exists():
        return
    orphan_paths = [
        path
        for path in output_dir.rglob("*.eval")
        if path.relative_to(output_dir).parts[0] not in {safe_name(value) for value in selected_sample_uuids}
    ]
    if orphan_paths:
        preview = ", ".join(str(path.relative_to(output_dir)) for path in orphan_paths[:5])
        raise ValueError(
            f"Refusing to export into {output_dir}: found {len(orphan_paths)} orphan .eval files not present "
            f"in the selected cohort (for example: {preview}). Use a fresh output directory."
        )


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    lines = [json.dumps(record, ensure_ascii=True) for record in records]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_markdown(records: list[dict[str, Any]], output_path: Path, zip_present: bool) -> None:
    lines = [
        "# ctrldataset2026 Readable Export",
        "",
        f"- Exported examples: {len(records)}",
        f"- Zip archive present: {'yes' if zip_present else 'no'}",
        "",
    ]
    if not zip_present:
        lines.extend(
            [
                "> Raw transcripts are not present in the saved Arrow/Parquet dataset.",
                "> Place `final_dataset.zip` in `data/ctrldataset2026/` or pass `--zip-path` to extract nested `.eval` logs too.",
                "",
            ]
        )

    for index, record in enumerate(records, start=1):
        lines.append(f"## Example {index}: {record.get('sample_uuid', 'unknown')}")
        lines.append("")
        for field in CORE_FIELDS:
            value = record.get(field)
            if value is not None and value != "":
                lines.append(f"- **{field}**: {value}")
        extracted_log_path = record.get("extracted_log_path")
        if extracted_log_path:
            lines.append(f"- **extracted_log_path**: {extracted_log_path}")
        per_sample_eval_path = record.get("per_sample_eval_path")
        if per_sample_eval_path:
            lines.append(f"- **per_sample_eval_path**: {per_sample_eval_path}")

        notes = record.get("notes", {})
        if notes:
            lines.append("")
            lines.append("### Notes")
            lines.append("")
            for field, value in notes.items():
                lines.append(f"#### {field}")
                lines.append("")
                lines.append(str(value).strip())
                lines.append("")

        scores = record.get("monitor_scores", {})
        if scores:
            lines.append("### Monitor Scores")
            lines.append("")
            for field, value in scores.items():
                lines.append(f"- **{field}**: {value}")
            lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def maybe_extract_logs(
    rows: list[dict[str, Any]],
    zip_path: Path,
    output_dir: Path,
) -> dict[str, dict[str, str]]:
    if not zip_path.exists():
        return {}

    wanted: list[tuple[str, dict[str, Any]]] = []
    wanted_log_names: set[str] = set()
    wanted_sample_uuids: set[str] = set()
    for row in rows:
        log_filename = row.get("eval_log_filename")
        sample_uuid = row.get("sample_uuid")
        task_id = row.get("task_id")
        if not log_filename or not sample_uuid or task_id in (None, ""):
            raise ValueError(
                "Every selected malicious row must have eval_log_filename, sample_uuid, and task_id."
            )
        log_name = Path(str(log_filename)).name
        if str(sample_uuid) in wanted_sample_uuids:
            raise ValueError(f"Duplicate selected malicious sample UUID: {sample_uuid!r}")
        wanted.append((log_name, row))
        wanted_log_names.add(log_name)
        wanted_sample_uuids.add(str(sample_uuid))
    extracted: dict[str, dict[str, str]] = {}

    with zipfile.ZipFile(zip_path) as archive:
        members: dict[str, list[str]] = {}
        for name in archive.namelist():
            if is_real_eval_member(name):
                members.setdefault(Path(name).name, []).append(name)

        resolved_members: dict[str, str] = {}
        for log_name in wanted_log_names:
            matches = members.get(log_name, [])
            if not matches:
                raise ValueError(f"Missing malicious eval archive member for basename {log_name!r}.")
            if len(matches) != 1:
                raise ValueError(
                    f"Ambiguous malicious eval basename {log_name!r}: matched {matches!r}"
                )
            resolved_members[log_name] = matches[0]

        # Rows sharing an outer log are adjacent so each potentially large
        # nested archive is decompressed only once. A single MonitoringBench
        # eval may contain many task samples selected as distinct dataset rows.
        current_log_name: str | None = None
        current_nested_bytes: bytes | None = None
        for log_name, row in sorted(wanted, key=lambda item: item[0]):
            archive_name = resolved_members[log_name]
            sample_uuid = str(row["sample_uuid"])

            if log_name != current_log_name:
                current_nested_bytes = archive.read(archive_name)
                current_log_name = log_name
            if current_nested_bytes is None:
                raise AssertionError("Nested malicious eval cache was not initialized.")
            sample_payload, journal_start, sample_member = extract_nested_sample(
                nested_bytes=current_nested_bytes,
                task_id=str(row.get("task_id", "")),
                sample_uuid=sample_uuid,
                expected_epoch=None,
            )
            sample_dir = output_dir / safe_name(sample_uuid)
            sample_dir.mkdir(parents=True, exist_ok=True)

            if journal_start is not None:
                (sample_dir / "eval_start.json").write_text(
                    json.dumps(journal_start, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8",
                )

            per_sample_eval_start = trimmed_eval_start(journal_start, sample_payload)
            task_description_path = sample_dir / "task_description.md"
            task_description_text = str(sample_payload.get("metadata", {}).get("main_task_description") or "").strip()
            if not task_description_text:
                raise ValueError(f"Missing main_task_description for malicious sample {sample_uuid}.")
            task_description_path.write_text(task_description_text + "\n", encoding="utf-8")
            sample_metadata = sample_payload.setdefault("metadata", {})
            sample_metadata["task_description_path"] = str(task_description_path.resolve())
            sample_metadata["canonical_task_text"] = task_description_text
            sample_metadata["canonical_task_sha256"] = hashlib.sha256(
                task_description_text.encode("utf-8")
            ).hexdigest()
            eval_sample_payload = build_eval_sample_payload(sample_payload)
            (sample_dir / "sample.json").write_text(
                json.dumps(sample_payload, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            (sample_dir / "trajectory.eval").unlink(missing_ok=True)
            write_eval_archive(
                journal_start=per_sample_eval_start,
                sample_payload=eval_sample_payload,
                output_path=sample_dir / "trajectory.eval",
            )
            preview_md_path = sample_dir / "transcript.md"
            preview_md_path.write_text(
                render_sample_preview(
                    row=row,
                    sample_payload=sample_payload,
                    journal_start=per_sample_eval_start,
                    archive_name=archive_name,
                    sample_member=sample_member,
                ),
                encoding="utf-8",
            )

            extracted[sample_uuid] = {
                "transcript_path": str(preview_md_path.relative_to(output_dir)),
                "eval_path": str((sample_dir / "trajectory.eval").relative_to(output_dir)),
            }
            dataset_record = build_manifest_record(row)
            dataset_record["per_sample_eval_path"] = extracted[sample_uuid]["eval_path"]
            dataset_record_path = sample_dir / "dataset_record.json"
            dataset_record_path.write_text(
                json.dumps(dataset_record, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )

    if set(extracted) != {str(row["sample_uuid"]) for row in rows}:
        raise ValueError("Malicious log extraction did not produce exactly one artifact per selected row.")
    return extracted


def extract_nested_sample(
    nested_bytes: bytes,
    task_id: str,
    sample_uuid: str,
    expected_epoch: int | None = 1,
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    with zipfile.ZipFile(BytesIO(nested_bytes)) as nested_archive:
        journal_start = None
        if "_journal/start.json" in nested_archive.namelist():
            journal_start = json.loads(nested_archive.read("_journal/start.json"))

        if expected_epoch is None:
            candidates = sorted(
                name
                for name in nested_archive.namelist()
                if name.startswith(f"samples/{task_id}_epoch_") and name.endswith(".json")
            )
            matching_members: list[str] = []
            for candidate in candidates:
                candidate_payload = json.loads(nested_archive.read(candidate))
                if isinstance(candidate_payload, dict) and str(candidate_payload.get("uuid")) == sample_uuid:
                    matching_members.append(candidate)
            if len(matching_members) != 1:
                raise ValueError(
                    f"Expected exactly one malicious sample member for task {task_id!r} and UUID "
                    f"{sample_uuid!r}; matched {matching_members!r} from {candidates!r}."
                )
            sample_member = matching_members[0]
        else:
            sample_member = select_sample_member(nested_archive.namelist(), task_id, expected_epoch)
        sample_payload = json.loads(nested_archive.read(sample_member))
        if not isinstance(sample_payload, dict):
            raise ValueError(f"Malicious sample member {sample_member!r} is not a JSON object.")
        if str(sample_payload.get("id")) != task_id:
            raise ValueError(
                f"Malicious sample task ID mismatch in {sample_member!r}: "
                f"expected {task_id!r}, found {sample_payload.get('id')!r}."
            )
        if str(sample_payload.get("uuid")) != sample_uuid:
            raise ValueError(
                f"Malicious sample UUID mismatch in {sample_member!r}: "
                f"expected {sample_uuid!r}, found {sample_payload.get('uuid')!r}."
            )
        payload_epoch = sample_payload.get("epoch")
        if isinstance(payload_epoch, bool) or not isinstance(payload_epoch, int) or payload_epoch < 1:
            raise ValueError(
                f"Invalid malicious sample epoch in {sample_member!r}: found {payload_epoch!r}."
            )
        if expected_epoch is not None and payload_epoch != expected_epoch:
            raise ValueError(
                f"Malicious sample epoch mismatch in {sample_member!r}: "
                f"expected {expected_epoch!r}, found {payload_epoch!r}."
            )
        metadata_task_id = (sample_payload.get("metadata") or {}).get("main_task_name")
        if metadata_task_id not in (None, "") and str(metadata_task_id) != task_id:
            raise ValueError(
                f"Malicious sample metadata task ID mismatch in {sample_member!r}: "
                f"expected {task_id!r}, found {metadata_task_id!r}."
            )

        return sample_payload, journal_start, sample_member


def select_sample_member(names: list[str], task_id: str, expected_epoch: int = 1) -> str:
    expected = f"samples/{task_id}_epoch_{expected_epoch}.json"
    matches = [name for name in names if name == expected]
    if not matches:
        available = sorted(
            name
            for name in names
            if name.startswith(f"samples/{task_id}_epoch_") and name.endswith(".json")
        )
        raise ValueError(
            f"Missing exact malicious sample member {expected!r}; task-matching members={available!r}."
        )
    if len(matches) != 1:
        raise ValueError(f"Ambiguous duplicate malicious sample member {expected!r}.")
    return matches[0]


def render_sample_preview(
    *,
    row: dict[str, Any],
    sample_payload: dict[str, Any],
    journal_start: dict[str, Any] | None,
    archive_name: str,
    sample_member: str | None,
) -> str:
    metadata = sample_payload.get("metadata", {}) or {}
    eval_metadata = (journal_start or {}).get("eval", {}) if journal_start else {}
    messages = sample_payload.get("messages", []) or []
    tool_events = [
        event
        for event in (sample_payload.get("events", []) or [])
        if event.get("event") == "tool"
    ]

    lines = [
        f"# Sample {row.get('sample_uuid', 'unknown')}",
        "",
        f"- eval_archive: {archive_name}",
        f"- sample_member: {sample_member or 'missing'}",
        f"- task_id: {row.get('task_id')}",
        f"- side_task_name: {row.get('side_task_name')}",
        f"- source_name: {row.get('source_name')}",
        f"- attack_model: {row.get('attack_model')}",
        f"- message_count: {len(messages)}",
        f"- tool_event_count: {len(tool_events)}",
        "",
    ]

    if eval_metadata:
        lines.extend(
            [
                "## Eval Metadata",
                "",
                f"- created: {eval_metadata.get('created')}",
                f"- task: {eval_metadata.get('task')}",
                f"- task_id: {eval_metadata.get('task_id')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Task Input",
            "",
            "```text",
            str(sample_payload.get("input", "")).strip(),
            "```",
            "",
        ]
    )

    for label, key in [
        ("Main Task Description", "main_task_description"),
        ("Side Task Description", "side_task_description"),
        ("Setting Description", "setting_description"),
    ]:
        value = metadata.get(key)
        if _has_text(value):
            lines.extend([f"## {label}", "", "```text", str(value).strip(), "```", ""])

    lines.extend(["## Messages", ""])
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        content = stringify_content(message.get("content")) or "<empty>"
        lines.extend(
            [
                f"### Message {index}: {role}",
                "",
                "```text",
                content,
                "```",
                "",
            ]
        )

    lines.extend(["## Tool Events", ""])
    for index, event in enumerate(tool_events, start=1):
        lines.append(f"### Tool Event {index}: {event.get('function') or event.get('name') or 'unknown'}")
        lines.append("")
        lines.append(f"- timestamp: {event.get('timestamp')}")
        if event.get("type") is not None:
            lines.append(f"- type: {event.get('type')}")
        if event.get("message_id") is not None:
            lines.append(f"- message_id: {event.get('message_id')}")
        lines.append("")

        arguments = event.get("arguments")
        if arguments:
            lines.extend(["#### Arguments", "", "```json", json.dumps(arguments, indent=2, ensure_ascii=True), "```", ""])

        result = event.get("result")
        output = event.get("result") if isinstance(result, str) else event.get("output")
        if output not in (None, ""):
            lines.extend(["#### Output", "", "```text", str(output)[:12000], "```", ""])

    return "\n".join(lines)


def iter_message_fragments(payload: Any) -> list[dict[str, str]]:
    fragments: list[dict[str, str]] = []
    stack: list[tuple[str, Any]] = [("$", payload)]

    while stack:
        path, node = stack.pop()
        if isinstance(node, dict):
            role = node.get("role")
            name = node.get("name") or node.get("tool_name") or node.get("function")
            content = stringify_content(node.get("content"))
            if isinstance(role, str) and content:
                fragments.append(
                    {
                        "path": path,
                        "role": role,
                        "name": str(name) if name is not None else "",
                        "content": content,
                    }
                )
            elif content and any(key in node for key in ("tool_calls", "output", "text", "message")):
                fragments.append(
                    {
                        "path": path,
                        "role": str(role) if role is not None else "",
                        "name": str(name) if name is not None else "",
                        "content": content,
                    }
                )

            for key, value in reversed(list(node.items())):
                stack.append((f"{path}.{key}", value))
        elif isinstance(node, list):
            for index in range(len(node) - 1, -1, -1):
                stack.append((f"{path}[{index}]", node[index]))

    return fragments


def is_real_eval_member(name: str) -> bool:
    path_name = Path(name).name
    return (
        name.endswith(".eval")
        and not name.startswith("__MACOSX/")
        and "/._" not in name
        and not path_name.startswith("._")
    )


def write_zip_inventory(zip_path: Path, output_dir: Path) -> None:
    if not zip_path.exists():
        return

    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/") or "/._" in info.filename:
                continue

            record: dict[str, Any] = {
                "archive_path": info.filename,
                "basename": Path(info.filename).name,
                "size_bytes": info.file_size,
            }
            if is_real_eval_member(info.filename):
                parts = Path(info.filename).parts
                if len(parts) >= 5:
                    record["side_task_name"] = parts[1]
                    record["source_name"] = parts[2]
                    record["subset"] = parts[3]
                record["kind"] = "eval_archive"
            elif info.filename.endswith(".parquet"):
                record["kind"] = "parquet"
            else:
                record["kind"] = "other"
            records.append(record)

    write_jsonl(records, output_dir / "zip_inventory.jsonl")


def try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def decode_bytes(raw_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "sample"


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset_dir, args.zip_path)
    selected_rows = select_rows(rows, args)

    assert_no_orphan_eval_files(
        args.output_dir,
        {str(row["sample_uuid"]) for row in selected_rows if row.get("sample_uuid")},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_zip_inventory(args.zip_path, args.output_dir)
    extracted_logs = maybe_extract_logs(selected_rows, args.zip_path, args.output_dir)

    records = [
        build_manifest_record(
            row,
            extracted_log_path=(extracted_logs.get(row.get("sample_uuid", "")) or {}).get("transcript_path"),
        )
        for row in selected_rows
    ]
    for record in records:
        artifact_paths = extracted_logs.get(record.get("sample_uuid", "")) or {}
        if artifact_paths.get("eval_path"):
            record["per_sample_eval_path"] = artifact_paths["eval_path"]

    write_jsonl(records, args.output_dir / "examples.jsonl")
    write_markdown(records, args.output_dir / "examples.md", args.zip_path.exists())

    print(f"Loaded rows: {len(rows)}")
    print(f"Selected rows: {len(selected_rows)}")
    print(f"Wrote: {(args.output_dir / 'examples.jsonl').resolve()}")
    print(f"Wrote: {(args.output_dir / 'examples.md').resolve()}")
    if args.zip_path.exists():
        print(f"Wrote: {(args.output_dir / 'zip_inventory.jsonl').resolve()}")
    if args.zip_path.exists():
        print(f"Raw logs extracted from: {args.zip_path.resolve()}")
    else:
        print(
            "Raw log zip not found. To extract transcripts, place final_dataset.zip at "
            f"{args.zip_path.resolve()} or pass --zip-path."
        )


if __name__ == "__main__":
    main()
