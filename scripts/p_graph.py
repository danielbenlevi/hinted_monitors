from __future__ import annotations

# brainstorming:
# explicitly change all env vars to their actual strings
# look at formal cyber works for catching suscpicious things, GNNs, put few-shot examples of bad explaining away in system prompt, try just regex
# for each type of bad action, regex for the bad explaining, meta-monitor for bad explaining away (motivated reasoning), NB for bad reasoning / meta / S-scores?
# adding this cyber-specific tooling seems valid, as it should help with known attack patterns, still gives the LLM judgment, and worse case we can run a no-hint
# monitor as well, resulting in little loss. maybe under this logic adding the specific attack framework knowledge is also valid? but that might be contaminating?

import argparse
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


INPUT_PATH = "transcript.txt"
OUT_DIR = None
SAMPLE_JSON_PATH = "sample.json"
STEP_SCAN_PAYLOADS_PATH = "step_scan_payloads.jsonl"
SCANNER_FINDINGS_PATH = "scanner_findings.jsonl"
SCAN_PAYLOADS_DIR = "scan_payloads"
DEFAULT_SCAN_CONTEXT_WINDOW = 2
DEFAULT_SCAN_OUTPUT_CHARS = 12000
DEFAULT_CONTEXT_OUTPUT_CHARS = 3000
SCANNER_TIMEOUT_SECONDS = 20
YARA_RULES_PATH = Path(__file__).with_name("bash_monitor_patterns.yar")


TOOLISH_PREFIXES = (
    "python ",
    "python3 ",
    "python -m pytest",
    "pytest",
    "git ",
    "cat ",
    "sed ",
    "head ",
    "tail ",
    "grep ",
    "rg ",
    "curl ",
    "wget ",
    "pip ",
    "npm ",
    "yarn ",
    "poetry ",
    "conda ",
    "apt ",
    "bash ",
    "sh ",
    "node ",
    "apply_patch",
)

TEST_COMMAND_PREFIXES = (
    "pytest",
    "python -m pytest",
    "python3 -m pytest",
    "unittest",
    "npm test",
    "jest",
    "go test",
    "cargo test",
    "mvn test",
    "gradle test",
)

SHELL_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
REDIRECTION_RE = re.compile(r"(?<![\d<])(>>?|<(?!<))\s*([^\s|&;]+)")
URL_RE = re.compile(r"\b(?:https?|ftp|git\+https?)://[^\s'\"`<>]+", re.IGNORECASE)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")
GREP_OUTPUT_RE = re.compile(
    r"^(?P<path>(?:[A-Za-z]:)?[^:\n]+?):(?P<line>\d+)(?::(?P<col>\d+))?:(?P<snippet>.*)$"
)
TRACEBACK_RE = re.compile(r'File "([^"]+)", line (\d+)')
FAILED_TEST_RE = re.compile(r"^(FAILED|ERROR)\s+([^\s]+)", re.MULTILINE)
PYTEST_SUMMARY_RE = re.compile(r"(\d+)\s+(failed|passed|error|errors|skipped|xfailed|xpassed)")
PY_DEF_RE = re.compile(r"^[^\S\r\n]*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
PY_CLASS_RE = re.compile(r"^[^\S\r\n]*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|:)", re.MULTILINE)
JS_FUNC_RE = re.compile(
    r"^[^\S\r\n]*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)
JS_CLASS_RE = re.compile(r"^[^\S\r\n]*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.MULTILINE)
JS_CONST_RE = re.compile(r"^[^\S\r\n]*(?:export\s+)?const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE)
GO_FUNC_RE = re.compile(r"^[^\S\r\n]*func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
RUST_FN_RE = re.compile(r"^[^\S\r\n]*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
JAVA_METHOD_RE = re.compile(
    r"^[^\S\r\n]*(?:public|private|protected)?[^\S\r\n]*(?:static[^\S\r\n]+)?[\w<>\[\], ?]+[^\S\r\n]+([A-Za-z_][A-Za-z0-9_]*)[^\S\r\n]*\(",
    re.MULTILINE,
)
IMPORT_RE = re.compile(
    r"^[^\S\r\n]*(?:from\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+import|import\s+([A-Za-z_][A-Za-z0-9_\.]*))",
    re.MULTILINE,
)
OPEN_CALL_RE = re.compile(r"open\(\s*['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]*)['\"])?")
MAKEDIRS_RE = re.compile(r"(?:os\.makedirs|Path\(['\"][^'\"]+['\"]\)\.mkdir)\(")
PATH_TOKEN_RE = re.compile(r"(?:[A-Za-z]:)?(?:/|\.{1,2}/)[^\s'\"`<>]+|[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+")
SECRET_HINT_RE = re.compile(r"(secret|token|password|api[_-]?key|private[_-]?key|auth)", re.IGNORECASE)


@dataclass
class Event:
    type: str
    target: str
    attrs: dict[str, Any] = field(default_factory=dict)
    evidence: str = ""
    source_kind: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "target": self.target,
            "attrs": self.attrs,
            "evidence": self.evidence,
            "source_kind": self.source_kind,
        }


@dataclass
class Step:
    step_id: int
    raw_text: str
    action_text: str
    output_text: str
    line_start: int
    line_end: int
    detected_command_type: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    assistant_message_index: int | None = None
    tool_message_index: int | None = None
    events: list[Event] = field(default_factory=list)


@dataclass
class StepScanPayload:
    step_id: int
    action_preview: str
    current_payload_text: str
    contextual_payload_text: str
    current_payload_path: Path | None = None
    contextual_payload_path: Path | None = None
    context_step_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action_preview": self.action_preview,
            "current_payload_path": str(self.current_payload_path) if self.current_payload_path else None,
            "contextual_payload_path": str(self.contextual_payload_path) if self.contextual_payload_path else None,
            "context_step_ids": self.context_step_ids,
            "current_payload_chars": len(self.current_payload_text),
            "contextual_payload_chars": len(self.contextual_payload_text),
        }


@dataclass
class StateLedger:
    files_seen: set[str] = field(default_factory=set)
    last_reader_by_file: dict[str, int] = field(default_factory=dict)
    last_writer_by_file: dict[str, int] = field(default_factory=dict)
    last_deleter_by_file: dict[str, int] = field(default_factory=dict)
    last_chmod_by_file: dict[str, int] = field(default_factory=dict)
    symbols_seen_by_file: dict[str, set[str]] = field(default_factory=dict)
    tests_seen: set[str] = field(default_factory=set)
    last_run_step_by_test: dict[str, int] = field(default_factory=dict)
    last_test_status_by_test: dict[str, str] = field(default_factory=dict)
    packages_seen: set[str] = field(default_factory=set)
    external_resources_seen: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_seen": sorted(self.files_seen),
            "last_reader_by_file": dict(sorted(self.last_reader_by_file.items())),
            "last_writer_by_file": dict(sorted(self.last_writer_by_file.items())),
            "last_deleter_by_file": dict(sorted(self.last_deleter_by_file.items())),
            "last_chmod_by_file": dict(sorted(self.last_chmod_by_file.items())),
            "symbols_seen_by_file": {
                path: sorted(symbols) for path, symbols in sorted(self.symbols_seen_by_file.items())
            },
            "tests_seen": sorted(self.tests_seen),
            "last_test_status_by_test": dict(sorted(self.last_test_status_by_test.items())),
            "packages_seen": sorted(self.packages_seen),
            "external_resources_seen": sorted(self.external_resources_seen),
        }


class ProvenanceGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []

    def add_node(self, node_id: str, node_type: str, label: str | None = None, attrs: dict[str, Any] | None = None) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "type": node_type,
                "label": label or node_id,
                "attrs": attrs or {},
            }

    def add_edge(self, source: str, target: str, edge_type: str, attrs: dict[str, Any] | None = None) -> None:
        self.edges.append(
            {
                "source": source,
                "target": target,
                "type": edge_type,
                "attrs": attrs or {},
            }
        )

    def to_dict(self, input_file: str) -> dict[str, Any]:
        return {
            "metadata": {
                "input_file": input_file,
                "no_execution": True,
                "no_llm_calls": True,
                "description": "Transcript-derived provenance graph.",
            },
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build transcript provenance artifacts from transcript text or sample.json.")
    parser.add_argument("input_path", nargs="?", default=INPUT_PATH)
    parser.add_argument("--sample-json", default=None)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--yara-rules", default=str(YARA_RULES_PATH))
    parser.add_argument("--scan-context-window", type=int, default=DEFAULT_SCAN_CONTEXT_WINDOW)
    parser.add_argument("--skip-scanners", action="store_true", help="Build payloads but skip YARA and Gitleaks execution.")
    args = parser.parse_args()

    transcript_path, sample_json_path = resolve_input_paths(args.input_path, args.sample_json)
    transcript_text = read_transcript(transcript_path)
    sample_payload = read_sample_json(sample_json_path)

    steps, warnings = segment_transcript(transcript_text, sample_payload)
    ledger = StateLedger()
    step_summaries: list[dict[str, Any]] = []
    events_by_step: dict[int, list[Event]] = {}

    for step in steps:
        step.events = dedupe_events(extract_events(step, warnings))
        before_state = capture_before_state(ledger, step.events)
        after_state_updates = update_state_ledger(ledger, step, step.events)
        summary = {
            "step_id": step.step_id,
            "line_start": step.line_start,
            "line_end": step.line_end,
            "action_preview": compact_preview(step.action_text or step.raw_text),
            "detected_command_type": step.detected_command_type,
            "before_state": before_state,
            "observed_events": [event.to_dict() for event in step.events],
            "after_state_updates": after_state_updates,
            "provenance_relations": {
                "depends_on_steps": [],
                "later_dependent_steps": [],
            },
        }
        step_summaries.append(summary)
        events_by_step[step.step_id] = step.events

    graph = build_graph(steps, events_by_step)
    add_cross_step_provenance_edges(graph, step_summaries, steps, events_by_step)

    out_dir = resolve_output_dir(args.out_dir, args.input_path, transcript_path, sample_json_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_payloads = build_step_scan_payloads(
        steps,
        step_summaries,
        context_window=max(0, args.scan_context_window),
    )
    persist_step_scan_payloads(scan_payloads, out_dir / SCAN_PAYLOADS_DIR)
    scanner_findings = [] if args.skip_scanners else run_step_scanners(scan_payloads, Path(args.yara_rules), warnings)
    write_jsonl(out_dir / "step_summaries.jsonl", step_summaries)
    write_jsonl(out_dir / STEP_SCAN_PAYLOADS_PATH, [payload.to_dict() for payload in scan_payloads])
    write_jsonl(out_dir / SCANNER_FINDINGS_PATH, scanner_findings)
    write_json(out_dir / "provenance_graph.json", graph.to_dict(transcript_path.name if transcript_path else str(sample_json_path)))
    write_json(out_dir / "state_ledger_final.json", ledger.to_dict())
    write_jsonl(out_dir / "extraction_warnings.jsonl", warnings)


def resolve_input_paths(input_path_value: str, sample_json_value: str | None) -> tuple[Path | None, Path | None]:
    input_path = Path(input_path_value)

    if input_path.is_dir():
        transcript_path = first_existing(
            input_path / "transcript.md",
            input_path / "transcript.txt",
            input_path / "transcript.jsonl",
        )
        sample_json_path = first_existing(
            Path(sample_json_value) if sample_json_value else None,
            input_path / "sample.json",
        )
        return transcript_path, sample_json_path

    transcript_candidate: Path | None = None
    sample_candidate: Path | None = None
    if input_path.exists() and input_path.suffix.lower() == ".json":
        sample_candidate = input_path
        transcript_candidate = first_existing(input_path.with_name("transcript.md"), input_path.with_name("transcript.txt"))
    else:
        transcript_candidate = input_path if input_path.exists() else None
        sample_candidate = first_existing(
            Path(sample_json_value) if sample_json_value else None,
            input_path.with_name("sample.json") if input_path.parent.exists() else None,
        )
    return transcript_candidate, sample_candidate


def resolve_output_dir(
    out_dir_value: str | None,
    input_path_value: str,
    transcript_path: Path | None,
    sample_json_path: Path | None,
) -> Path:
    if out_dir_value:
        return Path(out_dir_value)
    input_path = Path(input_path_value)
    if input_path.is_dir():
        return input_path
    if sample_json_path is not None:
        return sample_json_path.parent
    if transcript_path is not None:
        return transcript_path.parent
    return Path(".")


def first_existing(*paths: Path | None) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path
    return None


def read_transcript(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_sample_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def segment_transcript(text: str, sample_payload: dict[str, Any] | None) -> tuple[list[Step], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    if sample_payload is not None:
        return build_steps_from_sample(text, sample_payload, warnings), warnings
    if text.strip():
        warnings.append(
            {
                "step_id": None,
                "warning_type": "segmentation_fallback",
                "message": "No sample.json supplied; using heuristic text-only segmentation.",
                "evidence": text[:200],
            }
        )
        return build_steps_from_text(text), warnings
    return [], warnings


def build_steps_from_sample(
    transcript_text: str,
    sample_payload: dict[str, Any],
    warnings: list[dict[str, Any]],
) -> list[Step]:
    messages = sample_payload.get("messages", []) or []
    message_ranges = parse_transcript_message_ranges(transcript_text)
    steps: list[Step] = []
    step_id = 1

    for assistant_index, message in enumerate(messages):
        tool_calls = message.get("tool_calls") or []
        if message.get("role") != "assistant" or not tool_calls:
            continue

        for tool_call in tool_calls:
            tool_message_index = find_tool_message_index(messages, assistant_index + 1, tool_call.get("id"))
            output_text = ""
            if tool_message_index is not None:
                output_text = stringify_content(messages[tool_message_index].get("content"))

            action_text = render_tool_call(tool_call)
            assistant_range = message_ranges.get(assistant_index + 1)
            tool_range = message_ranges.get(tool_message_index + 1) if tool_message_index is not None else None
            line_start = assistant_range["line_start"] if assistant_range else estimate_message_line(messages, assistant_index)
            line_end = tool_range["line_end"] if tool_range else (assistant_range["line_end"] if assistant_range else line_start)

            raw_parts = [action_text]
            if output_text:
                raw_parts.extend(["", output_text])
            raw_text = "\n".join(raw_parts).strip()

            step = Step(
                step_id=step_id,
                raw_text=raw_text,
                action_text=action_text,
                output_text=output_text,
                line_start=line_start,
                line_end=line_end,
                detected_command_type=detect_command_type(action_text, tool_call.get("function")),
                tool_name=tool_call.get("function"),
                tool_args=tool_call.get("arguments") or {},
                assistant_message_index=assistant_index,
                tool_message_index=tool_message_index,
            )
            steps.append(step)
            step_id += 1

    if not steps:
        warnings.append(
            {
                "step_id": None,
                "warning_type": "segmentation_empty",
                "message": "sample.json was present but no assistant tool-call messages were found.",
                "evidence": json.dumps(sample_payload)[:500],
            }
        )

    return steps


def parse_transcript_message_ranges(text: str) -> dict[int, dict[str, Any]]:
    if not text.strip():
        return {}

    lines = text.splitlines()
    ranges: dict[int, dict[str, Any]] = {}
    current_number: int | None = None
    current_role = ""
    start_line = 1

    for idx, line in enumerate(lines, start=1):
        match = re.match(r"^### Message (\d+): (\w+)", line)
        if match:
            if current_number is not None:
                ranges[current_number] = {
                    "role": current_role,
                    "line_start": start_line,
                    "line_end": idx - 1,
                }
            current_number = int(match.group(1))
            current_role = match.group(2)
            start_line = idx

    if current_number is not None:
        ranges[current_number] = {
            "role": current_role,
            "line_start": start_line,
            "line_end": len(lines),
        }

    return ranges


def estimate_message_line(messages: list[dict[str, Any]], message_index: int) -> int:
    return (message_index + 1) * 10


def find_tool_message_index(messages: list[dict[str, Any]], start_index: int, tool_call_id: str | None) -> int | None:
    if not tool_call_id:
        return None
    for index in range(start_index, len(messages)):
        message = messages[index]
        if message.get("role") == "tool" and message.get("tool_call_id") == tool_call_id:
            return index
    return None


def render_tool_call(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") or "unknown_tool"
    arguments = tool_call.get("arguments") or {}
    if function == "bash":
        return str(arguments.get("cmd", "")).strip()
    if function == "python":
        return str(arguments.get("code", "")).strip()
    if function == "text_editor":
        return json.dumps({"tool": "text_editor", **arguments}, ensure_ascii=True)
    if function == "submit":
        return "submit"
    return json.dumps({"tool": function, **arguments}, ensure_ascii=True)


def build_steps_from_text(text: str) -> list[Step]:
    lines = text.splitlines()
    start_indices: list[int] = []
    in_fence = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if is_step_start(stripped, in_fence):
            start_indices.append(index)

    if not start_indices:
        return [
            Step(
                step_id=1,
                raw_text=text,
                action_text=text,
                output_text="",
                line_start=1,
                line_end=max(1, len(lines)),
                detected_command_type=detect_command_type(text),
            )
        ]

    steps: list[Step] = []
    for step_id, start in enumerate(start_indices, start=1):
        end = start_indices[step_id] - 1 if step_id < len(start_indices) else len(lines) - 1
        chunk_lines = lines[start : end + 1]
        raw_text = "\n".join(chunk_lines).strip()
        steps.append(
            Step(
                step_id=step_id,
                raw_text=raw_text,
                action_text=raw_text,
                output_text="",
                line_start=start + 1,
                line_end=end + 1,
                detected_command_type=detect_command_type(raw_text),
            )
        )
    return steps


def is_step_start(line: str, in_fence: bool) -> bool:
    if not line:
        return False
    if line.startswith("$ ") or line.startswith("> "):
        return True
    if line.startswith("*** Begin Patch") or line.startswith("diff --git") or line.startswith("--- a/") or line.startswith("+++ b/"):
        return True
    lowered = line.lower()
    if any(lowered.startswith(prefix) for prefix in TOOLISH_PREFIXES):
        return True
    if in_fence and any(lowered.startswith(prefix) for prefix in TOOLISH_PREFIXES):
        return True
    return False


def detect_command_type(text: str, tool_name: str | None = None) -> str:
    lowered = text.lower()
    if "*** begin patch" in lowered or lowered.startswith("apply_patch"):
        return "apply_patch"
    if "diff --git" in lowered or lowered.startswith("--- a/") or lowered.startswith("+++ b/"):
        return "diff"
    if any(prefix in lowered for prefix in ("pytest", "unittest", "cargo test", "npm test", "jest", "go test", "mvn test", "gradle test")):
        return "test"
    if re.search(r"(^|\s)(rg|grep)\s", lowered):
        return "grep"
    if tool_name in {"bash", "python", "text_editor"}:
        return "shell"
    if any(lowered.startswith(prefix) for prefix in TOOLISH_PREFIXES):
        return "shell"
    return "unknown"


def extract_events(step: Step, warnings: list[dict[str, Any]]) -> list[Event]:
    events: list[Event] = []

    if step.tool_name == "bash":
        events.extend(parse_shell_events(step, warnings))
    elif step.tool_name == "text_editor":
        events.extend(parse_text_editor_events(step))
    elif step.tool_name == "python":
        events.extend(parse_python_tool_events(step))
    else:
        if "*** Begin Patch" in step.raw_text:
            events.extend(parse_patch_events(step, warnings))
        if "diff --git" in step.raw_text or "\n--- a/" in step.raw_text:
            events.extend(parse_diff_events(step, warnings))
        if step.detected_command_type in {"shell", "test", "grep"}:
            events.extend(parse_shell_events(step, warnings))

    events.extend(parse_test_output_events(step))
    events.extend(parse_traceback_events(step))
    events.extend(parse_grep_output_events(step))
    events.extend(extract_pattern_flags(step.action_text, "command", step.step_id))

    if step.output_text:
        events.extend(extract_pattern_flags(step.output_text, "output", step.step_id))

    return events


def parse_shell_events(step: Step, warnings: list[dict[str, Any]]) -> list[Event]:
    command_text = step.action_text.strip()
    if not command_text:
        return []

    events: list[Event] = []
    shell_path_bindings: dict[str, list[str]] = {}
    segments = split_shell_command(command_text)
    if not segments:
        segments = [command_text]

    for segment in segments:
        heredoc_body = extract_heredoc_payload(segment)
        segment_head = segment.splitlines()[0] if "\n" in segment else segment
        update_shell_path_bindings(segment_head, shell_path_bindings)
        tokens = parse_shell_tokens(segment_head, segment)
        if tokens is None:
            try:
                tokens = shlex.split(segment_head, posix=True)
            except ValueError:
                warnings.append(
                    {
                        "step_id": step.step_id,
                        "warning_type": "complex_command_parse",
                        "message": "shlex could not parse command segment; using regex fallback.",
                        "evidence": segment_head,
                    }
                )
                tokens = regex_shell_split(segment_head)

        if not tokens:
            continue

        cmd = skip_env_assignments(tokens)
        if not cmd:
            continue
        first = cmd[0]

        if first in {"python", "python3"} and "-c" in cmd:
            inline_code = extract_inline_python_code(segment)
            if inline_code:
                python_events = parse_python_source_text(inline_code, source_kind="command")
                events.extend(expand_dynamic_event_targets(python_events, shell_path_bindings))

        if first in {"cat", "sed", "head", "tail", "less", "more", "awk", "nm", "ldd", "wc"}:
            for token in cmd[1:]:
                if token.startswith("-") or ">" in token or "<" in token:
                    continue
                if token == "1" or token == "2":
                    continue
                for path in expand_shell_path_token(token, shell_path_bindings):
                    attrs = {"operation": "read", "source": first}
                    if first in {"python", "bash", "sh", "node"}:
                        attrs["execution_role"] = "script"
                    if first == "cat":
                        attrs["symbols"] = extract_symbols_from_text(step.output_text)
                    events.append(Event("file_read", path, attrs, segment, "command"))
        elif first in {"grep", "rg"}:
            path_tokens = extract_grep_like_targets(cmd)
            for token in path_tokens:
                for path in expand_shell_path_token(token, shell_path_bindings):
                    events.append(Event("file_read", path, {"operation": "search", "source": first}, segment, "command"))
        elif first in {"ls", "find"}:
            for token in cmd[1:]:
                if token.startswith("-") or ">" in token or "<" in token:
                    continue
                for path in expand_shell_path_token(token, shell_path_bindings):
                    events.append(Event("file_read", path, {"operation": "list", "source": first}, segment, "command"))
        elif first in {"curl", "wget"}:
            for path in extract_http_output_paths(cmd):
                events.append(Event("file_write", path, {"operation": "download", "source": first}, segment, "command"))
        elif first in {"python", "python3", "bash", "sh", "node"}:
            script_path = first_non_option_path(cmd[1:])
            if script_path:
                for path in expand_shell_path_token(script_path, shell_path_bindings):
                    attrs = {"operation": "execute", "source": first, "execution_role": "script"}
                    events.append(Event("file_read", path, attrs, segment, "command"))
                    events.extend(extract_pattern_flags(f"{first} {path}", "command", step.step_id))
            if first in {"python", "python3"} and ("-m" in cmd or "-c" in cmd):
                events.append(Event("pattern_flag", "opaque_script_execution", {"source": first}, segment, "heuristic"))
        elif first == "cp" and len(cmd) >= 3:
            src_candidates = expand_shell_path_token(cmd[-2], shell_path_bindings)
            dst_candidates = expand_shell_path_token(cmd[-1], shell_path_bindings)
            if src_candidates:
                for src in src_candidates:
                    events.append(Event("file_read", src, {"operation": "copy_source", "source": first}, segment, "command"))
            if dst_candidates:
                from_path = src_candidates[0] if src_candidates else normalize_path(cmd[-2])
                for dst in dst_candidates:
                    events.append(Event("file_write", dst, {"operation": "copied", "source": first, "from": from_path}, segment, "command"))
        elif first == "ln" and "-s" in cmd:
            operands = [token for token in cmd[1:] if not token.startswith("-")]
            if len(operands) >= 2:
                sources = expand_shell_path_token(operands[-2], shell_path_bindings)
                destinations = expand_shell_path_token(operands[-1], shell_path_bindings)
                if sources and destinations:
                    events.append(
                        Event(
                            "file_write",
                            destinations[0],
                            {"operation": "symlink", "source": first, "from": sources[0]},
                            segment,
                            "command",
                        )
                    )
        elif first == "mv" and len(cmd) >= 3:
            src_candidates = expand_shell_path_token(cmd[-2], shell_path_bindings)
            dst_candidates = expand_shell_path_token(cmd[-1], shell_path_bindings)
            if src_candidates and dst_candidates:
                events.append(
                    Event(
                        "file_rename",
                        dst_candidates[0],
                        {"source": src_candidates[0], "operation": "renamed"},
                        segment,
                        "command",
                    )
                )
        elif first == "mkdir":
            for token in cmd[1:]:
                if token.startswith("-") or ">" in token or "<" in token:
                    continue
                for path in expand_shell_path_token(token, shell_path_bindings):
                    events.append(Event("file_write", path, {"operation": "mkdir", "source": first}, segment, "command"))
        elif first == "touch":
            for token in cmd[1:]:
                if token.startswith("-") or ">" in token or "<" in token:
                    continue
                for path in expand_shell_path_token(token, shell_path_bindings):
                    events.append(Event("file_write", path, {"operation": "touch", "source": first}, segment, "command"))
        elif first == "rm":
            for token in cmd[1:]:
                if token.startswith("-") or ">" in token or "<" in token:
                    continue
                for path in expand_shell_path_token(token, shell_path_bindings):
                    events.append(Event("file_delete", path, {"operation": "rm", "source": first}, segment, "command"))
        elif first == "chmod":
            mode = cmd[1] if len(cmd) > 1 else ""
            for token in cmd[2:]:
                if token.startswith("-") or ">" in token or "<" in token:
                    continue
                for path in expand_shell_path_token(token, shell_path_bindings):
                    events.append(Event("file_chmod", path, {"mode": mode, "source": first}, segment, "command"))
        elif first == "chown":
            owner = cmd[1] if len(cmd) > 1 else ""
            for token in cmd[2:]:
                if token.startswith("-") or ">" in token or "<" in token:
                    continue
                for path in expand_shell_path_token(token, shell_path_bindings):
                    events.append(Event("file_chmod", path, {"owner": owner, "source": first}, segment, "command"))
        elif first == "tee":
            for token in cmd[1:]:
                if token.startswith("-") or ">" in token or "<" in token:
                    continue
                for path in expand_shell_path_token(token, shell_path_bindings):
                    events.append(Event("file_write", path, {"operation": "tee", "source": first}, segment, "command"))

        if first.startswith("/") and first not in {"/bin/sh", "/bin/bash", "/usr/bin/env"}:
            events.append(
                Event(
                    "file_read",
                    normalize_path(first),
                    {"operation": "execute", "source": "absolute_command", "execution_role": "program"},
                    segment,
                    "command",
                )
            )

        if is_test_command(cmd):
            test_targets = extract_test_targets_from_command(cmd)
            if not test_targets:
                test_targets = [" ".join(cmd)]
            for target in test_targets:
                events.append(Event("test_run", target, {"status": infer_test_status(step.output_text)}, segment, "command"))

        events.extend(parse_package_command(cmd, segment_head))
        if command_executes_network_contact(cmd, segment_head):
            events.extend(parse_external_resources(segment_head, "command"))

        heredoc_target: str | None = None
        for redir_op, redir_target in find_shell_redirections(segment_head):
            paths = expand_shell_path_token(redir_target, shell_path_bindings)
            if not paths:
                path = normalize_path(redir_target)
                if looks_dynamic_shell_path(path):
                    continue
                warnings.append(
                    {
                        "step_id": step.step_id,
                        "warning_type": "ambiguous_parse",
                        "message": "Could not determine destination path for redirection.",
                        "evidence": segment,
                    }
                )
                continue
            if redir_op == "<":
                for path in paths:
                    events.append(
                        Event(
                            "file_read",
                            path,
                            {"operation": "redirect_input", "source": "redirection"},
                            segment,
                            "command",
                        )
                    )
            else:
                operation = "append" if redir_op == ">>" else "overwrite"
                for path in paths:
                    events.append(
                        Event(
                            "file_write",
                            path,
                            {"operation": operation, "source": "redirection"},
                            segment,
                            "command",
                        )
                    )
                heredoc_target = paths[0]

        if heredoc_body and heredoc_target:
            for symbol in extract_symbols_from_text(heredoc_body):
                events.append(
                    Event(
                        "symbol_change",
                        symbol,
                        {"file": heredoc_target},
                        compact_preview(heredoc_body),
                        "heuristic",
                    )
                )
            events.extend(parse_imports_as_package_use(heredoc_body))
            events.extend(extract_pattern_flags(heredoc_body, "heuristic", step.step_id))

    return events


def parse_text_editor_events(step: Step) -> list[Event]:
    args = step.tool_args
    command = str(args.get("command", "")).strip()
    path = normalize_path(str(args.get("path", "")).strip())
    events: list[Event] = []
    text_for_symbols = ""

    if command == "view" and path:
        symbols = extract_symbols_from_text(strip_line_numbers(step.output_text))
        events.append(
            Event(
                "file_read",
                path,
                {"operation": "view", "source": "text_editor", "symbols": symbols},
                json.dumps(args, ensure_ascii=True),
                "command",
            )
        )
        text_for_symbols = strip_line_numbers(step.output_text)
    elif command == "create" and path:
        file_text = str(args.get("file_text", ""))
        symbols = extract_symbols_from_text(file_text)
        attrs = {
            "operation": "created",
            "source": "text_editor",
            "symbols": symbols,
            "added_ips": unique_list(IP_RE.findall(file_text)),
        }
        events.append(
            Event(
                "file_write",
                path,
                attrs,
                json.dumps(args, ensure_ascii=True)[:4000],
                "command",
            )
        )
        text_for_symbols = file_text
    elif command == "str_replace" and path:
        old_str = str(args.get("old_str", ""))
        new_str = str(args.get("new_str", ""))
        symbols = sorted(set(extract_symbols_from_text(old_str) + extract_symbols_from_text(new_str)))
        old_ips = set(IP_RE.findall(old_str))
        new_ips = set(IP_RE.findall(new_str))
        events.append(
            Event(
                "file_write",
                path,
                {
                    "operation": "modified",
                    "source": "text_editor",
                    "symbols": symbols,
                    "added_ips": sorted(new_ips - old_ips),
                    "removed_ips": sorted(old_ips - new_ips),
                },
                (
                    f"str_replace {path}; old={compact_preview(old_str, 700)}; "
                    f"new={compact_preview(new_str, 700)}"
                ),
                "command",
            )
        )
        text_for_symbols = "\n".join([old_str, new_str])
    elif command == "insert" and path:
        insert_text = str(args.get("insert_text", ""))
        symbols = extract_symbols_from_text(insert_text)
        events.append(
            Event(
                "file_write",
                path,
                {"operation": "modified", "source": "text_editor", "symbols": symbols},
                json.dumps(args, ensure_ascii=True)[:4000],
                "command",
            )
        )
        text_for_symbols = insert_text

    if path and text_for_symbols:
        for symbol in extract_symbols_from_text(text_for_symbols):
            events.append(
                Event(
                    "symbol_change",
                    symbol,
                    {"file": path},
                    compact_preview(text_for_symbols),
                    "heuristic",
                )
            )
        events.extend(parse_imports_as_package_use(text_for_symbols))
        events.extend(extract_pattern_flags(text_for_symbols, "heuristic", step.step_id))

    return events


def parse_python_tool_events(step: Step) -> list[Event]:
    code = step.action_text
    events: list[Event] = [Event("pattern_flag", "opaque_script_execution", {}, compact_preview(code), "heuristic")]
    events.extend(parse_python_source_text(code, source_kind="command"))
    events.extend(extract_pattern_flags(code, "command", step.step_id))
    return events


def parse_python_source_text(code: str, source_kind: str) -> list[Event]:
    events: list[Event] = []

    for match in OPEN_CALL_RE.finditer(code):
        path = normalize_path(match.group(1))
        mode = match.group(2) or "r"
        if "w" in mode or "a" in mode or "x" in mode:
            events.append(Event("file_write", path, {"operation": f"open:{mode}", "source": "python"}, match.group(0), source_kind))
        else:
            events.append(Event("file_read", path, {"operation": f"open:{mode}", "source": "python"}, match.group(0), source_kind))

    for path in re.findall(r"os\.makedirs\(\s*['\"]([^'\"]+)['\"]", code):
        events.append(Event("file_write", normalize_path(path), {"operation": "mkdir", "source": "python"}, f"os.makedirs('{path}')", source_kind))

    events.extend(parse_external_resources(code, source_kind))
    events.extend(parse_imports_as_package_use(code))
    return events


def parse_patch_events(step: Step, warnings: list[dict[str, Any]]) -> list[Event]:
    lines = step.raw_text.splitlines()
    events: list[Event] = []
    current_file: str | None = None
    current_added: list[str] = []
    current_removed: list[str] = []
    pending_rename_from: str | None = None

    def flush_symbols(file_path: str | None) -> None:
        if not file_path:
            return
        text = "\n".join(current_added + current_removed)
        for symbol in extract_symbols_from_text(text):
            events.append(Event("symbol_change", symbol, {"file": file_path}, compact_preview(text), "patch"))
        current_added.clear()
        current_removed.clear()

    for line in lines:
        if line.startswith("*** Update File: "):
            flush_symbols(current_file)
            current_file = normalize_path(line.split(": ", 1)[1])
            events.append(Event("file_write", current_file, {"operation": "modified", "source": "apply_patch"}, line, "patch"))
        elif line.startswith("*** Add File: "):
            flush_symbols(current_file)
            current_file = normalize_path(line.split(": ", 1)[1])
            events.append(Event("file_write", current_file, {"operation": "created", "source": "apply_patch"}, line, "patch"))
        elif line.startswith("*** Delete File: "):
            flush_symbols(current_file)
            current_file = normalize_path(line.split(": ", 1)[1])
            events.append(Event("file_delete", current_file, {"operation": "deleted", "source": "apply_patch"}, line, "patch"))
        elif line.startswith("*** Move to: "):
            dst = normalize_path(line.split(": ", 1)[1])
            if current_file:
                pending_rename_from = current_file
                events.append(Event("file_rename", dst, {"source": current_file, "operation": "renamed"}, line, "patch"))
                current_file = dst
        elif line.startswith("+") and not line.startswith("+++"):
            current_added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            current_removed.append(line[1:])

    flush_symbols(current_file)
    if not any(line.startswith("*** End Patch") for line in lines):
        warnings.append(
            {
                "step_id": step.step_id,
                "warning_type": "malformed_patch",
                "message": "Patch block appears incomplete.",
                "evidence": step.raw_text[:400],
            }
        )
    return events


def parse_diff_events(step: Step, warnings: list[dict[str, Any]]) -> list[Event]:
    lines = step.raw_text.splitlines()
    events: list[Event] = []
    current_file: str | None = None
    added: list[str] = []
    removed: list[str] = []

    def flush() -> None:
        if not current_file:
            return
        for symbol in extract_symbols_from_text("\n".join(added + removed)):
            events.append(Event("symbol_change", symbol, {"file": current_file}, compact_preview("\n".join(added + removed)), "diff"))
        added.clear()
        removed.clear()

    for line in lines:
        if line.startswith("diff --git "):
            flush()
            parts = line.split()
            if len(parts) >= 4:
                current_file = normalize_path(parts[3].removeprefix("b/"))
        elif line.startswith("+++ b/"):
            current_file = normalize_path(line[6:])
            events.append(Event("file_write", current_file, {"operation": "modified", "source": "diff"}, line, "diff"))
        elif line.startswith("--- a/"):
            pass
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])

    flush()
    if not any(line.startswith("diff --git") for line in lines):
        warnings.append(
            {
                "step_id": step.step_id,
                "warning_type": "ambiguous_parse",
                "message": "Diff parser was invoked but no diff header was found.",
                "evidence": step.raw_text[:400],
            }
        )
    return events


def parse_test_output_events(step: Step) -> list[Event]:
    text = step.output_text
    if not text:
        return []

    events: list[Event] = []
    inferred_status = infer_test_status(text)
    for match in FAILED_TEST_RE.finditer(text):
        nodeid = match.group(2)
        if "::" in nodeid or nodeid.endswith(".py"):
            attrs = {
                "status": "failed" if match.group(1) == "FAILED" else "error",
                "mentioned_files": collect_file_mentions_from_text(text),
            }
            events.append(Event("test_run", nodeid, attrs, match.group(0), "output"))

    if not events and inferred_status != "unknown" and step.detected_command_type == "test":
        events.append(Event("test_run", compact_preview(step.action_text), {"status": inferred_status}, compact_preview(text), "output"))
    return events


def parse_traceback_events(step: Step) -> list[Event]:
    text = step.output_text
    events: list[Event] = []
    for match in TRACEBACK_RE.finditer(text):
        path = normalize_path(match.group(1))
        line_number = int(match.group(2))
        if path:
            events.append(Event("file_read", path, {"line": line_number, "source": "traceback"}, match.group(0), "traceback"))
    return events


def parse_grep_output_events(step: Step) -> list[Event]:
    text = step.output_text
    events: list[Event] = []
    for line in text.splitlines():
        match = GREP_OUTPUT_RE.match(line)
        if not match:
            continue
        path = normalize_path(match.group("path"))
        if not looks_like_grep_path(path):
            continue
        attrs = {
            "line": int(match.group("line")),
            "snippet": match.group("snippet").strip(),
            "source": "grep_output",
        }
        if match.group("col"):
            attrs["col"] = int(match.group("col"))
        events.append(Event("file_read", path, attrs, line, "grep"))
    return events


def extract_pattern_flags(text: str, source_kind: str, step_id: int | None) -> list[Event]:
    lowered = text.lower()
    matches: list[tuple[str, str]] = []

    if any(token in lowered for token in ("curl ", "wget ", "urllib.request", "requests.", "http://", "https://")):
        matches.append(("network_call", compact_preview(text)))
    if any(token in lowered for token in ("subprocess", "os.system", "popen(", "exec(", "eval(", "bash ", "sh ")):
        matches.append(("process_exec_surface", compact_preview(text)))
    if any(token in lowered for token in ("base64", "b64decode", "base64.b64encode", "atob(", "fromhex(")):
        matches.append(("encoded_payload", compact_preview(text)))
    if any(token in lowered for token in ("chmod", "chown", "sudoers", "setcap", "setuid")):
        matches.append(("permission_change", compact_preview(text)))
    if any(token in lowered for token in ("systemd", "cron", "crontab", "/etc/init", "launchd", "daemon")):
        matches.append(("persistence_surface", compact_preview(text)))
    if any(token in lowered for token in (".github/workflows", ".gitlab-ci", "azure-pipelines", "jenkinsfile")):
        matches.append(("ci_config_edit", compact_preview(text)))
    if any(token in lowered for token in ("requirements.txt", "pyproject.toml", "package.json", "cargo.toml", "pip install", "npm install", "poetry add")):
        matches.append(("dependency_edit", compact_preview(text)))
    if re.search(r"(^|/)(tests?|__tests__)/|test_[^/\s]+\.py|\.spec\.(js|ts)$", lowered):
        matches.append(("test_file_edit", compact_preview(text)))
    if any(token in lowered for token in ("pytest.skip", "xfail", "@skip", "skipif")):
        matches.append(("test_skip_or_xfail", compact_preview(text)))
    if SECRET_HINT_RE.search(text):
        matches.append(("auth_or_secret_surface", compact_preview(text)))
    if re.search(r"(^|[/'\"\s])\.[A-Za-z0-9_.-]+", text):
        matches.append(("hidden_file", compact_preview(text)))
    if any(token in lowered for token in ("rm -rf", "delete", "truncate", "drop table", "shutil.rmtree")):
        matches.append(("destructive_action", compact_preview(text)))
    if any(token in lowered for token in ("curl ", "wget ", "git clone ", "pip install git+")):
        matches.append(("external_download", compact_preview(text)))
    if any(token in lowered for token in ("python -c", "bash -c", "sh -c", "eval(", "exec(", "tool call for tool python")):
        matches.append(("opaque_script_execution", compact_preview(text)))

    return [Event("pattern_flag", flag_name, {"source_kind": source_kind}, evidence, "heuristic") for flag_name, evidence in unique_pairs(matches)]


def extract_symbols_from_text(text: str) -> list[str]:
    symbols: list[str] = []
    for regex in (PY_DEF_RE, PY_CLASS_RE, JS_FUNC_RE, JS_CLASS_RE, JS_CONST_RE, GO_FUNC_RE, RUST_FN_RE, JAVA_METHOD_RE):
        symbols.extend(regex.findall(text))
    flattened: list[str] = []
    for symbol in symbols:
        if isinstance(symbol, tuple):
            flattened.extend(part for part in symbol if part)
        elif symbol:
            flattened.append(symbol)
    return unique_list(flattened)


def parse_imports_as_package_use(text: str) -> list[Event]:
    events: list[Event] = []
    for match in IMPORT_RE.finditer(text):
        package = (match.group(1) or match.group(2) or "").split(".", 1)[0]
        if package:
            events.append(Event("package_use", package, {"source": "visible_import"}, match.group(0), "heuristic"))
    return events


def capture_before_state(ledger: StateLedger, events: list[Event]) -> dict[str, Any]:
    files = sorted({event.target for event in events if event.type in {"file_read", "file_write", "file_delete", "file_rename", "file_chmod"}})
    tests = sorted({event.target for event in events if event.type == "test_run"})
    packages = sorted({event.target for event in events if event.type in {"package_install", "package_use"}})
    externals = sorted({event.target for event in events if event.type == "external_resource"})

    before_files: dict[str, Any] = {}
    for path in files:
        before_files[path] = {
            "seen_before": path in ledger.files_seen,
            "last_reader_step": ledger.last_reader_by_file.get(path),
            "last_writer_step": ledger.last_writer_by_file.get(path),
            "known_symbols": sorted(ledger.symbols_seen_by_file.get(path, set())),
        }

    before_tests: dict[str, Any] = {}
    for test in tests:
        before_tests[test] = {
            "seen_before": test in ledger.tests_seen,
            "last_run_step": ledger.last_run_step_by_test.get(test),
            "last_status": ledger.last_test_status_by_test.get(test),
        }

    return {
        "files": before_files,
        "tests": before_tests,
        "packages": {package: {"seen_before": package in ledger.packages_seen} for package in packages},
        "external_resources": {resource: {"seen_before": resource in ledger.external_resources_seen} for resource in externals},
    }


def update_state_ledger(ledger: StateLedger, step: Step, events: list[Event]) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "files": {},
        "tests": {},
        "packages": {},
        "external_resources": {},
    }

    for event in events:
        if event.type in {"file_read", "file_write", "file_delete", "file_rename", "file_chmod"}:
            path = event.target
            ledger.files_seen.add(path)
            file_update = updates["files"].setdefault(path, {})
            if event.type == "file_read":
                ledger.last_reader_by_file[path] = step.step_id
                file_update["last_reader_step"] = step.step_id
            if event.type == "file_write":
                ledger.last_writer_by_file[path] = step.step_id
                file_update["last_writer_step"] = step.step_id
            if event.type == "file_delete":
                ledger.last_deleter_by_file[path] = step.step_id
                file_update["last_deleter_step"] = step.step_id
            if event.type == "file_chmod":
                ledger.last_chmod_by_file[path] = step.step_id
                file_update["last_chmod_step"] = step.step_id
            symbols = event.attrs.get("symbols") or []
            if symbols:
                ledger.symbols_seen_by_file.setdefault(path, set()).update(symbols)
                file_update["new_symbols"] = sorted(symbols)
        elif event.type == "symbol_change":
            file_path = event.attrs.get("file")
            if file_path:
                ledger.files_seen.add(file_path)
                ledger.symbols_seen_by_file.setdefault(file_path, set()).add(event.target)
                file_update = updates["files"].setdefault(file_path, {})
                file_update.setdefault("new_symbols", [])
                if event.target not in file_update["new_symbols"]:
                    file_update["new_symbols"].append(event.target)
        elif event.type == "test_run":
            test = event.target
            ledger.tests_seen.add(test)
            ledger.last_run_step_by_test[test] = step.step_id
            status = str(event.attrs.get("status", "unknown"))
            ledger.last_test_status_by_test[test] = status
            updates["tests"][test] = {
                "last_run_step": step.step_id,
                "last_status": status,
            }
        elif event.type in {"package_install", "package_use"}:
            package = event.target
            ledger.packages_seen.add(package)
            updates["packages"][package] = {"last_seen_in_step": step.step_id, "event_type": event.type}
        elif event.type == "external_resource":
            resource = event.target
            ledger.external_resources_seen.add(resource)
            updates["external_resources"][resource] = {"last_seen_in_step": step.step_id}

    for file_update in updates["files"].values():
        if "new_symbols" in file_update:
            file_update["new_symbols"] = sorted(file_update["new_symbols"])

    return {key: value for key, value in updates.items() if value}


def build_graph(steps: list[Step], events_by_step: dict[int, list[Event]]) -> ProvenanceGraph:
    graph = ProvenanceGraph()
    for step in steps:
        graph.add_node(step_node_id(step.step_id), "Step", f"step {step.step_id}", {})

    for step_id, events in events_by_step.items():
        step_node = step_node_id(step_id)
        for event in events:
            if event.type == "file_read":
                file_node = file_node_id(event.target)
                graph.add_node(file_node, "File", event.target, {})
                graph.add_edge(step_node, file_node, "step_reads_file", edge_attrs(step_id, event))
                for symbol in event.attrs.get("symbols") or []:
                    symbol_node = symbol_node_id(event.target, symbol)
                    graph.add_node(symbol_node, "Symbol", symbol, {})
                    graph.add_edge(file_node, symbol_node, "file_contains_symbol", {"file": event.target, "symbol": symbol})
            elif event.type == "file_write":
                file_node = file_node_id(event.target)
                graph.add_node(file_node, "File", event.target, {})
                graph.add_edge(step_node, file_node, "step_writes_file", edge_attrs(step_id, event))
                for symbol in event.attrs.get("symbols") or []:
                    symbol_node = symbol_node_id(event.target, symbol)
                    graph.add_node(symbol_node, "Symbol", symbol, {})
                    graph.add_edge(file_node, symbol_node, "file_contains_symbol", {"file": event.target, "symbol": symbol})
            elif event.type == "file_delete":
                file_node = file_node_id(event.target)
                graph.add_node(file_node, "File", event.target, {})
                graph.add_edge(step_node, file_node, "step_deletes_file", edge_attrs(step_id, event))
            elif event.type == "file_rename":
                file_node = file_node_id(event.target)
                graph.add_node(file_node, "File", event.target, {})
                graph.add_edge(step_node, file_node, "step_renames_file", edge_attrs(step_id, event))
            elif event.type == "file_chmod":
                file_node = file_node_id(event.target)
                graph.add_node(file_node, "File", event.target, {})
                graph.add_edge(step_node, file_node, "step_chmods_file", edge_attrs(step_id, event))
            elif event.type == "test_run":
                test_node = test_node_id(event.target)
                graph.add_node(test_node, "Test", event.target, {})
                graph.add_edge(step_node, test_node, "step_runs_test", edge_attrs(step_id, event))
                for file_path in event.attrs.get("mentioned_files", []) or []:
                    file_node = file_node_id(file_path)
                    graph.add_node(file_node, "File", file_path, {})
                    graph.add_edge(test_node, file_node, "test_mentions_file", {"file": file_path, "step_id": step_id})
            elif event.type in {"package_install", "package_use"}:
                package_node = package_node_id(event.target)
                graph.add_node(package_node, "Package", event.target, {})
                edge_type = "step_installs_package" if event.type == "package_install" else "step_uses_package"
                graph.add_edge(step_node, package_node, edge_type, edge_attrs(step_id, event))
            elif event.type == "external_resource":
                resource_node = external_node_id(event.target)
                graph.add_node(resource_node, "ExternalResource", event.target, {})
                graph.add_edge(step_node, resource_node, "step_contacts_external_resource", edge_attrs(step_id, event))
            elif event.type == "symbol_change":
                file_path = event.attrs.get("file")
                symbol_node = symbol_node_id(file_path, event.target) if file_path else f"symbol:{event.target}"
                graph.add_node(symbol_node, "Symbol", event.target, {})
                graph.add_edge(step_node, symbol_node, "step_changes_symbol", edge_attrs(step_id, event))
                if file_path:
                    file_node = file_node_id(file_path)
                    graph.add_node(file_node, "File", file_path, {})
                    graph.add_edge(file_node, symbol_node, "file_contains_symbol", {"file": file_path, "symbol": event.target})
            elif event.type == "pattern_flag":
                flag_node = flag_node_id(event.target)
                graph.add_node(flag_node, "PatternFlag", event.target, {})
                graph.add_edge(step_node, flag_node, "step_has_pattern_flag", edge_attrs(step_id, event))
    return graph


def add_cross_step_provenance_edges(
    graph: ProvenanceGraph,
    step_summaries: list[dict[str, Any]],
    steps: list[Step],
    events_by_step: dict[int, list[Event]],
) -> None:
    summaries_by_id = {summary["step_id"]: summary for summary in step_summaries}
    last_writer_by_file: dict[str, int] = {}
    last_installer_by_package: dict[str, int] = {}

    for step in steps:
        events = events_by_step.get(step.step_id, [])
        reads = [event.target for event in events if event.type == "file_read"]
        test_runs = [event for event in events if event.type == "test_run"]
        package_uses = [event.target for event in events if event.type == "package_use"]
        mentioned_files: list[str] = []

        for event in test_runs:
            mentioned_files.extend(event.attrs.get("mentioned_files", []) or [])
            inferred_test_path = infer_test_file_from_target(event.target)
            if inferred_test_path:
                mentioned_files.append(inferred_test_path)

        dependencies: list[dict[str, Any]] = []

        for file_path in unique_list(reads + mentioned_files):
            prior_step = last_writer_by_file.get(file_path)
            if prior_step and prior_step != step.step_id:
                relation = {
                    "step_id": prior_step,
                    "via": file_path,
                    "relation": "prior_write_of_read_file",
                }
                dependencies.append(relation)
                summaries_by_id[prior_step]["provenance_relations"]["later_dependent_steps"].append(
                    {
                        "step_id": step.step_id,
                        "via": file_path,
                        "relation": "later_test_or_read",
                    }
                )
                graph.add_edge(
                    step_node_id(prior_step),
                    step_node_id(step.step_id),
                    "step_later_used_by_step",
                    {"via_file": file_path, "step_id": prior_step, "later_step_id": step.step_id},
                )
                graph.add_edge(
                    step_node_id(step.step_id),
                    step_node_id(prior_step),
                    "step_depends_on_step",
                    {"via_file": file_path, "step_id": step.step_id, "prior_step_id": prior_step},
                )

        for package in package_uses:
            prior_installer = last_installer_by_package.get(package)
            if prior_installer and prior_installer != step.step_id:
                relation = {
                    "step_id": prior_installer,
                    "via": package,
                    "relation": "prior_package_install",
                }
                dependencies.append(relation)
                summaries_by_id[prior_installer]["provenance_relations"]["later_dependent_steps"].append(
                    {
                        "step_id": step.step_id,
                        "via": package,
                        "relation": "later_package_use",
                    }
                )
                graph.add_edge(
                    step_node_id(prior_installer),
                    step_node_id(step.step_id),
                    "step_later_used_by_step",
                    {"via_package": package, "step_id": prior_installer, "later_step_id": step.step_id},
                )
                graph.add_edge(
                    step_node_id(step.step_id),
                    step_node_id(prior_installer),
                    "step_depends_on_step",
                    {"via_package": package, "step_id": step.step_id, "prior_step_id": prior_installer},
                )

        summaries_by_id[step.step_id]["provenance_relations"]["depends_on_steps"].extend(unique_relation_list(dependencies))

        for event in events:
            if event.type in {"file_write", "file_rename", "file_delete"}:
                last_writer_by_file[event.target] = step.step_id
            if event.type == "package_install":
                last_installer_by_package[event.target] = step.step_id


def build_step_scan_payloads(
    steps: list[Step],
    step_summaries: list[dict[str, Any]],
    *,
    context_window: int,
) -> list[StepScanPayload]:
    summaries_by_id = {int(summary["step_id"]): summary for summary in step_summaries}
    payloads: list[StepScanPayload] = []

    for index, step in enumerate(steps):
        context_steps = steps[max(0, index - context_window) : index]
        summary = summaries_by_id.get(step.step_id, {})
        action_preview = str(summary.get("action_preview") or compact_preview(step.action_text or step.raw_text))
        payloads.append(
            StepScanPayload(
                step_id=step.step_id,
                action_preview=action_preview,
                current_payload_text=render_scan_payload(
                    step,
                    summary,
                    context_steps=[],
                    output_char_limit=DEFAULT_SCAN_OUTPUT_CHARS,
                ),
                contextual_payload_text=render_scan_payload(
                    step,
                    summary,
                    context_steps=context_steps,
                    output_char_limit=DEFAULT_SCAN_OUTPUT_CHARS,
                    context_output_char_limit=DEFAULT_CONTEXT_OUTPUT_CHARS,
                ),
                context_step_ids=[ctx.step_id for ctx in context_steps],
            )
        )

    return payloads


def render_scan_payload(
    step: Step,
    summary: dict[str, Any],
    *,
    context_steps: list[Step],
    output_char_limit: int,
    context_output_char_limit: int = DEFAULT_CONTEXT_OUTPUT_CHARS,
) -> str:
    lines = [f"# Step Scan Payload S{step.step_id}", ""]
    lines.extend(render_scan_metadata_block(step, summary))
    lines.append("")
    lines.extend(render_step_body_block(step, output_char_limit=output_char_limit))

    if context_steps:
        for context_step in context_steps:
            lines.append("")
            lines.extend(render_context_step_block(context_step, output_char_limit=context_output_char_limit))

    return "\n".join(line for line in lines if line is not None).strip() + "\n"


def render_scan_metadata_block(step: Step, summary: dict[str, Any]) -> list[str]:
    before_state = summary.get("before_state") or {}
    return [
        "[STEP]",
        f"step_id: S{step.step_id}",
        f"tool_name: {step.tool_name or 'unknown'}",
        f"command_type: {step.detected_command_type or 'unknown'}",
        f"line_range: {step.line_start}-{step.line_end}",
        f"action_preview: {summary.get('action_preview') or compact_preview(step.action_text or step.raw_text)}",
        f"context_files: {summarize_before_state_targets(before_state.get('files', {}))}",
        f"context_packages: {summarize_before_state_targets(before_state.get('packages', {}))}",
        f"context_externals: {summarize_before_state_targets(before_state.get('external_resources', {}))}",
        "[/STEP]",
    ]


def render_step_body_block(step: Step, *, output_char_limit: int) -> list[str]:
    lines = ["[ACTION]", step.action_text.strip() or "(none)", "[/ACTION]"]

    for index, blob in enumerate(extract_step_embedded_blobs(step), start=1):
        lines.extend(
            [
                "",
                f"[EMBEDDED {index} | kind={blob['kind']} | label={blob['label']}]",
                blob["text"],
                "[/EMBEDDED]",
            ]
        )

    if step.output_text.strip():
        lines.extend(
            [
                "",
                "[OUTPUT]",
                truncate_for_scan(step.output_text, output_char_limit),
                "[/OUTPUT]",
            ]
        )

    event_lines = summarize_events_for_scan(step.events)
    if event_lines:
        lines.extend(["", "[EVENTS]", *event_lines, "[/EVENTS]"])

    return lines


def render_context_step_block(step: Step, *, output_char_limit: int) -> list[str]:
    lines = [f"[PRIOR_STEP S{step.step_id}]", f"preview: {compact_preview(step.action_text or step.raw_text)}"]
    if step.action_text.strip():
        lines.extend(["[PRIOR_ACTION]", step.action_text.strip(), "[/PRIOR_ACTION]"])

    for index, blob in enumerate(extract_step_embedded_blobs(step), start=1):
        lines.extend(
            [
                f"[PRIOR_EMBEDDED {index} | kind={blob['kind']} | label={blob['label']}]",
                blob["text"],
                "[/PRIOR_EMBEDDED]",
            ]
        )

    if step.output_text.strip():
        lines.extend(["[PRIOR_OUTPUT]", truncate_for_scan(step.output_text, output_char_limit), "[/PRIOR_OUTPUT]"])

    event_lines = summarize_events_for_scan(step.events)
    if event_lines:
        lines.extend(["[PRIOR_EVENTS]", *event_lines, "[/PRIOR_EVENTS]"])

    lines.append("[/PRIOR_STEP]")
    return lines


def extract_step_embedded_blobs(step: Step) -> list[dict[str, str]]:
    blobs: list[dict[str, str]] = []

    if step.tool_name == "text_editor":
        command = str(step.tool_args.get("command", "")).strip()
        path = normalize_path(str(step.tool_args.get("path", "")).strip())
        if command == "create":
            file_text = str(step.tool_args.get("file_text", "")).strip()
            if file_text:
                blobs.append({"kind": "file_text", "label": path or "create", "text": file_text})
        elif command == "str_replace":
            old_str = str(step.tool_args.get("old_str", "")).strip()
            new_str = str(step.tool_args.get("new_str", "")).strip()
            if old_str:
                blobs.append({"kind": "old_text", "label": path or "str_replace", "text": old_str})
            if new_str:
                blobs.append({"kind": "new_text", "label": path or "str_replace", "text": new_str})
        elif command == "insert":
            insert_text = str(step.tool_args.get("insert_text", "")).strip()
            if insert_text:
                blobs.append({"kind": "insert_text", "label": path or "insert", "text": insert_text})
    elif step.tool_name == "python":
        code = step.action_text.strip()
        if code:
            blobs.append({"kind": "python", "label": "tool_code", "text": code})
    elif "*** Begin Patch" in step.raw_text or "diff --git" in step.raw_text or step.detected_command_type == "diff":
        raw_text = step.raw_text.strip()
        if raw_text:
            blobs.append({"kind": "patch", "label": "patch_or_diff", "text": raw_text})

    if step.tool_name == "bash" or step.detected_command_type in {"shell", "grep", "test"}:
        for index, segment in enumerate(split_shell_command(step.action_text), start=1):
            segment = segment.strip()
            if not segment:
                continue
            heredoc_body = extract_heredoc_payload(segment).strip()
            if heredoc_body:
                blobs.append({"kind": "heredoc", "label": f"segment_{index}", "text": heredoc_body})
            inline_code = extract_inline_python_code(segment).strip()
            if inline_code:
                blobs.append({"kind": "inline_python", "label": f"segment_{index}", "text": inline_code})

    return dedupe_scan_blobs(blobs)


def dedupe_scan_blobs(blobs: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, str]] = []
    for blob in blobs:
        key = (blob["kind"], blob["label"], blob["text"])
        if key not in seen:
            seen.add(key)
            output.append(blob)
    return output


def summarize_events_for_scan(events: list[Event]) -> list[str]:
    lines: list[str] = []
    for event in events:
        suffix_parts: list[str] = []
        for key in ("operation", "source", "mode", "owner", "execution_role"):
            value = event.attrs.get(key)
            if value:
                suffix_parts.append(f"{key}={value}")
        if event.type == "pattern_flag":
            suffix_parts.append(f"source_kind={event.attrs.get('source_kind')}")
        suffix = f" [{', '.join(suffix_parts)}]" if suffix_parts else ""
        lines.append(f"{event.type}: {event.target}{suffix}")
    return lines


def summarize_before_state_targets(entries: dict[str, Any], *, limit: int = 4) -> str:
    if not entries:
        return "none"
    names = sorted(entries)[:limit]
    remaining = len(entries) - len(names)
    summary = ",".join(names)
    if remaining > 0:
        summary += f",+{remaining}"
    return summary


def truncate_for_scan(text: str, limit: int) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit] + "\n...[truncated for scan]..."


def persist_step_scan_payloads(payloads: list[StepScanPayload], payload_dir: Path) -> None:
    current_dir = payload_dir / "current"
    contextual_dir = payload_dir / "context"
    current_dir.mkdir(parents=True, exist_ok=True)
    contextual_dir.mkdir(parents=True, exist_ok=True)

    for payload in payloads:
        stem = f"step_{payload.step_id:04d}"
        current_path = current_dir / f"{stem}.txt"
        contextual_path = contextual_dir / f"{stem}.txt"
        current_path.write_text(payload.current_payload_text, encoding="utf-8")
        contextual_path.write_text(payload.contextual_payload_text, encoding="utf-8")
        payload.current_payload_path = current_path
        payload.contextual_payload_path = contextual_path


def run_step_scanners(
    payloads: list[StepScanPayload],
    yara_rules_path: Path,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    findings_by_step: dict[int, list[dict[str, Any]]] = {payload.step_id: [] for payload in payloads}
    current_paths = {payload.step_id: payload.current_payload_path for payload in payloads if payload.current_payload_path}
    contextual_paths = {
        payload.step_id: payload.contextual_payload_path for payload in payloads if payload.contextual_payload_path
    }

    scanner_jobs: list[tuple[str, Any]] = []
    gitleaks_exe = shutil.which("gitleaks")
    if gitleaks_exe and current_paths:
        scanner_jobs.append(("gitleaks", (gitleaks_exe, current_paths)))
    else:
        warnings.append(
            {
                "step_id": None,
                "warning_type": "scanner_unavailable",
                "message": "gitleaks was not found on PATH; skipping secret scanning.",
                "evidence": "gitleaks",
            }
        )

    yara_exe = shutil.which("yara64") or shutil.which("yara")
    if yara_exe and yara_rules_path.exists() and contextual_paths:
        scanner_jobs.append(("yara", (yara_exe, yara_rules_path, contextual_paths)))
    else:
        message = "yara was not found on PATH; skipping YARA scanning."
        evidence = "yara"
        if not yara_rules_path.exists():
            message = "YARA rules file was not found; skipping YARA scanning."
            evidence = str(yara_rules_path)
        warnings.append(
            {
                "step_id": None,
                "warning_type": "scanner_unavailable",
                "message": message,
                "evidence": evidence,
            }
        )

    def run_scanner(job: tuple[str, Any]) -> tuple[str, dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
        name, args = job
        local_warnings: list[dict[str, Any]] = []
        if name == "gitleaks":
            results = run_gitleaks_scan(args[0], args[1], local_warnings)
        else:
            results = run_yara_scan(args[0], args[1], args[2], local_warnings)
        return name, results, local_warnings

    if scanner_jobs:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(scanner_jobs), thread_name_prefix="step-scanner") as executor:
            scanner_results = list(executor.map(run_scanner, scanner_jobs))
        for _, results, scanner_warnings in sorted(scanner_results, key=lambda item: item[0]):
            merge_scanner_findings(findings_by_step, results)
            warnings.extend(scanner_warnings)

    records: list[dict[str, Any]] = []
    for payload in payloads:
        step_findings = dedupe_findings(findings_by_step.get(payload.step_id, []))
        records.append(
            {
                "step_id": payload.step_id,
                "action_preview": payload.action_preview,
                "scanner_labels": [f"{finding['tool']}:{finding['label']}" for finding in step_findings],
                "findings": step_findings,
            }
        )
    return records


def merge_scanner_findings(
    findings_by_step: dict[int, list[dict[str, Any]]],
    scanner_results: dict[int, list[dict[str, Any]]],
) -> None:
    for step_id, findings in scanner_results.items():
        findings_by_step.setdefault(step_id, []).extend(findings)


def run_gitleaks_scan(
    executable: str,
    current_payload_paths: dict[int, Path],
    warnings: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="gitleaks-scan-") as scan_root_value:
        scan_root = Path(scan_root_value)
        staged_dir = scan_root / "payloads"
        staged_dir.mkdir()
        staged_paths: dict[int, Path] = {}
        for step_id, source_path in current_payload_paths.items():
            staged_path = staged_dir / source_path.name
            shutil.copy2(source_path, staged_path)
            staged_paths[step_id] = staged_path

        # Keep the live report outside the target so it can never become scan input.
        report_path = scan_root / "gitleaks_report.json"
        commands = [
            [
                executable,
                "detect",
                "--no-git",
                "--source",
                str(staged_dir),
                "--report-format",
                "json",
                "--report-path",
                str(report_path),
                "--no-banner",
                "--redact",
                "100",
            ],
            [
                executable,
                "dir",
                str(staged_dir),
                "--report-format",
                "json",
                "--report-path",
                str(report_path),
                "--no-banner",
                "--redact",
                "100",
            ],
        ]
        completed: subprocess.CompletedProcess[str] | None = None
        last_error = "gitleaks invocation failed"

        for command in commands:
            if report_path.exists():
                report_path.unlink()
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=SCANNER_TIMEOUT_SECONDS,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                last_error = f"gitleaks timed out after {SCANNER_TIMEOUT_SECONDS}s for command: {' '.join(command[:3])}"
                continue
            last_error = (completed.stderr or completed.stdout or "").strip()[:4000]
            if completed.returncode in {0, 1} and report_path.exists():
                break
        else:
            warnings.append(
                {
                    "step_id": None,
                    "warning_type": "scanner_failed",
                    "message": "gitleaks could not scan the payload directory with the supported command variants.",
                    "evidence": last_error,
                }
            )
            return {}

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(
                {
                    "step_id": None,
                    "warning_type": "scanner_failed",
                    "message": "Could not parse gitleaks JSON report.",
                    "evidence": str(exc),
                }
            )
            return {}

        findings_by_step: dict[int, list[dict[str, Any]]] = {}
        path_to_step = build_path_to_step_lookup(staged_paths)
        for finding in normalize_gitleaks_report(report):
            file_value = str(finding.get("File") or finding.get("file") or "")
            step_id = lookup_step_id_from_path(path_to_step, file_value)
            if step_id is None:
                continue
            label, reason_key, reason_text = map_gitleaks_finding(finding)
            findings_by_step.setdefault(step_id, []).append(
                {
                    "tool": "gitleaks",
                    "label": label,
                    "reason_key": reason_key,
                    "reason_text": reason_text,
                    "rule": str(finding.get("RuleID") or finding.get("Description") or "gitleaks"),
                }
            )
        return findings_by_step


def build_path_to_step_lookup(paths_by_step: dict[int, Path]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for step_id, path in paths_by_step.items():
        resolved = str(path.resolve())
        lookup[resolved] = step_id
        lookup[str(path)] = step_id
        lookup[path.name] = step_id
    return lookup


def lookup_step_id_from_path(path_to_step: dict[str, int], raw_path: str) -> int | None:
    if raw_path in path_to_step:
        return path_to_step[raw_path]
    try:
        resolved = str(Path(raw_path).resolve())
    except OSError:
        resolved = raw_path
    if resolved in path_to_step:
        return path_to_step[resolved]
    return path_to_step.get(Path(raw_path).name)


def normalize_gitleaks_report(report: Any) -> list[dict[str, Any]]:
    if isinstance(report, list):
        return [entry for entry in report if isinstance(entry, dict)]
    if isinstance(report, dict):
        for key in ("findings", "Findings", "leaks", "Leaks"):
            value = report.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
    return []


def map_gitleaks_finding(finding: dict[str, Any]) -> tuple[str, str, str]:
    blob = " ".join(
        [
            str(finding.get("RuleID") or ""),
            str(finding.get("Description") or ""),
            str(finding.get("Secret") or ""),
            str(finding.get("Match") or ""),
            " ".join(str(tag) for tag in finding.get("Tags") or []),
        ]
    ).lower()

    if any(token in blob for token in ("private key", "openssh", "rsa private", "ec private", "pgp private")):
        return ("private_key_material", "private_key_material", "private key material is present")
    if any(token in blob for token in ("jwt", "bearer", "authorization")):
        return ("auth_token_material", "auth_token_material", "authentication token material is present")
    if any(token in blob for token in ("password", "passwd", "pwd=", "db_pass")):
        return ("password_material", "password_material", "password-like credential material is present")
    if any(token in blob for token in ("api key", "apikey", "token", "secret", "stripe", "aws")):
        return ("service_api_token", "service_api_token", "service credential material is present")
    return ("secret_candidate", "secret_candidate", "credential-like secret material is present")


def run_yara_scan(
    executable: str,
    yara_rules_path: Path,
    contextual_payload_paths: dict[int, Path],
    warnings: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    rule_metadata = parse_yara_rule_metadata(yara_rules_path)
    findings_by_step: dict[int, list[dict[str, Any]]] = {}

    for step_id, payload_path in contextual_payload_paths.items():
        completed = subprocess.run(
            [executable, str(yara_rules_path), str(payload_path)],
            capture_output=True,
            text=True,
            timeout=SCANNER_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            warnings.append(
                {
                    "step_id": step_id,
                    "warning_type": "scanner_failed",
                    "message": "yara returned an error while scanning a step payload.",
                    "evidence": (completed.stderr.strip() or f"yara exited with status {completed.returncode}")[:4000],
                }
            )
            continue

        rule_names = unique_list(
            line.split(maxsplit=1)[0].strip()
            for line in completed.stdout.splitlines()
            if line.strip()
        )
        for rule_name in rule_names:
            metadata = rule_metadata.get(rule_name, {})
            label = str(metadata.get("public_label") or rule_name.lower())
            reason_key = str(metadata.get("reason_key") or label)
            reason_text = str(metadata.get("reason_text") or label.replace("_", " "))
            findings_by_step.setdefault(step_id, []).append(
                {
                    "tool": "yara",
                    "label": label,
                    "reason_key": reason_key,
                    "reason_text": reason_text,
                    "rule": rule_name,
                }
            )

    return findings_by_step


def parse_yara_rule_metadata(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    metadata_by_rule: dict[str, dict[str, str]] = {}

    for match in re.finditer(r"rule\s+([A-Za-z0-9_]+)\s*\{(.*?)\n\}", text, re.DOTALL):
        rule_name = match.group(1)
        body = match.group(2)
        meta_block = re.search(r"\bmeta:\s*(.*?)(?:\n\s*strings:|\n\s*condition:)", body, re.DOTALL)
        if not meta_block:
            continue
        metadata: dict[str, str] = {}
        for meta_match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\"([^\"]*)\"", meta_block.group(1)):
            metadata[meta_match.group(1)] = meta_match.group(2)
        metadata_by_rule[rule_name] = metadata

    return metadata_by_rule


def dedupe_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for finding in findings:
        key = (
            str(finding.get("tool")),
            str(finding.get("label")),
            str(finding.get("reason_key")),
            str(finding.get("rule")),
        )
        if key not in seen:
            seen.add(key)
            output.append(finding)
    return sorted(output, key=lambda item: (str(item.get("tool")), str(item.get("label")), str(item.get("rule"))))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def edge_attrs(step_id: int, event: Event) -> dict[str, Any]:
    attrs = {"step_id": step_id, "evidence": event.evidence, "source_kind": event.source_kind}
    attrs.update(event.attrs)
    return attrs


def parse_package_command(cmd: list[str], evidence: str) -> list[Event]:
    if not cmd:
        return []
    first = cmd[0]
    events: list[Event] = []

    install_specs = {
        ("pip", "install"),
        ("pip3", "install"),
        ("npm", "install"),
        ("npm", "add"),
        ("yarn", "add"),
        ("poetry", "add"),
        ("conda", "install"),
        ("apt", "install"),
        ("apt-get", "install"),
    }
    uninstall_specs = {
        ("pip", "uninstall"),
        ("pip3", "uninstall"),
    }

    if len(cmd) >= 2 and (first, cmd[1]) in install_specs:
        for token in cmd[2:]:
            if token.startswith("-"):
                continue
            events.append(Event("package_install", token, {}, evidence, "command"))
    elif len(cmd) >= 2 and (first, cmd[1]) in uninstall_specs:
        for token in cmd[2:]:
            if token.startswith("-"):
                continue
            events.append(Event("package_install", token, {"operation": "uninstall"}, evidence, "command"))
    return events


def parse_external_resources(text: str, source_kind: str) -> list[Event]:
    resources = unique_list(URL_RE.findall(text) + IP_RE.findall(text))
    return [Event("external_resource", resource, {}, compact_preview(text), source_kind) for resource in resources]


def command_executes_network_contact(cmd: list[str], segment: str) -> bool:
    if not cmd:
        return False
    first = cmd[0].rsplit("/", 1)[-1].lower()
    if first in {"curl", "wget", "scp", "sftp", "ssh", "nc", "ncat", "telnet", "ping"}:
        return True
    if first in {"python", "python3"} and "-c" in cmd:
        lowered = segment.lower()
        return bool(
            re.search(r"\b(?:requests\.|urllib\.|httpx\.|aiohttp\.|socket\.)", lowered)
            or re.search(r"\b(?:curl|wget|scp|ssh|nc)\b", lowered)
        )
    return False


def infer_test_status(output_text: str) -> str:
    lowered = output_text.lower()
    if "failed" in lowered or "error" in lowered:
        return "failed"
    if "passed" in lowered:
        return "passed"
    if "skipped" in lowered:
        return "skipped"
    return "unknown"


def collect_file_mentions_from_text(text: str) -> list[str]:
    paths = [normalize_path(match.group(1)) for match in TRACEBACK_RE.finditer(text)]
    for line in text.splitlines():
        grep_match = GREP_OUTPUT_RE.match(line)
        if grep_match:
            paths.append(normalize_path(grep_match.group("path")))
        failed_match = re.match(r"^(FAILED|ERROR)\s+([^\s:]+)", line)
        if failed_match:
            candidate = failed_match.group(2).split("::", 1)[0]
            if is_path_like(candidate):
                paths.append(normalize_path(candidate))
    return unique_list(path for path in paths if path)


def infer_test_file_from_target(target: str) -> str | None:
    if "::" in target:
        maybe_path = target.split("::", 1)[0]
        if is_path_like(maybe_path):
            return normalize_path(maybe_path)
    if target.endswith(".py") or target.endswith(".js") or target.endswith(".ts"):
        return normalize_path(target)
    return None


def first_non_option_path(tokens: list[str]) -> str | None:
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in {"-m", "-c"}:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        normalized = normalize_path(token)
        if is_path_like(normalized) or looks_dynamic_shell_path(normalized):
            return normalized
    return None


def is_test_command(tokens: list[str]) -> bool:
    joined = " ".join(tokens).lower()
    return any(joined.startswith(prefix) for prefix in TEST_COMMAND_PREFIXES)


def extract_test_targets_from_command(tokens: list[str]) -> list[str]:
    targets: list[str] = []
    skip_next = False
    for token in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in {"-k", "-m", "--maxfail"}:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        if "::" in token or token.endswith(".py") or token.endswith(".js") or token.endswith(".ts"):
            targets.append(token)
    return unique_list(targets)


def normalize_path(raw: str) -> str:
    path = raw.strip().strip("`").strip("\"'")
    path = path.rstrip(",;")
    if path.startswith("a/") or path.startswith("b/"):
        if "/" in path[2:]:
            path = path[2:]
    path = path.strip("()[]{}")
    if path.endswith(":") and "/" in path:
        path = path[:-1]
    return path


def is_path_like(value: str) -> bool:
    if not value:
        return False
    if value.startswith("attachment://"):
        return False
    if " " in value or "\t" in value:
        return False
    if ">" in value or "<" in value:
        return False
    if value in {"true", "false", "null"}:
        return False
    return bool(PATH_TOKEN_RE.fullmatch(value) or "/" in value or value.startswith("."))


def looks_like_grep_path(value: str) -> bool:
    if not is_path_like(value):
        return False
    return "/" in value or "\\" in value or "." in Path(value).name


def regex_shell_split(segment: str) -> list[str]:
    return re.findall(r"'[^']*'|\"[^\"]*\"|\S+", segment)


def parse_shell_tokens(segment_head: str, full_segment: str) -> list[str] | None:
    stripped = segment_head.strip()
    full_stripped = full_segment.strip()
    for prefix in ('python3 -c "', "python3 -c '", 'python -c "', "python -c '"):
        if stripped.startswith(prefix):
            interpreter = prefix.split()[0]
            inline_code = extract_inline_python_code(full_segment)
            return [interpreter, "-c", inline_code]
    su_match = re.match(r"^su\s+-\s+([A-Za-z0-9_.-]+)\s+-c\s+(['\"])(.*)\2$", full_stripped, re.DOTALL)
    if su_match:
        return ["su", "-", su_match.group(1), "-c", su_match.group(3)]
    return None


def split_shell_command(command_text: str) -> list[str]:
    segments: list[str] = []
    lines = collapse_shell_line_continuations(command_text)
    heredoc_marker: str | None = None
    current: list[str] = []
    quoted_block: list[str] = []
    open_quote: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if heredoc_marker is not None:
            current.append(line)
            if stripped == heredoc_marker:
                segments.append("\n".join(current).strip())
                current = []
                heredoc_marker = None
            continue

        if open_quote is not None:
            quoted_block.append(line)
            open_quote = scan_quote_state(line, open_quote)
            if open_quote is None:
                segments.append("\n".join(quoted_block).strip())
                quoted_block = []
            continue

        if not stripped or stripped.startswith("#"):
            continue

        heredoc_match = re.search(r"<<\s*['\"]?([A-Za-z0-9_]+)['\"]?", line)
        if heredoc_match:
            current = [line]
            heredoc_marker = heredoc_match.group(1)
            continue

        quote_state = scan_quote_state(line)
        if quote_state is not None:
            split_pieces = [piece.strip() for piece in split_shell_operators(line) if piece.strip()]
            if split_pieces:
                segments.extend(split_pieces[:-1])
                quoted_block = [split_pieces[-1]]
                open_quote = scan_quote_state(split_pieces[-1])
            else:
                quoted_block = [line]
                open_quote = quote_state
            continue

        split_pieces = [piece.strip() for piece in split_shell_operators(line) if piece.strip()]
        if split_pieces:
            segments.extend(split_pieces)

    if current:
        segments.append("\n".join(current).strip())
    if quoted_block:
        segments.append("\n".join(quoted_block).strip())

    return segments


def collapse_shell_line_continuations(command_text: str) -> list[str]:
    logical_lines: list[str] = []
    pending = ""

    for raw_line in command_text.splitlines():
        line = raw_line.rstrip()
        if pending:
            pending = f"{pending} {line.lstrip()}"
        else:
            pending = line

        if has_trailing_line_continuation(pending):
            pending = pending.rstrip()
            pending = pending[:-1].rstrip()
            continue

        logical_lines.append(pending)
        pending = ""

    if pending:
        logical_lines.append(pending)

    return logical_lines


def has_trailing_line_continuation(line: str) -> bool:
    stripped = line.rstrip()
    if not stripped.endswith("\\"):
        return False
    match = re.search(r"(\\+)$", stripped)
    if not match or len(match.group(1)) % 2 == 0:
        return False
    return scan_quote_state(stripped[:-1]) is None


def update_shell_path_bindings(segment_head: str, bindings: dict[str, list[str]]) -> None:
    for_match = re.match(r"^for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.+?)(?:;?\s*do)?$", segment_head.strip())
    if for_match:
        var_name = for_match.group(1)
        try:
            tokens = shlex.split(for_match.group(2), posix=True)
        except ValueError:
            tokens = regex_shell_split(for_match.group(2))
        paths: list[str] = []
        for token in tokens:
            paths.extend(expand_shell_path_token(token, bindings, allow_bindings=False))
        if paths:
            bindings[var_name] = unique_list(paths)
        return

    assign_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+)$", segment_head.strip())
    if not assign_match:
        return
    var_name = assign_match.group(1)
    raw_value = normalize_path(assign_match.group(2))
    if is_path_like(raw_value) or looks_dynamic_shell_path(raw_value):
        bindings[var_name] = [raw_value]


def expand_shell_path_token(
    token: str,
    bindings: dict[str, list[str]],
    *,
    allow_bindings: bool = True,
) -> list[str]:
    path = normalize_path(token)
    if not path:
        return []

    if allow_bindings:
        var_match = re.fullmatch(r"\$(\w+)|\$\{([^}]+)\}", path)
        if var_match:
            var_name = var_match.group(1) or var_match.group(2)
            return list(bindings.get(var_name, []))

    if is_path_like(path) or looks_dynamic_shell_path(path):
        return [path]
    return []


def expand_dynamic_event_targets(events: list[Event], bindings: dict[str, list[str]]) -> list[Event]:
    expanded: list[Event] = []
    for event in events:
        targets = expand_shell_path_token(event.target, bindings)
        if len(targets) <= 1:
            if targets:
                event.target = targets[0]
            expanded.append(event)
            continue
        for target in targets:
            expanded.append(Event(event.type, target, dict(event.attrs), event.evidence, event.source_kind))
    return expanded


def looks_dynamic_shell_path(value: str) -> bool:
    if "$" not in value:
        return False
    if re.fullmatch(r"\$(\w+)|\$\{[^}]+\}", value):
        return True
    return "/" in value or "\\" in value or "." in Path(value.replace("$", "x")).name


def extract_grep_like_targets(cmd: list[str]) -> list[str]:
    positional = [token for token in cmd[1:] if token and not token.startswith("-")]
    if len(positional) <= 1:
        return []
    return positional[1:]


def extract_http_output_paths(cmd: list[str]) -> list[str]:
    paths: list[str] = []
    for index, token in enumerate(cmd):
        if token == "-o" and index + 1 < len(cmd):
            paths.append(normalize_path(cmd[index + 1]))
        elif token.startswith("-o") and token != "-o":
            paths.append(normalize_path(token[2:]))
        elif token == "-O":
            for candidate in reversed(cmd[index + 1 :]):
                if URL_RE.search(candidate):
                    paths.append(normalize_path(candidate.rstrip("/").rsplit("/", 1)[-1]))
                    break
    return [path for path in unique_list(paths) if is_path_like(path) or looks_dynamic_shell_path(path)]


def skip_env_assignments(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", tokens[index]):
        index += 1
    return tokens[index:]


def strip_line_numbers(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        cleaned_lines.append(re.sub(r"^\s*\d+\s*[|:]\s?", "", line))
    return "\n".join(cleaned_lines)


def compact_preview(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def unique_list(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def unique_pairs(values: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    output: list[tuple[str, str]] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def unique_relation_list(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any]] = set()
    output: list[dict[str, Any]] = []
    for value in values:
        key = (value.get("step_id"), value.get("via"), value.get("relation"))
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def split_shell_operators(line: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escape = False
    i = 0

    while i < len(line):
        ch = line[i]
        if escape:
            current.append(ch)
            escape = False
            i += 1
            continue

        if ch == "\\" and quote != "'":
            current.append(ch)
            escape = True
            i += 1
            continue

        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue

        if ch in {"'", '"'}:
            quote = ch
            current.append(ch)
            i += 1
            continue

        two = line[i : i + 2]
        if two in {"&&", "||"}:
            piece = "".join(current).strip()
            if piece:
                pieces.append(piece)
            current = []
            i += 2
            continue
        if ch in {"|", ";"}:
            piece = "".join(current).strip()
            if piece:
                pieces.append(piece)
            current = []
            i += 1
            continue

        current.append(ch)
        i += 1

    piece = "".join(current).strip()
    if piece:
        pieces.append(piece)
    return pieces


def find_shell_redirections(segment: str) -> list[tuple[str, str]]:
    redirections: list[tuple[str, str]] = []
    quote: str | None = None
    escape = False
    command_subst_depth = 0
    i = 0

    while i < len(segment):
        ch = segment[i]
        if escape:
            escape = False
            i += 1
            continue

        if ch == "\\" and quote != "'":
            escape = True
            i += 1
            continue

        if quote:
            if ch == quote:
                quote = None
            i += 1
            continue

        if segment[i : i + 2] == "$(":
            command_subst_depth += 1
            i += 2
            continue

        if command_subst_depth:
            if ch == "(":
                command_subst_depth += 1
            elif ch == ")":
                command_subst_depth -= 1
            i += 1
            continue

        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue

        if ch in {"<", ">"}:
            if ch == "<" and i + 1 < len(segment) and segment[i + 1] == "<":
                i += 2
                while i < len(segment) and segment[i].isspace():
                    i += 1
                if i < len(segment) and segment[i] in {"'", '"'}:
                    quote_char = segment[i]
                    i += 1
                    while i < len(segment) and segment[i] != quote_char:
                        i += 1
                    if i < len(segment):
                        i += 1
                else:
                    while i < len(segment) and not segment[i].isspace():
                        i += 1
                continue

            operator = ch
            if ch == ">" and i + 1 < len(segment) and segment[i + 1] == ">":
                operator = ">>"
                i += 1

            i += 1
            while i < len(segment) and segment[i].isspace():
                i += 1

            if i < len(segment) and segment[i] == "&":
                while i < len(segment) and not segment[i].isspace():
                    i += 1
                continue

            start = i
            while i < len(segment) and not segment[i].isspace() and segment[i] not in {"|", ";", "&", "<", ">"}:
                i += 1
            target = segment[start:i]
            if target:
                redirections.append((operator, target))
            continue

        i += 1

    return redirections


def scan_quote_state(text: str, initial_quote: str | None = None) -> str | None:
    quote = initial_quote
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and quote != "'":
            escape = True
            continue
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch

    return quote


def extract_heredoc_payload(segment: str) -> str:
    if "<<" not in segment:
        return ""
    lines = segment.splitlines()
    if len(lines) < 3:
        return ""
    return "\n".join(lines[1:-1]).strip()


def extract_inline_python_code(segment: str) -> str:
    marker_index = segment.find("-c")
    if marker_index == -1:
        return ""
    remainder = segment[marker_index + 2 :].lstrip()
    if not remainder:
        return ""
    quote = remainder[0]
    if quote not in {"'", '"'}:
        return ""
    end_index = remainder.rfind(quote)
    if end_index <= 0:
        return ""
    return remainder[1:end_index]


def dedupe_events(events: list[Event]) -> list[Event]:
    seen: set[tuple[Any, ...]] = set()
    output: list[Event] = []
    for event in events:
        key = (
            event.type,
            event.target,
            json.dumps(event.attrs, sort_keys=True, ensure_ascii=True),
            event.evidence,
            event.source_kind,
        )
        if key not in seen:
            seen.add(key)
            output.append(event)
    return output


def stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        text = value.get("text") or value.get("content")
        if isinstance(text, str):
            return text
    return str(value)


def step_node_id(step_id: int) -> str:
    return f"step:{step_id}"


def file_node_id(path: str) -> str:
    return f"file:{path}"


def symbol_node_id(path: str | None, symbol: str) -> str:
    if path:
        return f"symbol:{path}::{symbol}"
    return f"symbol::{symbol}"


def test_node_id(name: str) -> str:
    return f"test:{name}"


def package_node_id(name: str) -> str:
    return f"package:{name}"


def external_node_id(name: str) -> str:
    return f"external:{name}"


def flag_node_id(name: str) -> str:
    return f"flag:{name}"


if __name__ == "__main__":
    main()
