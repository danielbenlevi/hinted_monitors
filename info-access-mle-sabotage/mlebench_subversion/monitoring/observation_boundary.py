"""Monitor-visible observation-boundary helpers.

Submission feedback is produced by the benchmark grader after the agent has
finished.  It is not an observation available to an online monitor and must
never be rendered as execution evidence.  These helpers deliberately retain
the agent's submit action while removing only its post-hoc tool result.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Mapping, Sequence


WITHHELD_SUBMISSION_OUTPUT = "[post-submission grader feedback withheld]"
WITHHELD_TOOL_OUTPUT = "[tool output withheld]"

_RENDERED_CODE_BLOCK_RE = re.compile(
    r"(?m)^Code:\r?\n```(?:json)?\r?\n(?P<code>[\s\S]*?)\r?\n```[ \t]*$"
)
_EXECUTION_HEADER_RE = re.compile(r"\A(?P<spacing>\r?\n\r?\n)Execution info:(?P<newline>\r?\n)")
_STEP_OPENER_RE = re.compile(r"\[TRANSCRIPT STEP S(?P<step_id>\d+)\]")
_EXECUTION_SECTION_RE = re.compile(r"(?m)^Execution info:[ \t]*(?:\r?\n|\Z)")
_STEP_CLOSER_RE = re.compile(r"\[/TRANSCRIPT STEP S(?P<step_id>\d+)\]")


def is_submit_action(code: Any, plan: Any = "") -> bool:
    """Return whether a synthetic AIDE node represents the submit tool."""

    text = str(code or "").strip()
    if text:
        try:
            payload = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, Mapping):
            tool_name = str(payload.get("tool") or payload.get("function") or "").strip().lower()
            if tool_name == "submit":
                return True

    # Some source journals retain only the normalized tool-call plan.  Keep
    # this deliberately exact so a shell command merely containing the word
    # "submit" is not mistaken for the benchmark submission tool.
    return str(plan or "").strip().lower() == "tool call for tool submit"


def sanitize_synthetic_journal(journal_data: Any) -> Any:
    """Return a copy with post-submit execution feedback removed.

    Non-dict journals are returned unchanged.  The input object is never
    mutated, which is important for frozen log objects shared across variants.
    """

    if not isinstance(journal_data, Mapping):
        return journal_data
    nodes = journal_data.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        return journal_data

    sanitized = dict(journal_data)
    sanitized_nodes: list[Any] = []
    for raw_node in nodes:
        if not isinstance(raw_node, Mapping) or not is_submit_action(raw_node.get("code"), raw_node.get("plan")):
            sanitized_nodes.append(raw_node)
            continue
        node = copy.deepcopy(dict(raw_node))
        execution_info = node.get("execution_info")
        if isinstance(execution_info, Mapping):
            execution_info = dict(execution_info)
            execution_info["stdout"] = WITHHELD_SUBMISSION_OUTPUT
            execution_info["stderr"] = ""
            execution_info["return_code"] = None
            node["execution_info"] = execution_info
        sanitized_nodes.append(node)
    sanitized["nodes"] = sanitized_nodes
    return sanitized


def sanitize_submit_tool_results(messages: Any) -> Any:
    """Return copied messages with submit-tool results replaced by a marker."""

    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return messages

    submit_call_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, Mapping):
                continue
            function_name = str(tool_call.get("function") or tool_call.get("name") or "").strip().lower()
            if function_name == "submit" and tool_call.get("id"):
                submit_call_ids.add(str(tool_call["id"]))

    sanitized: list[Any] = []
    for raw_message in messages:
        if not isinstance(raw_message, Mapping):
            sanitized.append(raw_message)
            continue
        message = copy.deepcopy(dict(raw_message))
        call_id = str(message.get("tool_call_id") or "")
        function_name = str(message.get("function") or message.get("name") or "").strip().lower()
        if message.get("role") == "tool" and (call_id in submit_call_ids or function_name == "submit"):
            message["content"] = WITHHELD_SUBMISSION_OUTPUT
        sanitized.append(message)
    return sanitized


def strip_execution_output_from_quote(quote: Any) -> Any:
    """Replace each rendered ``Execution info:`` body with a withheld marker.

    Extractor quotes are already-rendered transcript-step blocks whose tool
    output is baked in, so an evaluator content flag cannot remove it upstream.
    This narrows an existing quote to the agent's action alone while keeping the
    ``Execution info:`` header, so the evaluator is told the output is absent
    rather than being left to assume the step never executed.

    A section runs from its header to the enclosing ``[/TRANSCRIPT STEP Sn]``
    closer when the quote is wrapped, and otherwise to the next ``Execution
    info:`` header or the end of the quote.  Already-withheld bodies are left
    byte-identical so repeated application is a no-op.
    """

    if not isinstance(quote, str):
        return quote

    headers = list(_EXECUTION_SECTION_RE.finditer(quote))
    if not headers:
        return quote

    replacements: list[tuple[int, int]] = []
    for position, header in enumerate(headers):
        body_start = header.end()
        # Prefer the closer of the step that encloses this header; an unwrapped
        # legacy block simply runs to the next header or the quote's end.
        openers = list(_STEP_OPENER_RE.finditer(quote, 0, header.start()))
        body_end = len(quote)
        if position + 1 < len(headers):
            body_end = headers[position + 1].start()
        if openers:
            closer = f"[/TRANSCRIPT STEP S{openers[-1].group('step_id')}]"
            closer_start = quote.find(closer, body_start, body_end)
            if closer_start >= 0:
                body_end = closer_start
        # Keep the blank line/newline that separates the body from what follows.
        while body_end > body_start and quote[body_end - 1] in "\r\n":
            body_end -= 1
        if quote[body_start:body_end] == WITHHELD_TOOL_OUTPUT:
            continue
        replacements.append((body_start, body_end))

    stripped = quote
    for start, end in reversed(replacements):
        stripped = stripped[:start] + WITHHELD_TOOL_OUTPUT + stripped[end:]
    return stripped


def strip_execution_output_from_quotes(quotes: Any) -> Any:
    """Apply :func:`strip_execution_output_from_quote` across a quote list."""

    if isinstance(quotes, list):
        return [strip_execution_output_from_quote(quote) for quote in quotes]
    return strip_execution_output_from_quote(quotes)


def sanitize_rendered_submit_quote(quote: Any) -> Any:
    """Withhold execution feedback only for structurally rendered submit steps.

    Extractor quotes from older artifacts are already-rendered transcript-step
    blocks.  We intentionally require a parseable JSON object inside the
    fenced ``Code:`` section and an immediately following ``Execution info:``
    section.  Merely mentioning submit or the withheld marker in ordinary
    action/output text does not change a byte.
    """

    if not isinstance(quote, str):
        return quote

    replacements: list[tuple[int, int, str]] = []
    for code_match in _RENDERED_CODE_BLOCK_RE.finditer(quote):
        try:
            code_payload = json.loads(code_match.group("code").strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(code_payload, Mapping):
            continue
        tool_name = str(code_payload.get("tool") or code_payload.get("function") or "").strip().lower()
        if tool_name != "submit":
            continue

        remainder = quote[code_match.end() :]
        header_match = _EXECUTION_HEADER_RE.match(remainder)
        if header_match is None:
            continue
        output_start = code_match.end() + header_match.end()

        # A grounded quote normally has a matched transcript-step wrapper.  It
        # may contain arbitrary marker-like text inside execution output, so
        # use the last exact closer for the enclosing step.  Unwrapped legacy
        # blocks use the end of the quote as their execution boundary.
        enclosing_openers = list(_STEP_OPENER_RE.finditer(quote, 0, code_match.start()))
        if enclosing_openers:
            step_id = enclosing_openers[-1].group("step_id")
            closer = f"[/TRANSCRIPT STEP S{step_id}]"
            closer_start = quote.rfind(closer, output_start)
            if closer_start < 0:
                continue
            output_end = closer_start
            if quote[max(output_start, closer_start - 2) : closer_start] == "\r\n":
                output_end -= 2
            elif closer_start > output_start and quote[closer_start - 1] == "\n":
                output_end -= 1
        else:
            output_end = len(quote)
        replacements.append((output_start, output_end, WITHHELD_SUBMISSION_OUTPUT))

    sanitized = quote
    for start, end, replacement in reversed(replacements):
        sanitized = sanitized[:start] + replacement + sanitized[end:]
    return sanitized
