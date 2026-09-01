"""Load exact human-reviewed Extract-and-Evaluate context windows.

The Markdown documents are the experiment's source of truth.  This module
extracts only the model-visible system/user fences and forced-tool wording;
the explanatory Markdown outside those fences is never sent to the model.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


REVIEWED_PROMPT_CONTEXT_MODES = (
    "default",
    "eae_base",
    "eae_base_2",
    "our",
    "our_low",
    "our_low_2",
    "our_low_3",
)
DEFAULT_REVIEWED_PROMPT_CONTEXT_MODE = "our_low"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCUMENTS = {
    ("eae_base", "extractor"): _REPO_ROOT / "markdown_prompts" / "eae_base_extract.md",
    ("eae_base", "evaluator"): _REPO_ROOT / "markdown_prompts" / "eae_base_eval.md",
    ("eae_base_2", "evaluator"): _REPO_ROOT / "markdown_prompts" / "eae_base_eval_2.md",
    ("our", "extractor"): _REPO_ROOT / "markdown_prompts" / "our_extract.md",
    ("our", "evaluator"): _REPO_ROOT / "markdown_prompts" / "our_eval.md",
    ("our_low", "extractor"): _REPO_ROOT / "markdown_prompts" / "our_extract_low.md",
    ("our_low", "evaluator"): _REPO_ROOT / "markdown_prompts" / "our_eval_low.md",
    ("our_low_2", "extractor"): _REPO_ROOT / "markdown_prompts" / "our_extract_low.md",
    ("our_low_2", "evaluator"): _REPO_ROOT / "markdown_prompts" / "our_eval_low_2.md",
    ("our_low_3", "extractor"): _REPO_ROOT / "markdown_prompts" / "our_extract_low.md",
    ("our_low_3", "evaluator"): _REPO_ROOT / "markdown_prompts" / "our_eval_low_3.md",
}

_TASK_PLACEHOLDER = "<<CANONICAL USER TASK — exact frozen task-authority text>>"
_EXTRACTOR_HISTORY_END = "<<REPEAT FOR EVERY BEST-PATH STEP S2...SN>>"
_EXTRACTOR_CANDIDATE_START = "DETERMINISTIC SECURITY-CANDIDATE INDEX (lexical and fallible)"
_EXTRACTOR_CANDIDATE_END = (
    "<<ZERO OR MORE BOUNDED S# LINES CONTAINING EXTRACTED URL/IP/PATH/SIGNAL CANDIDATES>>"
)
_EXTRACTOR_FORMAL_START = "AUTHORIZATION-AWARE FORMAL EFFECT NAVIGATION (fallible leads, not verdicts)"
_EXTRACTOR_FORMAL_END = (
    "<<ZERO OR MORE BOUNDED STABLE-EFFECT LINES WITH OCCURRENCES, REFERENCES, EFFECT, SCOPE, "
    "AUTHORIZATION, PROVENANCE, AND LIFECYCLE>>"
)
_EVALUATOR_HISTORY_START = "EVIDENCE UNIT 1 (transcript step S1)"
_DYNAMIC_LABELS_PLACEHOLDER = "<<DYNAMIC LIST OF ALL DISPLAYED EVIDENCE-UNIT LABELS>>"


@dataclass(frozen=True)
class ReviewedPromptContext:
    mode: str
    stage: str
    source_path: Path
    source_sha256: str
    system: str
    user_template: str
    tool_description: str
    tool_parameters: dict[str, str]
    history_start: str
    history_end: str
    permits_formal_hints: bool


def _extract_message_fence(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*\n\n~~~~text\n(?P<body>[\s\S]*?)\n~~~~\s*$",
        text,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Reviewed prompt document is missing {heading!r} fenced content.")
    return match.group("body")


def _replace_inclusive(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        raise ValueError(f"Reviewed prompt template is missing start marker: {start!r}")
    end_index = text.find(end, start_index)
    if end_index < 0:
        raise ValueError(f"Reviewed prompt template is missing end marker: {end!r}")
    return text[:start_index] + replacement + text[end_index + len(end) :]


@lru_cache(maxsize=16)
def load_reviewed_prompt_context(mode: str, stage: str) -> ReviewedPromptContext:
    if mode not in REVIEWED_PROMPT_CONTEXT_MODES or mode == "default":
        raise ValueError(f"No reviewed prompt document for mode={mode!r}.")
    if stage not in {"extractor", "evaluator"}:
        raise ValueError(f"Unknown reviewed prompt stage: {stage!r}")
    path = _DOCUMENTS[(mode, stage)]
    raw = path.read_text(encoding="utf-8")
    system = _extract_message_fence(raw, "Message 1 — system")
    user_template = _extract_message_fence(raw, "Message 2 — user")
    if "SCORE ON TEST SET" in system or "SCORE ON TEST SET" in user_template:
        raise ValueError(f"Reviewed prompt document still contains the removed test-score field: {path}")
    tool_match = re.search(r"^```json\n(?P<body>\{[\s\S]*?\})\n```\s*$", raw, re.MULTILINE)
    if tool_match is None:
        raise ValueError(f"Reviewed prompt document has no JSON tool definition: {path}")
    tool = json.loads(tool_match.group("body"))
    expected_name = "submit_step_spans" if stage == "extractor" else "submit_score"
    if tool.get("name") != expected_name or tool.get("tool_choice") != "any":
        raise ValueError(f"Unexpected forced-tool declaration in {path}: {tool!r}")
    parameters = tool.get("parameters")
    if not isinstance(parameters, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in parameters.items()
    ):
        raise ValueError(f"Reviewed prompt tool parameters must be string descriptions: {path}")
    return ReviewedPromptContext(
        mode=mode,
        stage=stage,
        source_path=path,
        source_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        system=system,
        user_template=user_template,
        tool_description=str(tool["description"]),
        tool_parameters=dict(parameters),
        history_start=(
            "[TRANSCRIPT STEP S1]"
            if stage == "extractor" and mode == "eae_base"
            else "[PROV-HINT S1]"
            if stage == "extractor"
            else _EVALUATOR_HISTORY_START
        ),
        history_end=(
            _EXTRACTOR_HISTORY_END
            if stage == "extractor"
            else "[/TRANSCRIPT STEP S1]\n\n..."
            if mode in {"eae_base", "eae_base_2"}
            else "[/PROV-HINT]"
        ),
        permits_formal_hints=mode not in {"eae_base", "eae_base_2"},
    )


def render_reviewed_extractor_user(
    context: ReviewedPromptContext,
    *,
    canonical_task: str,
    history: str,
    candidate_index: str,
    formal_navigation_index: str,
) -> str:
    rendered = context.user_template.replace(_TASK_PLACEHOLDER, canonical_task)
    rendered = _replace_inclusive(
        rendered,
        context.history_start,
        context.history_end,
        history.rstrip(),
    )
    if context.permits_formal_hints:
        rendered = _replace_inclusive(
            rendered,
            _EXTRACTOR_CANDIDATE_START,
            _EXTRACTOR_CANDIDATE_END,
            candidate_index.rstrip(),
        )
        rendered = _replace_inclusive(
            rendered,
            _EXTRACTOR_FORMAL_START,
            _EXTRACTOR_FORMAL_END,
            formal_navigation_index.rstrip(),
        )
    elif candidate_index.strip() or formal_navigation_index.strip():
        raise ValueError("Base EaE prompt forbids deterministic or formal hint augmentation.")
    return rendered


def render_reviewed_evaluator_user(
    context: ReviewedPromptContext,
    *,
    canonical_task: str,
    evidence_units: str,
) -> str:
    rendered = context.user_template.replace(_TASK_PLACEHOLDER, canonical_task)
    rendered = _replace_inclusive(
        rendered,
        context.history_start,
        context.history_end,
        evidence_units.rstrip(),
    )
    return rendered


def reviewed_tool_text(
    context: ReviewedPromptContext,
    *,
    required_unit_labels: list[str] | None = None,
) -> dict[str, Any]:
    parameters = dict(context.tool_parameters)
    if required_unit_labels is not None:
        replacement = repr(required_unit_labels or ["(none supplied)"])
        parameters = {
            key: value.replace(_DYNAMIC_LABELS_PLACEHOLDER, replacement)
            for key, value in parameters.items()
        }
    unresolved = [value for value in parameters.values() if "<<" in value]
    if unresolved:
        raise ValueError("Reviewed forced-tool wording contains an unresolved placeholder.")
    return {"description": context.tool_description, "parameters": parameters}
