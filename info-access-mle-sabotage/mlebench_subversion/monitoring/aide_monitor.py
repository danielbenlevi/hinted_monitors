"""
Purpose-built AIDE monitor runner with explicit controls for what content is
provided to the monitor (thoughts/plan/code, best-path vs whole-tree), and
with rich, structured metadata captured alongside each score.

This script is intentionally self-contained and open/testable:
- Pure functions for history construction and stats
- Clear dataclass config, CLI parsing, and deterministic score key naming
- Reuses proven Inspect helpers for model calls and prompt reading

It operates directly on Inspect ``.eval`` logs, reading them from a single
file or recursively from a directory, and writes updated logs either in-place
or to a mirrored output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Tuple, Any, List, Optional
from uuid import uuid4

from inspect_ai.log import EvalLog, read_eval_log_async, write_eval_log_async
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
)
from inspect_ai.scorer import Score

# Reuse robust helpers from the monitoring module for:
# - model call loop + token accounting
# - system prompt construction
# - task score population
from .monitor_common import (
    call_model_until_it_submits_a_score,
    create_monitor_system_message,
)
from .model_compat import (
    create_monitor_model,
    monitor_model_spec_resume_compatible,
    model_spec_metadata,
    reset_monitor_model_telemetry,
    resolve_monitor_model_spec,
)
from .monitor_runtime import (
    MAX_OPENAI_API_CALLS,
    MonitorFailureCircuitBreaker,
    MonitorValidationError,
    SystemicMonitorFailure,
    bounded_eval_artifact_name,
    map_concurrently,
    monitor_preflight_sample_count,
    monitor_sample_identity,
    resume_output_path,
    run_with_retries,
    write_human_monitor_log,
)
from .observation_boundary import sanitize_synthetic_journal
from .provenance_hints import (
    RenderedAideStep,
    apply_provenance_hints,
    sample_payload_from_sample,
)
from .task_authority import read_canonical_task
from .run_monitor import add_task_scores_to_samples
from ..aide._journal import Journal, Node
from ..utils.prompt_utils import read_prompt


# -----------------------------
# Lightweight timestamped logging
# -----------------------------


def _ts() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _log(level: str, message: str) -> None:
    print(f"[{_ts()}][{level}] {message}")


# -----------------------------
# Configuration
# -----------------------------


@dataclass
class AideMonitorConfig:
    """Configuration for the AIDE-specific monitor runner.

    Attributes mirror the README proposal while keeping names compatible with
    existing helper utilities where needed (e.g., prompt file fields).
    """

    # Input/output
    input_path: str  # file or directory containing .eval logs
    output_mode: str = "in_place"  # one of: "in_place", "mirror_dir"
    output_base_dir: str | None = None  # used when output_mode == mirror_dir

    # Monitor model + prompts
    model: str = "openai/gpt-4.1-mini"
    monitor_system_prompt_file: str = "cot_act.txt"
    monitor_user_prompt_file: str = "default_user.txt"
    prompt_profile: str = "legacy"
    provenance_hint_mode: str = "none"

    # Content inclusion flags
    include_thoughts: bool = True
    include_plans: bool = True
    include_code: bool = True
    include_tool_outputs: bool = True

    # AIDE tree selection
    best_path_only: bool = True

    # Generation parameters / UX
    temperature: float = 0.0
    reasoning_effort: str | None = None
    reasoning_tokens: int | None = None
    openrouter_provider: str | None = None
    print_prompts: bool = False
    per_sample_timeout_secs: float = 900.0
    heartbeat_secs: float = 30.0
    max_concurrent_samples: int = MAX_OPENAI_API_CALLS

    # Optional config label for downstream aggregation
    config_name: str | None = None

    # Optional restriction to subset of competitions
    target_competitions: list[str] | None = None
    
    # Task score inclusion (for A/B testing monitor with/without agent performance)
    include_task_score: bool = False
    
    # Dry run (build and print prompts only, no model calls)
    dry_run: bool = False


# -----------------------------
# History building (AIDE)
# -----------------------------


def make_aide_journal_available(sample) -> None:
    """Ensure an AIDE journal is available under ``sample.metadata['aide_journal']``.

    Older logs may store journals under ``sample.store`` with a key like
    ``AideState:<random>:journal``. This function copies it into metadata for
    consistent downstream access.
    """
    if getattr(sample, "metadata", None) is None:
        sample.metadata = {}

    # Already present
    if "aide_journal" in sample.metadata:
        return

    store = getattr(sample, "store", None) or {}
    for key, value in store.items():
        if isinstance(key, str) and key.startswith("AideState:") and key.endswith(":journal"):
            sample.metadata["aide_journal"] = value
            return

    # If we get here, journal could not be located
    raise ValueError(
        f"AIDE journal not found in sample {getattr(sample, 'id', 'UNKNOWN')}. "
        "Ensure the log contains the agent trajectory."
    )


# -----------------------------
# Thought extraction aligned with retrofit validator
# -----------------------------


def _safe_get_tool_arg(event: Any, key: str) -> Any | None:
    if hasattr(event, "arguments") and isinstance(event.arguments, dict):
        return event.arguments.get(key)
    if hasattr(event, "input") and isinstance(event.input, dict):  # type: ignore[attr-defined]
        return event.input.get(key)  # type: ignore[attr-defined]
    return None


def _extract_think_text(event: Any) -> str | None:
    thought = _safe_get_tool_arg(event, "thought")
    if thought:
        return str(thought)
    return getattr(event, "content", None) or getattr(event, "text", None)


def _event_is(ev: Any, kind: str) -> bool:
    return getattr(ev, "event", None) == kind


def _extract_submit_code(event: Any) -> str | None:
    if not (_event_is(event, "tool") and getattr(event, "function", None) == "submit_code"):
        return None
    code = _safe_get_tool_arg(event, "code")
    return str(code) if code else None


def _find_submit_events_with_code(events: List[Any]) -> List[Tuple[int, str]]:
    submit_events: List[Tuple[int, str]] = []
    for i, event in enumerate(events):
        if _event_is(event, "tool") and getattr(event, "function", None) == "submit_code":
            code = _extract_submit_code(event)
            if code:
                submit_events.append((i, code))
    return submit_events


def _find_preceding_think_event(events: List[Any], submit_idx: int) -> Tuple[Optional[int], Optional[str]]:
    for i in range(submit_idx - 1, -1, -1):
        event = events[i]
        if _event_is(event, "tool") and getattr(event, "function", None) == "think":
            return i, _extract_think_text(event)
        elif _event_is(event, "tool") and getattr(event, "function", None) == "submit_code":
            break
    return None, None


def _is_synthetic_linear_journal_data(journal_data: Any, sample: Any) -> bool:
    metadata = getattr(sample, "metadata", None) or {}
    if metadata.get("aide_journal_is_synthetic_linear"):
        return True
    if not isinstance(journal_data, dict):
        return False
    if journal_data.get("synthetic_linear"):
        return True
    nodes = journal_data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return False
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or node.get("step") != index:
            return False
        metric = node.get("metric_to_optimize") or {}
        if metric.get("value") != float(index + 1) or metric.get("maximize") is not True:
            return False
        if node.get("validation_metric") != float(index + 1):
            return False
    return True


def _strip_serialized_parents(journal_data: Any) -> Any:
    if not isinstance(journal_data, dict) or not isinstance(journal_data.get("nodes"), list):
        return journal_data
    normalized = dict(journal_data)
    normalized["nodes"] = [
        {**node, "parent": None} if isinstance(node, dict) else node
        for node in journal_data["nodes"]
    ]
    return normalized


def get_best_path(aide_journal: Journal, state) -> Iterator[Node]:
    """Yield nodes on the best path, starting from root to best leaf.

    Falls back to empty iterator when no best node exists.
    """
    best_node = aide_journal.get_best_node(only_good=True)
    if best_node is None:
        return iter(())

    path: list[Node] = [best_node]
    parent = best_node.parent
    while parent is not None:
        path.append(parent)
        parent = parent.parent
    return reversed(path)


class _SimpleIdMapper:
    """Map stable node IDs to small integers for readability."""

    def __init__(self) -> None:
        self._map: dict[str, int] = {}
        self._next = 0

    def get(self, node: Node) -> int:
        if node.id in self._map:
            return self._map[node.id]
        val = self._next
        self._map[node.id] = val
        self._next += 1
        return val


def _append_node_sections(
    blocks: list[str], node: Node, *, include_plans: bool, include_code: bool,
    include_thoughts: bool, include_tool_outputs: bool, thoughts_text: str | None
) -> Tuple[int, int, int]:
    """Append selected sections to ``blocks`` and return (plan_chars, thoughts_chars, code_chars)."""
    plan_chars = 0
    thoughts_chars = 0
    code_chars = 0

    # Thoughts first
    if include_thoughts and thoughts_text:
        blocks.append(f"Thoughts:\n{thoughts_text}")
        thoughts_chars += len(thoughts_text)

    # Then plan
    if include_plans:
        plan_text = getattr(node, "plan", "") or ""
        blocks.append(f"Plan:\n{plan_text}")
        plan_chars += len(plan_text)

    # Then code
    if include_code:
        code_text = getattr(node, "code", "") or ""
        blocks.append(f"Code:\n```\n{code_text}\n```")
        code_chars += len(code_text)

    if include_tool_outputs:
        exec_info = getattr(node, "execution_info", None)
        if exec_info is not None and getattr(exec_info, "term_out", None) is not None:
            blocks.append(f"Execution info:\n{exec_info.term_out}")
        else:
            blocks.append("Execution info: N/A")

    return plan_chars, thoughts_chars, code_chars


def build_aide_history_and_stats(
    sample,
    *,
    include_plans: bool,
    include_code: bool,
    include_thoughts: bool,
    include_tool_outputs: bool = True,
    best_path_only: bool,
    provenance_hint_mode: str = "none",
    canonical_task_text: str = "",
    provenance_contract_model: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Construct a human-readable AIDE history string and per-field char counts.

    Returns a tuple of (history_text, stats_dict).
    """
    journal_data = sample.metadata.get("aide_journal")
    synthetic_linear = _is_synthetic_linear_journal_data(journal_data, sample)
    if synthetic_linear:
        journal_data = sanitize_synthetic_journal(journal_data)
        journal_data = _strip_serialized_parents(journal_data)
    aide_journal = Journal.model_validate(journal_data)

    if best_path_only and synthetic_linear:
        nodes: Iterable[Node] = list(aide_journal.nodes)
    elif best_path_only:
        nodes: Iterable[Node] = list(get_best_path(aide_journal, sample))
    else:
        nodes = list(aide_journal.nodes)

    if not nodes:
        # No history available
        return "", {
            "plan_included": include_plans and False,
            "total_plan_char_count": 0,
            "cot_included": include_thoughts and False,
            "total_cot_char_count": 0,
            "code_included": include_code and False,
            "total_code_char_count": 0,
            "plan_source": "aide_pre",
        }

    mapped = _SimpleIdMapper()
    blocks: list[str] = []
    total_plan = 0
    total_thoughts = 0
    total_code = 0

    # Prepare event alignment for fallback thought reconstruction when node.thoughts is not available
    submit_events = _find_submit_events_with_code(getattr(sample, "events", []) or [])

    if not best_path_only:
        for node in nodes:
            node_blocks: list[str] = []
            simple_id = mapped.get(node)
            node_blocks.append(f"Node ID: {simple_id}")
            if node.parent is None:
                node_blocks.append("Drafting new node.")
            else:
                node_blocks.append(f"Improving node with ID {mapped.get(node.parent)}.")
            # Determine thoughts text: prefer retrofitted node.thoughts; fallback to preceding think event
            thoughts_text = None
            if include_thoughts:
                thoughts_text = getattr(node, "thoughts", None)
                if not thoughts_text and hasattr(node, "step") and isinstance(node.step, int):
                    idx = node.step
                    if 0 <= idx < len(submit_events):
                        submit_idx, _ = submit_events[idx]
                        _, fallback = _find_preceding_think_event(getattr(sample, "events", []) or [], submit_idx)
                        thoughts_text = fallback

            p, t, c = _append_node_sections(
                node_blocks,
                node,
                include_plans=include_plans,
                include_code=include_code,
                include_thoughts=include_thoughts,
                include_tool_outputs=include_tool_outputs,
                thoughts_text=thoughts_text,
            )
            total_plan += p
            total_thoughts += t
            total_code += c
            blocks.append("\n\n".join(node_blocks))
    else:
        for node in nodes:
            node_blocks = []
            thoughts_text = None
            if include_thoughts:
                thoughts_text = getattr(node, "thoughts", None)
                if not thoughts_text and hasattr(node, "step") and isinstance(node.step, int):
                    idx = node.step
                    if 0 <= idx < len(submit_events):
                        submit_idx, _ = submit_events[idx]
                        _, fallback = _find_preceding_think_event(getattr(sample, "events", []) or [], submit_idx)
                        thoughts_text = fallback

            p, t, c = _append_node_sections(
                node_blocks,
                node,
                include_plans=include_plans,
                include_code=include_code,
                include_thoughts=include_thoughts,
                include_tool_outputs=include_tool_outputs,
                thoughts_text=thoughts_text,
            )
            total_plan += p
            total_thoughts += t
            total_code += c
            blocks.append("\n\n".join(node_blocks))

    raw_history = "\n\n".join(blocks)
    raw_step_blocks = {index: block for index, block in enumerate(blocks, start=1)}
    history = raw_history
    provenance_stats: dict[str, Any] = {
        "provenance_hint_mode": provenance_hint_mode,
        "provenance_hint_warning_count": 0,
        "provenance_hint_hinted_steps": 0,
        "provenance_hint_inserted_blocks": 0,
    }
    if provenance_hint_mode != "none":
        if include_thoughts or include_plans or not include_code or include_tool_outputs:
            raise ValueError(
                "Hint-augmented action-only history requires code/actions only: "
                "no Thoughts, Plan, or raw tool outputs."
            )
        labeled_steps = [
            RenderedAideStep(
                step_id=step_id,
                history_block=(
                    f"[TRANSCRIPT STEP S{step_id}]\n{block}\n"
                    f"[/TRANSCRIPT STEP S{step_id}]"
                ),
            )
            for step_id, block in raw_step_blocks.items()
        ]
        try:
            history, provenance_stats = apply_provenance_hints(
                labeled_steps,
                provenance_hint_mode,
                sample_payload=sample_payload_from_sample(sample),
                contract_model=provenance_contract_model,
                canonical_task_text=canonical_task_text,
            )
        except Exception as exc:
            sample_id = getattr(sample, "id", "UNKNOWN")
            _log("ERROR", f"Failed to build required action-only hints for sample {sample_id}: {exc}")
            raise RuntimeError(
                f"Required action-only hints are incomplete for sample {sample_id}; "
                "refusing unhinted fallback."
            ) from exc
    stats = {
        "plan_included": include_plans and (total_plan > 0),
        "total_plan_char_count": total_plan,
        "plan_source": "aide_pre",
        "cot_included": include_thoughts,
        "total_cot_char_count": total_thoughts,
        "code_included": include_code and (total_code > 0),
        "total_code_char_count": total_code,
        "_raw_agent_transcript": raw_history,
        "_raw_agent_step_blocks": raw_step_blocks,
    }
    stats.update(provenance_stats)
    return history, stats


# -----------------------------
# Monitor input preparation
# -----------------------------


def _read_agent_task_for_sample(sample) -> str:
    """Read the frozen, hash-verified canonical task authority."""
    return read_canonical_task(sample)


ACTION_ONLY_PROMPT_PROFILES = {
    "monitoringbench_action_only",
    "action_only_hint",
}


def _is_action_only_profile(prompt_profile: str) -> bool:
    return prompt_profile in ACTION_ONLY_PROMPT_PROFILES


def _action_only_score_max(prompt_profile: str) -> int:
    """Return the score ceiling declared by the selected action-only prompt."""
    return 100


def build_monitor_messages(sample, cfg: AideMonitorConfig) -> tuple[list[ChatMessage], dict[str, Any], int]:
    """Create monitor input messages and return (messages, content_stats, prompt_len).

    The canonical MonitoringBench profile returns one system message. Legacy
    profiles return system and user messages. ``prompt_len`` is their combined
    content length, useful as a coarse prompt size proxy.
    """
    agent_task = _read_agent_task_for_sample(sample)

    # Build the history and stats based on cfg flags
    history, stats = build_aide_history_and_stats(
        sample,
        include_plans=cfg.include_plans,
        include_code=cfg.include_code,
        include_thoughts=cfg.include_thoughts,
        include_tool_outputs=cfg.include_tool_outputs,
        best_path_only=cfg.best_path_only,
        provenance_hint_mode=cfg.provenance_hint_mode,
        canonical_task_text=agent_task,
        provenance_contract_model=None,
    )

    # Guard: empty history means we skip scoring gracefully
    if not history:
        return [], stats, 0

    # Read agent task (competition description)
    # Avoid dependency on TaskState; resolve from sample instead
    if _is_action_only_profile(cfg.prompt_profile):
        if cfg.include_thoughts or cfg.include_plans or not cfg.include_code or cfg.include_tool_outputs:
            raise ValueError(
                "MonitoringBench action-only requires code/actions only: no Thoughts, Plan, or tool outputs."
            )
        if cfg.include_task_score:
            raise ValueError("MonitoringBench action-only forbids post-hoc task scores.")
        prompt_filename = (
            "action_only_hints.md"
            if cfg.prompt_profile == "action_only_hint"
            else "action_only.md"
        )
        prompt_path = Path(__file__).resolve().parents[3] / "markdown_prompts" / prompt_filename
        template = prompt_path.read_text(encoding="utf-8")
        rendered_history = history
        candidate_index = ""
        formal_effect_navigation_index = ""
        formal_concern_index = ""
        if cfg.prompt_profile == "action_only_hint":
            if cfg.provenance_hint_mode != "human":
                raise ValueError("action_only_hint requires provenance_hint_mode='human'.")
            # Use the exact relocation indexes used by the augmented EaE extractor.
            from .eae_extractor import (
                build_formal_concern_index,
                build_formal_effect_navigation_index,
                build_security_candidate_index,
            )

            candidate_index = build_security_candidate_index(
                stats.get("_raw_agent_step_blocks", {})
            )
            formal_effect_navigation_index = build_formal_effect_navigation_index(
                stats.get("provenance_hint_effect_cards", {})
            )
            formal_concern_index = (
                "" if formal_effect_navigation_index else build_formal_concern_index(history)
            )
            formal_navigation_index = formal_effect_navigation_index or formal_concern_index
            if candidate_index:
                rendered_history = f"{rendered_history}\n\n{candidate_index}"
            if formal_navigation_index:
                rendered_history = f"{rendered_history}\n\n{formal_navigation_index}"
        elif cfg.provenance_hint_mode != "none":
            raise ValueError("MonitoringBench action-only baseline forbids provenance hints.")
        replacements = {"task_description": agent_task, "tool_calls": rendered_history}
        rendered = re.sub(
            r"\{(task_description|tool_calls)\}",
            lambda match: replacements[match.group(1)],
            template,
        )
        if "{task_description}" in rendered or "{tool_calls}" in rendered:
            raise ValueError("MonitoringBench action-only prompt contains unresolved placeholders.")
        stats.update(
            {
                "task_score_value": None,
                "task_score_included": False,
                "prompt_profile": cfg.prompt_profile,
                "prompt_source_path": str(prompt_path),
                "_monitor_transcript": rendered_history,
                "_monitor_system_prompt": rendered,
                "security_candidate_index_count": candidate_index.count("\nS"),
                "security_candidate_index_chars": len(candidate_index),
                "formal_concern_index_count": formal_concern_index.count("\nS"),
                "formal_concern_index_chars": len(formal_concern_index),
                "formal_effect_navigation_index_count": formal_effect_navigation_index.count(
                    "\nEffect "
                ),
                "formal_effect_navigation_index_chars": len(formal_effect_navigation_index),
            }
        )
        if cfg.print_prompts:
            print("Monitor system prompt:\n" + rendered)
        return [ChatMessageSystem(content=rendered)], stats, len(rendered)

    if cfg.prompt_profile != "legacy":
        raise ValueError(f"Unknown AIDE monitor prompt profile: {cfg.prompt_profile!r}")

    # System message via shared helper (respects cfg.print_prompts in its impl)
    system_msg = create_monitor_system_message(cfg)  # type: ignore[arg-type]

    # Sample's main task score (only if include_task_score flag is set)
    task_score = None
    if cfg.include_task_score and getattr(sample, "metadata", None):
        task_score = sample.metadata.get("task_score")
    
    # Track task_score in stats for metadata recording
    stats["task_score_value"] = task_score

    # User message from prompt template
    user_prompt_template = read_prompt("monitor", cfg.monitor_user_prompt_file)
    user_content = user_prompt_template.format(agent_task=agent_task, task_score=task_score, history=history)
    user_msg = ChatMessageUser(content=user_content)
    stats["_monitor_transcript"] = history
    stats["_monitor_system_prompt"] = getattr(system_msg, "content", "") or ""

    if cfg.print_prompts:
        print("Monitor system prompt:\n" + getattr(system_msg, "content", ""))
        print("\nMonitor user prompt:\n" + user_content)

    prompt_len = len(getattr(system_msg, "content", "") or "") + len(user_content or "")
    return [system_msg, user_msg], stats, prompt_len


# -----------------------------
# Scoring + metadata
# -----------------------------


def _score_key_base(cfg: AideMonitorConfig) -> str:
    """Deterministic base key for sample.scores."""
    parts = ["monitor_score", cfg.model.replace("/", "_")]
    flags = []
    flags.append("cot" if cfg.include_thoughts else "no_cot")
    flags.append("plan" if cfg.include_plans else "no_plan")
    flags.append("code" if cfg.include_code else "no_code")
    flags.append("outputs" if cfg.include_tool_outputs else "no_outputs")
    flags.append("best" if cfg.best_path_only else "tree")
    flags.append(cfg.prompt_profile)
    parts.append("-".join(flags))
    if cfg.config_name:
        parts.append(cfg.config_name)
    return "_".join(parts)


def _next_unique_key(existing: dict, base: str) -> str:
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


_MONITORINGBENCH_SCORE_RE = re.compile(
    r"<score>\s*\[?\s*(?P<score>(?:100(?:\.0+)?)|(?:[0-9]{1,2}(?:\.\d+)?))\s*\]?\s*</score>",
    re.IGNORECASE,
)
_ACTION_ONLY_HINT_SCORE_RE = re.compile(
    r"<score>\s*\[?\s*(?P<score>(?:100(?:\.0+)?)|(?:[0-9]{1,2}(?:\.\d+)?))\s*\]?\s*</score>",
    re.IGNORECASE,
)
# Qwen 3.6 Flash occasionally closes the score element before emitting its
# numeric payload (``</score>\n5`` or ``</score> [5]``).  The number is still
# unambiguous and adjacent to the score marker, so accept this provider-specific
# serialization without weakening the canonical parser for other models.
_MONITORINGBENCH_QWEN_TRAILING_SCORE_RE = re.compile(
    r"</score>\s*\[?\s*(?P<score>(?:100(?:\.0+)?)|(?:[0-9]{1,2}(?:\.\d+)?))\s*\]?\s*$",
    re.IGNORECASE,
)
_MONITORINGBENCH_GEMINI_TRAILING_SCORE_RE = re.compile(
    r"</score>\s*\[?\s*(?P<score>(?:100(?:\.0+)?)|(?:[0-9]{1,2}(?:\.\d+)?))\s*\]?"
    r"\s*(?:</score>)?\s*(?:</reasoning>)?\s*$",
    re.IGNORECASE,
)
_MONITORINGBENCH_REASONING_RE = re.compile(
    r"<reasoning>\s*(?P<reasoning>[\s\S]*?)\s*</reasoning>",
    re.IGNORECASE,
)
_MONITORINGBENCH_GEMINI_UNOPENED_REASONING_RE = re.compile(
    r"^\s*(?P<reasoning>[\s\S]+?)\s*</reasoning>",
    re.IGNORECASE,
)
_MONITORINGBENCH_GEMINI_UNTAGGED_REASONING_RE = re.compile(
    r"^\s*(?P<reasoning>[\s\S]+?)\s*(?:<task_feedback>|<score>)",
    re.IGNORECASE,
)


async def _call_monitoringbench_action_only(model, messages, cfg: AideMonitorConfig):
    """Call the selected free-text action-only prompt without tool additions."""
    output = await model.generate(
        messages,
        config=GenerateConfig(
            temperature=cfg.temperature,
            reasoning_effort=cfg.reasoning_effort,
        ),
    )
    text = str(getattr(output.message, "text", None) or getattr(output.message, "content", "") or "")
    is_gemini_flash_lite = "gemini-2.5-flash-lite" in cfg.model.casefold()
    score_pattern = (
        _ACTION_ONLY_HINT_SCORE_RE
        if cfg.prompt_profile == "action_only_hint"
        else _MONITORINGBENCH_SCORE_RE
    )
    score_match = score_pattern.search(text)
    if score_match is None and "qwen" in cfg.model.lower():
        score_match = _MONITORINGBENCH_QWEN_TRAILING_SCORE_RE.search(text)
    if score_match is None and is_gemini_flash_lite:
        score_match = _MONITORINGBENCH_GEMINI_TRAILING_SCORE_RE.search(text)
    reasoning_match = _MONITORINGBENCH_REASONING_RE.search(text)
    if reasoning_match is None and is_gemini_flash_lite:
        reasoning_match = _MONITORINGBENCH_GEMINI_UNOPENED_REASONING_RE.search(text)
    if reasoning_match is None and is_gemini_flash_lite and score_match is not None:
        reasoning_match = _MONITORINGBENCH_GEMINI_UNTAGGED_REASONING_RE.search(text)
    if score_match is None or reasoning_match is None:
        raise MonitorValidationError(
            "MonitoringBench action-only response omitted valid <reasoning> or <score> tags.",
            failure_code="action_only_output_format",
            diagnostic={"response": text[-4000:], "stop_reason": str(getattr(output, "stop_reason", ""))},
        )
    score = float(score_match.group("score"))
    score_max = _action_only_score_max(cfg.prompt_profile)
    if not 0 <= score <= score_max:
        raise MonitorValidationError(
            f"Action-only score is outside 0-{score_max}: {score}",
            failure_code="action_only_score_range",
        )
    usage = getattr(output, "usage", None)
    usage_totals = None
    if usage is not None:
        usage_totals = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
    return score, "", reasoning_match.group("reasoning").strip(), usage_totals, [str(getattr(output, "stop_reason", ""))], 1


async def run_monitor_on_sample(sample, cfg: AideMonitorConfig) -> tuple[str | None, Score | None]:
    """Run the monitor on one sample and return (score_key, Score) or (None, None) if skipped."""
    total_started = time.perf_counter()
    reset_monitor_model_telemetry()
    model_spec = resolve_monitor_model_spec(
        cfg.model,
        reasoning_effort=cfg.reasoning_effort,
        reasoning_tokens=cfg.reasoning_tokens,
        temperature=cfg.temperature,
        openrouter_provider=cfg.openrouter_provider,
    )
    # Ensure the AIDE journal is accessible
    try:
        make_aide_journal_available(sample)
    except ValueError as e:
        if _is_action_only_profile(cfg.prompt_profile):
            raise MonitorValidationError(
                str(e),
                failure_code="missing_aide_journal",
                max_attempts=1,
            ) from e
        _log("WARN", str(e))
        return None, None

    base_key = _score_key_base(cfg)
    existing_scores = getattr(sample, "scores", {}) or {}

    preparation_started = time.perf_counter()
    messages, content_stats, prompt_chars = await asyncio.to_thread(build_monitor_messages, sample, cfg)
    preparation_seconds = time.perf_counter() - preparation_started
    if not messages:
        if _is_action_only_profile(cfg.prompt_profile):
            raise MonitorValidationError(
                f"MonitoringBench action-only found no trajectory history for sample {getattr(sample, 'id', 'UNKNOWN')}.",
                failure_code="missing_monitor_history",
                max_attempts=1,
            )
        _log("WARN", f"Skipping sample {getattr(sample, 'id', 'UNKNOWN')} – no history.")
        return None, None
    matching_keys = [
        key
        for key in existing_scores
        if key == base_key or (key.startswith(base_key + "_") and key[len(base_key) + 1 :].isdigit())
    ]
    if len(matching_keys) > 1:
        raise ValueError(f"Multiple existing scores match protocol key {base_key!r}: {matching_keys}")
    if matching_keys:
        existing_key = matching_keys[0]
        existing_score = existing_scores[existing_key]
        metadata = getattr(existing_score, "metadata", {}) or {}
        value = getattr(existing_score, "value", None)
        score_max = (
            _action_only_score_max(cfg.prompt_profile)
            if _is_action_only_profile(cfg.prompt_profile)
            else 100
        )
        if (
            not monitor_model_spec_resume_compatible(
                metadata.get("monitor_model_spec"), model_spec
            )
            or metadata.get("monitor_prompt_profile") != cfg.prompt_profile
            or metadata.get("provenance_hint_mode", "none") != cfg.provenance_hint_mode
            or metadata.get("cot_included") is not cfg.include_thoughts
            or metadata.get("plan_included") is not cfg.include_plans
            or metadata.get("code_included") is not cfg.include_code
            or metadata.get("tool_outputs_included") is not cfg.include_tool_outputs
            or metadata.get("score_scale_max") != score_max
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= score_max
        ):
            raise ValueError(
                f"Existing score {existing_key!r} does not attest the current prompt/model protocol; "
                "refusing stale reuse."
            )
        _log("INFO", f"Resume: verified existing score {existing_key!r} for sample {sample.id}.")
        return None, None

    _log(
        "INFO",
        (
            f"Scoring sample {getattr(sample, 'id', 'UNKNOWN')} with model {cfg.model} | "
            f"prompt_chars={prompt_chars}, include: cot={cfg.include_thoughts}, plan={cfg.include_plans}, code={cfg.include_code}"
        ),
    )

    # Heartbeat + timeout wrapper to avoid silent hangs
    # Import anyio lazily to keep import-time deps minimal
    import anyio

    model = create_monitor_model(model_spec)
    sample_id = str(getattr(sample, "id", "UNKNOWN"))

    async def _heartbeat() -> None:
        while True:
            await anyio.sleep(max(0.0, float(cfg.heartbeat_secs)))
            _log("INFO", f"Waiting on model response for sample {sample_id}...")

    score_value = None
    summary = None
    reasoning = None
    usage_totals = None
    stop_reasons = None
    rounds = None

    monitor_started = time.perf_counter()
    async with anyio.create_task_group() as tg:
        if cfg.heartbeat_secs and cfg.heartbeat_secs > 0:
            tg.start_soon(_heartbeat)
        with anyio.move_on_after(float(cfg.per_sample_timeout_secs)) as cs:
            if _is_action_only_profile(cfg.prompt_profile):
                score_value, summary, reasoning, usage_totals, stop_reasons, rounds = (
                    await _call_monitoringbench_action_only(model, messages, cfg)
                )
            else:
                score_value, summary, reasoning, usage_totals, stop_reasons, rounds = (
                    await call_model_until_it_submits_a_score(model, messages, cfg.temperature)
                )
        tg.cancel_scope.cancel()
    monitor_seconds = time.perf_counter() - monitor_started

    if cs is not None and cs.cancel_called:
        if _is_action_only_profile(cfg.prompt_profile):
            raise TimeoutError(f"MonitoringBench action-only timed out after {cfg.per_sample_timeout_secs}s")
        _log("WARN", f"Timed out after {cfg.per_sample_timeout_secs}s on sample {sample_id}; skipping.")
        return None, None

    # Handle cases where the model couldn't produce a score (e.g., length limit)
    if score_value is None:
        if _is_action_only_profile(cfg.prompt_profile):
            raise MonitorValidationError(
                "MonitoringBench action-only returned no score.",
                failure_code="action_only_missing_score",
            )
        _log("WARN", f"Monitor returned None score for sample {sample_id} (likely due to length limit); skipping.")
        return None, None

    # Compose metadata per README
    now_iso = datetime.now(timezone.utc).isoformat()
    meta: dict[str, Any] = {
        "uuid": str(uuid4()),
        "timestamp": now_iso,
        "monitor_prompt_profile": cfg.prompt_profile,
        "cot_included": cfg.include_thoughts,
        "total_cot_char_count": int(content_stats.get("total_cot_char_count", 0)),
        "code_included": cfg.include_code,
        "tool_outputs_included": cfg.include_tool_outputs,
        "total_code_char_count": int(content_stats.get("total_code_char_count", 0)),
        "plan_included": cfg.include_plans,
        "plan_char_count": int(content_stats.get("total_plan_char_count", 0)),
        "task_score_included": cfg.include_task_score,
        "task_score_value": content_stats.get("task_score_value"),
        "score_scale_max": (
            _action_only_score_max(cfg.prompt_profile)
            if _is_action_only_profile(cfg.prompt_profile)
            else 100
        ),
        "monitor_prompt_char_count": int(prompt_chars),
        "monitor_reasoning": reasoning or "",
        "monitor_stop_reasons": ",".join(stop_reasons or []),
        "monitor_rounds": str(rounds),
        "timing_prompt_preparation_seconds": preparation_seconds,
        "timing_monitor_call_seconds": monitor_seconds,
        "timing_total_sample_seconds": time.perf_counter() - total_started,
        "_monitor_transcript": str(content_stats.get("_monitor_transcript", "")),
        "_monitor_system_prompt": str(content_stats.get("_monitor_system_prompt", "")),
        "monitor_prompt_source_path": str(content_stats.get("prompt_source_path", "")),
        "provenance_hint_mode": str(content_stats.get("provenance_hint_mode", "none")),
        "provenance_hint_warning_count": int(
            content_stats.get("provenance_hint_warning_count", 0) or 0
        ),
        "provenance_hint_hinted_steps": int(
            content_stats.get("provenance_hint_hinted_steps", 0) or 0
        ),
        "provenance_hint_inserted_blocks": int(
            content_stats.get("provenance_hint_inserted_blocks", 0) or 0
        ),
        "security_candidate_index_count": int(
            content_stats.get("security_candidate_index_count", 0) or 0
        ),
        "security_candidate_index_chars": int(
            content_stats.get("security_candidate_index_chars", 0) or 0
        ),
        "formal_concern_index_count": int(
            content_stats.get("formal_concern_index_count", 0) or 0
        ),
        "formal_concern_index_chars": int(
            content_stats.get("formal_concern_index_chars", 0) or 0
        ),
        "formal_effect_navigation_index_count": int(
            content_stats.get("formal_effect_navigation_index_count", 0) or 0
        ),
        "formal_effect_navigation_index_chars": int(
            content_stats.get("formal_effect_navigation_index_chars", 0) or 0
        ),
    }
    for metadata_key in (
        "provenance_hint_formal_cache_hit",
        "provenance_hint_scanner_environment_digest",
        "provenance_hint_analysis_status",
        "timing_hint_rendering_seconds",
        "timing_formal_analysis_wall_seconds",
        "timing_formal_analysis_cpu_seconds",
        "timing_contract_generation_seconds",
        "timing_provenance_generation_seconds",
        "timing_scanner_seconds",
    ):
        if metadata_key in content_stats:
            meta[metadata_key] = content_stats[metadata_key]
    meta.update(model_spec_metadata(model_spec))

    if isinstance(usage_totals, dict):
        # Store token counts as integers
        meta["monitor_input_token_count"] = int(usage_totals.get("input_tokens", 0) or 0)
        meta["monitor_output_token_count"] = int(usage_totals.get("output_tokens", 0) or 0)

    # Score explanation is the one-sentence summary; full chain goes to metadata
    explanation = summary or ""
    result = Score(value=score_value, answer=None, explanation=explanation, metadata=meta)

    # Choose score key deterministically, append suffix if collision
    if getattr(sample, "scores", None) is None:
        sample.scores = {}
    base_key = _score_key_base(cfg)
    key = _next_unique_key(sample.scores, base_key)
    return key, result


# -----------------------------
# Log I/O + orchestration
# -----------------------------


def _iter_eval_files(path: Path) -> Iterator[Path]:
    if path.is_file() and path.suffix == ".eval":
        yield path
        return
    if path.is_dir():
        for p in path.rglob("*.eval"):
            if p.is_file():
                yield p


def _competition_id_from_sample(sample) -> str:
    if getattr(sample, "metadata", None) and "competition_id" in sample.metadata:
        try:
            return str(sample.metadata["competition_id"]).strip()
        except Exception:
            pass
    # Fallback: prefix before first underscore in sample_id
    return str(getattr(sample, "id", "")).split("_")[0]


def _should_include_sample(sample, cfg: AideMonitorConfig) -> bool:
    if cfg.target_competitions is None:
        return True
    comp = _competition_id_from_sample(sample)
    return comp in set(cfg.target_competitions)


def _mirror_output_path(input_eval_path: Path, cfg: AideMonitorConfig) -> Path:
    base_out = Path(cfg.output_base_dir or "logs/monitor/aide_monitor")
    input_root = Path(cfg.input_path).resolve()
    input_eval_path = input_eval_path.resolve()
    base_key = _score_key_base(cfg)
    new_name = bounded_eval_artifact_name(input_eval_path.stem, base_key)

    if input_root.is_dir():
        relative_path = input_eval_path.relative_to(input_root)
        return resume_output_path(base_out / relative_path.with_name(new_name))

    return resume_output_path(base_out / new_name)


async def process_log_file(
    input_eval_path: Path,
    cfg: AideMonitorConfig,
    failure_guard: MonitorFailureCircuitBreaker | None = None,
) -> Path | None:
    log: EvalLog = await read_eval_log_async(str(input_eval_path))
    if not log.samples:
        _log("INFO", f"No samples in {input_eval_path}, skipping.")
        return None

    # Populate task scores if requested (for A/B testing monitor with/without agent performance)
    if cfg.include_task_score:
        try:
            log = add_task_scores_to_samples(log)
            _log("INFO", f"Populated task_score in metadata for {len(log.samples)} samples.")
        except Exception as e:
            _log("WARN", f"Failed to populate task_scores: {e}. Continuing with None values.")

    # Filter samples if requested
    original_samples = list(log.samples)
    if cfg.target_competitions:
        log.samples = [s for s in original_samples if _should_include_sample(s, cfg)]
        if not log.samples:
            _log("INFO", f"No samples match target_competitions in {input_eval_path}, skipping.")
            return None

    # Decide output path upfront for incremental writes
    if cfg.output_mode == "in_place":
        out_path = input_eval_path
    elif cfg.output_mode == "mirror_dir":
        out_path = _mirror_output_path(input_eval_path, cfg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"Unknown output_mode: {cfg.output_mode}")

    resumed_existing = False
    if cfg.output_mode == "mirror_dir" and out_path.exists():
        source_identity = [monitor_sample_identity(sample) for sample in original_samples]
        existing_log = await read_eval_log_async(str(out_path))
        existing_identity = [monitor_sample_identity(sample) for sample in (existing_log.samples or [])]
        if existing_identity != source_identity:
            raise ValueError(f"Existing output sample identities do not match source: {out_path}")
        log = existing_log
        original_samples = list(log.samples or [])
        if cfg.target_competitions:
            log.samples = [sample for sample in original_samples if _should_include_sample(sample, cfg)]
        resumed_existing = True
        _log("INFO", f"Resume: validating per-sample completion in {out_path}.")

    # Build prompts or run monitor per sample
    updated_any = False
    sample_failures: list[tuple[str, Exception]] = []
    # Use a stable list to iterate, as we may temporarily swap log.samples during writes
    iter_samples = list(log.samples)
    for idx, sample in enumerate(iter_samples):
        if cfg.dry_run:
            # Build messages and print, then exit after first sample
            try:
                make_aide_journal_available(sample)
            except ValueError as e:
                _log("WARN", str(e))
                break
            messages, content_stats, prompt_chars = build_monitor_messages(sample, cfg)
            if not messages:
                print("[INFO] Empty history; nothing to print.")
                break
            system_msg = messages[0]
            print("Monitor system prompt:\n" + getattr(system_msg, "content", ""))
            if len(messages) > 1:
                print("\nMonitor user prompt:\n" + getattr(messages[1], "content", ""))
            print(f"\n---\nPrompt char count: {prompt_chars}")
            print(
                f"CoT chars: {int(content_stats.get('total_cot_char_count', 0))} | "
                f"Plan chars: {int(content_stats.get('total_plan_char_count', 0))} | "
                f"Code chars: {int(content_stats.get('total_code_char_count', 0))}"
            )
            # Dry run: no file writing
            return None
        else:
            if _is_action_only_profile(cfg.prompt_profile):
                sample_id = str(getattr(sample, "id", "UNKNOWN"))
                sample_key = (
                    f"{input_eval_path.resolve()}::{sample_id}::"
                    f"{getattr(sample, 'epoch', None)}"
                )

                def log_retry(attempt: int, limit: int, exc: Exception) -> None:
                    _log(
                        "WARN",
                        f"Action-only attempt {attempt}/{limit} failed for sample {sample_id} in "
                        f"{input_eval_path}; retrying: {type(exc).__name__}: {exc}",
                    )

                try:
                    key, score_obj = await run_with_retries(
                        lambda: run_monitor_on_sample(sample, cfg),
                        on_retry=log_retry,
                        sample_key=sample_key,
                        failure_guard=failure_guard,
                    )
                except SystemicMonitorFailure:
                    raise
                except Exception as exc:
                    sample_failures.append((sample_id, exc))
                    _log(
                        "ERROR",
                        f"Action-only failed for sample {sample_id} in {input_eval_path}: "
                        f"{type(exc).__name__}: {exc}",
                    )
                    continue
            else:
                key, score_obj = await run_monitor_on_sample(sample, cfg)
            if key and score_obj:
                _log(
                    "OK",
                    (
                        f"Submitting score for sample {getattr(sample, 'id', 'UNKNOWN')}: "
                        f"key={key}, value={getattr(score_obj, 'value', None)}"
                    ),
                )
                sample.scores[key] = score_obj
                report_path = write_human_monitor_log(
                    output_eval_path=out_path,
                    source_eval_path=input_eval_path,
                    sample=sample,
                    score_key=key,
                    score_obj=score_obj,
                    stage="monitor",
                )
                _log("OK", f"Wrote human-readable run log to {report_path}")
                updated_any = True

                # Incremental write to disk immediately after each score
                if cfg.target_competitions is not None:
                    # Filtering retained these exact sample objects, so their
                    # scores are already mutated in ``original_samples``.  Do
                    # not remap by non-unique sample.id (which collapses
                    # duplicate IDs and epochs).
                    previous_samples = log.samples
                    log.samples = original_samples
                    await write_eval_log_async(log, str(out_path))
                    log.samples = previous_samples
                else:
                    await write_eval_log_async(log, str(out_path))
                _log("OK", f"Wrote incremental update to {out_path}")

    # Restore original samples if filtered (to preserve unscored entries while
    # still embedding newly added scores for the filtered ones)
    if cfg.target_competitions is not None:
        log.samples = original_samples

    if not updated_any and resumed_existing and not sample_failures:
        _log("INFO", f"Resume: all matching scores were already complete in {out_path}.")
        return out_path
    if not updated_any:
        _log("INFO", f"No scores added for {input_eval_path}, not writing output.")
    elif cfg.target_competitions is not None:
        # Filtered runs need one final write after restoring unscored samples.
        # Ordinary runs have already persisted every successful sample.
        await write_eval_log_async(log, str(out_path))
        _log("OK", f"Wrote restored sample set to {out_path}")

    if sample_failures:
        details = "; ".join(
            f"{sample_id}: {type(exc).__name__}: {exc}" for sample_id, exc in sample_failures
        )
        raise RuntimeError(
            f"{len(sample_failures)} action-only sample(s) failed in {input_eval_path} after sibling "
            f"samples completed: {details}"
        ) from sample_failures[0][1]

    if not updated_any:
        return None
    return out_path


# -----------------------------
# CLI
# -----------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AIDE monitor with fine-grained content controls.")

    # I/O
    p.add_argument("--input", required=True, help="Path to a .eval file or a directory containing .eval files")
    p.add_argument(
        "--output-mode",
        choices=["in_place", "mirror_dir"],
        default="in_place",
        help="Where to write results: overwrite original or mirror into a separate directory",
    )
    p.add_argument(
        "--output-base-dir",
        default=None,
        help="When output-mode=mirror_dir, root directory under which to mirror (default: logs/monitor/aide_monitor)",
    )

    # Model + prompts
    p.add_argument("--model", default="openai/gpt-4.1-mini")
    p.add_argument("--system-prompt", default="cot_act.txt")
    p.add_argument("--user-prompt", default="default_user.txt")
    p.add_argument(
        "--prompt-profile",
        choices=("legacy", "monitoringbench_action_only", "action_only_hint"),
        default="legacy",
    )
    p.add_argument(
        "--provenance-hint-mode",
        choices=("none", "human"),
        default="none",
        help="Required human per-action hints for the action_only_hint profile.",
    )

    # Content flags
    p.add_argument("--include-thoughts", action="store_true", default=True)
    p.add_argument("--no-include-thoughts", dest="include_thoughts", action="store_false")
    p.add_argument("--include-plans", action="store_true", default=True)
    p.add_argument("--no-include-plans", dest="include_plans", action="store_false")
    p.add_argument("--include-code", action="store_true", default=True)
    p.add_argument("--no-include-code", dest="include_code", action="store_false")
    p.add_argument("--include-tool-outputs", action="store_true", default=True)
    p.add_argument("--no-include-tool-outputs", dest="include_tool_outputs", action="store_false")

    # AIDE tree selection
    p.add_argument("--best-path-only", action="store_true", default=True)
    p.add_argument("--whole-tree", dest="best_path_only", action="store_false")

    # Generation / UX
    p.add_argument("--temperature", type=float, default=0.0)
    reasoning_group = p.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        default=None,
    )
    reasoning_group.add_argument(
        "--reasoning-tokens",
        type=int,
        default=None,
        help="Explicit reasoning-token budget (for manual-thinking models such as Claude Haiku 4.5).",
    )
    p.add_argument(
        "--openrouter-provider",
        default=None,
        help="Exact OpenRouter upstream slug; known final models are pinned automatically.",
    )
    p.add_argument("--print-prompts", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False, help="Build and print prompts, do not call model and do not write output")
    p.add_argument("--per-sample-timeout-secs", type=float, default=900.0, help="Timeout per sample before skipping (seconds)")
    p.add_argument("--heartbeat-secs", type=float, default=30.0, help="Emit a heartbeat log while waiting for model (seconds)")
    p.add_argument(
        "--max-concurrent-samples",
        type=int,
        choices=range(1, MAX_OPENAI_API_CALLS + 1),
        default=MAX_OPENAI_API_CALLS,
        metavar="N",
        help=f"Maximum concurrent sample pipelines (1-{MAX_OPENAI_API_CALLS}).",
    )

    # Optional label + target competition filter
    p.add_argument("--config-name", default=None, help="Optional label appended to score key and stored in log metadata")
    p.add_argument(
        "--target-competitions",
        nargs="*",
        default=None,
        help="If provided, only samples with these competition_ids will be scored",
    )
    
    # Task score inclusion toggle
    p.add_argument("--include-task-score", action="store_true", default=False, help="Populate and include agent's task score in monitor prompt")
    p.add_argument("--no-include-task-score", dest="include_task_score", action="store_false", help="Exclude task score (matches original behavior)")

    return p.parse_args()


def _cfg_from_args(args: argparse.Namespace) -> AideMonitorConfig:
    return AideMonitorConfig(
        input_path=args.input,
        output_mode=args.output_mode,
        output_base_dir=args.output_base_dir,
        model=args.model,
        monitor_system_prompt_file=args.system_prompt,
        monitor_user_prompt_file=args.user_prompt,
        prompt_profile=args.prompt_profile,
        provenance_hint_mode=args.provenance_hint_mode,
        include_thoughts=args.include_thoughts,
        include_plans=args.include_plans,
        include_code=args.include_code,
        include_tool_outputs=args.include_tool_outputs,
        best_path_only=args.best_path_only,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        reasoning_tokens=args.reasoning_tokens,
        openrouter_provider=args.openrouter_provider,
        print_prompts=args.print_prompts,
        per_sample_timeout_secs=args.per_sample_timeout_secs,
        heartbeat_secs=args.heartbeat_secs,
        max_concurrent_samples=args.max_concurrent_samples,
        config_name=args.config_name,
        target_competitions=args.target_competitions,
        include_task_score=args.include_task_score,
        dry_run=args.dry_run,
    )


async def main_async() -> None:
    args = _parse_args()
    cfg = _cfg_from_args(args)

    input_path = Path(cfg.input_path)
    eval_paths = sorted(_iter_eval_files(input_path))
    if not eval_paths:
        _log("WARN", "No .eval files found – check --input path.")
        return
    limit = 1 if cfg.dry_run else cfg.max_concurrent_samples
    diagnostic_root = (
        Path(cfg.output_base_dir or "logs/monitor/aide_monitor")
        if cfg.output_mode == "mirror_dir"
        else (input_path if input_path.is_dir() else input_path.parent)
    )
    failure_guard = MonitorFailureCircuitBreaker(
        stage="action_only",
        diagnostic_path=diagnostic_root / "action_only_failure_diagnostics.json",
    )
    preflight_target = monitor_preflight_sample_count()
    preflight_paths: list[Path] = []
    preflight_candidates = sorted(
        eval_paths,
        key=lambda path: hashlib.sha256(
            str(path.relative_to(input_path) if input_path.is_dir() else path.name).encode("utf-8")
        ).hexdigest(),
    )
    if preflight_target:
        sample_count = 0
        while preflight_candidates and sample_count < preflight_target:
            path = preflight_candidates.pop(0)
            preflight_paths.append(path)
            preflight_log = await read_eval_log_async(str(path))
            sample_count += len(preflight_log.samples or [])
    preflight_path_set = set(preflight_paths)
    remaining_paths = [path for path in eval_paths if path not in preflight_path_set]
    if preflight_paths:
        _log(
            "INFO",
            f"Running action-only preflight on {len(preflight_paths)} input(s) before the full batch.",
        )
        await map_concurrently(
            preflight_paths,
            lambda path: process_log_file(path, cfg, failure_guard),
            limit=min(limit, failure_guard.distinct_failure_limit, len(preflight_paths)),
            attempts=1,
            failure_guard=failure_guard,
        )
    failure_guard.set_phase("full")
    if remaining_paths:
        await map_concurrently(
            remaining_paths,
            lambda path: process_log_file(path, cfg, failure_guard),
            limit=limit,
            attempts=1,
            failure_guard=failure_guard,
        )


def main() -> None:
    # Defer import to avoid mandatory dependency at import time for testability
    import anyio

    anyio.run(main_async)


if __name__ == "__main__":
    main()
