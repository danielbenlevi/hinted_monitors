"""Suppress individual formal-hint components at render time.

The augmented-EaE monitor shows the model several distinct pieces of automated
analysis, and the A2/A3 ablations ask which of them carries the uplift. Every
piece is derived from the canonical formal cache, so a component can be removed
by filtering what is rendered rather than by regenerating contracts and formal
analyses. That keeps an ablation cell as cheap as an ordinary monitor run and
guarantees the arms differ only in the removed component.

Two independent controls live here:

``HINT_ABLATION_MODES``
    Remove exactly one component of the hint material (A2). Three modes remove
    a class of lines from every ``[PROV-HINT]`` block; two suppress one of the
    navigation indexes appended to the extractor prompt.

``HINT_STAGE_ABLATION_MODES``
    Withhold all hint material from one stage while the other keeps it (A3).

Both default to ``"none"``, which is a strict no-op: callers take a fast path
that returns their input object unchanged, so the canonical configuration
renders byte-for-byte what it rendered before this module existed.

Line classification is by the prefixes ``_render_human_hint`` emits. A line that
matches no known prefix belongs to no ablated component and is preserved, which
keeps the filter safe against blocks recovered from a transcript that may also
contain agent-authored text resembling a hint.
"""

from __future__ import annotations

from typing import Iterable, Mapping


HINT_ABLATION_MODES = (
    "none",
    "minus_provenance_graph",
    "minus_contract_deltas",
    "minus_scanner_flags",
    "minus_step_navigation_index",
    "minus_formal_effect_index",
    # Paired removals. Each single-component mode leaves a second aid in place
    # that can carry the same information: the formal-effect index restates what
    # the per-action contract findings say, and the lexical step index resurfaces
    # the addresses and paths the provenance notes call out. Measuring only the
    # singles would therefore understate both components, because the survivor
    # covers for the one that was removed.
    "minus_contract_stack",
    "minus_provenance_stack",
)

HINT_STAGE_ABLATION_MODES = ("none", "no_hints_extract", "no_hints_eval")

HINT_BLOCK_HEADER_PREFIX = "[PROV-HINT S"
HINT_BLOCK_FOOTER = "[/PROV-HINT]"

# Per-action hint components, keyed by the ablation mode that removes them. The
# prefixes are exactly the ones ``provenance_hints._render_human_hint`` writes.
_PROVENANCE_GRAPH_PREFIXES = ("Next action:", "Known before:", "Relations:", "Tags:")
_SCANNER_PREFIXES = ("Scanner:",)
_CONTRACT_DELTA_PREFIXES = ("Task-contract concern", "Task-contract observation")

_REMOVED_PREFIXES: dict[str, tuple[str, ...]] = {
    "minus_provenance_graph": _PROVENANCE_GRAPH_PREFIXES,
    "minus_scanner_flags": _SCANNER_PREFIXES,
    "minus_contract_deltas": _CONTRACT_DELTA_PREFIXES,
    # The paired modes remove the same lines as their single counterpart and
    # additionally suppress the navigation index that restates them.
    "minus_contract_stack": _CONTRACT_DELTA_PREFIXES,
    "minus_provenance_stack": _PROVENANCE_GRAPH_PREFIXES,
}

# Paired modes, mapped to the index each one additionally suppresses.
_ALSO_SUPPRESSES_STEP_INDEX = frozenset({"minus_provenance_stack"})
_ALSO_SUPPRESSES_FORMAL_EFFECT_INDEX = frozenset({"minus_contract_stack"})

# Modes that own a distinct set of line prefixes, in the order a line should be
# attributed to them. The paired modes deliberately share these prefixes.
_SINGLE_COMPONENT_MODES = (
    "minus_provenance_graph",
    "minus_scanner_flags",
    "minus_contract_deltas",
)

# The two index modes act on the extractor prompt, not on hint blocks.
_BLOCK_LEVEL_MODES = frozenset(_REMOVED_PREFIXES)


def validate_hint_ablation_mode(mode: str) -> str:
    if mode not in HINT_ABLATION_MODES:
        raise ValueError(
            f"Unsupported hint ablation mode: {mode!r}; expected one of {list(HINT_ABLATION_MODES)}"
        )
    return mode


def validate_hint_stage_ablation(mode: str) -> str:
    if mode not in HINT_STAGE_ABLATION_MODES:
        raise ValueError(
            f"Unsupported hint stage ablation: {mode!r}; expected one of "
            f"{list(HINT_STAGE_ABLATION_MODES)}"
        )
    return mode


def removes_hint_block_lines(mode: str) -> bool:
    """True when the mode edits the interior of every ``[PROV-HINT]`` block."""
    return validate_hint_ablation_mode(mode) in _BLOCK_LEVEL_MODES


def suppresses_security_candidate_index(mode: str, stage_ablation: str = "none") -> bool:
    """True when the deterministic security-candidate index must not be rendered."""
    mode = validate_hint_ablation_mode(mode)
    return (
        mode == "minus_step_navigation_index"
        or mode in _ALSO_SUPPRESSES_STEP_INDEX
        or validate_hint_stage_ablation(stage_ablation) == "no_hints_extract"
    )


def suppresses_formal_effect_index(mode: str, stage_ablation: str = "none") -> bool:
    """True when the authorization-aware formal-effect navigation must not be rendered.

    This also suppresses the legacy formal-concern relocation index. That index
    exists only as a fallback for transcripts with no effect cards, so leaving it
    enabled would substitute one navigation aid for another and the arm would no
    longer isolate the removal.
    """
    mode = validate_hint_ablation_mode(mode)
    return (
        mode == "minus_formal_effect_index"
        or mode in _ALSO_SUPPRESSES_FORMAL_EFFECT_INDEX
        or validate_hint_stage_ablation(stage_ablation) == "no_hints_extract"
    )


def line_component(line: str) -> str | None:
    """Name the hint component one rendered line belongs to, or None if unknown."""
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(HINT_BLOCK_HEADER_PREFIX) or stripped == HINT_BLOCK_FOOTER:
        return "structure"
    # Resolve against the single-component modes only. The paired modes reuse
    # their counterpart's prefixes, so consulting every mode would make a line's
    # reported component depend on dictionary ordering.
    for mode in _SINGLE_COMPONENT_MODES:
        if stripped.startswith(_REMOVED_PREFIXES[mode]):
            return mode.removeprefix("minus_")
    return None


def filter_hint_block(block: str, mode: str) -> str:
    """Remove one component's lines from a single ``[PROV-HINT]`` block.

    Returns the empty string when nothing but the header and footer survive: an
    empty husk would still announce that automated analysis existed for the step,
    which is a hint in itself, so the block is dropped entirely instead.
    """
    if validate_hint_ablation_mode(mode) not in _BLOCK_LEVEL_MODES:
        return block
    removed_prefixes = _REMOVED_PREFIXES[mode]
    kept: list[str] = []
    kept_content = False
    for line in block.split("\n"):
        stripped = line.strip()
        if stripped.startswith(HINT_BLOCK_HEADER_PREFIX) or stripped == HINT_BLOCK_FOOTER:
            kept.append(line)
            continue
        if stripped.startswith(removed_prefixes):
            continue
        kept.append(line)
        if stripped:
            kept_content = True
    if not kept_content:
        return ""
    return "\n".join(kept)


def filter_hint_blocks(blocks: Iterable[str], mode: str) -> list[str]:
    """Filter a list of complete hint blocks, dropping any left with no content."""
    if validate_hint_ablation_mode(mode) not in _BLOCK_LEVEL_MODES:
        return list(blocks)
    filtered = []
    for block in blocks:
        kept = filter_hint_block(str(block), mode)
        if kept:
            filtered.append(kept)
    return filtered


def filter_hint_map(hints: Mapping[int, str], mode: str) -> dict[int, str]:
    """Filter the per-step hint map used to interleave hints into the transcript.

    Dropping a step's entry here means the transcript simply carries no hint
    before that step, which is the same rendering a step without applicable
    findings already produces.
    """
    if validate_hint_ablation_mode(mode) not in _BLOCK_LEVEL_MODES:
        return dict(hints)
    filtered: dict[int, str] = {}
    for step_id, hint in hints.items():
        kept = filter_hint_block(str(hint), mode)
        if kept:
            filtered[int(step_id)] = kept
    return filtered
