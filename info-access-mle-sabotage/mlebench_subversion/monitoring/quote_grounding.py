from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_HINT_STEP_RE = re.compile(r"\[PROV-HINT\s+S(\d+)\]")
_MARKDOWN_FENCE_RE = re.compile(r"^\s*```(?:text|bash|sh|python)?\s*\n(?P<body>[\s\S]*?)\n```\s*$", re.IGNORECASE)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_GENERIC_LINES = {
    "plan:",
    "thoughts:",
    "code:",
    "execution info:",
    "execution info: n/a",
    "stdout:",
    "stderr:",
    "tool-assisted action.",
}


@dataclass(frozen=True)
class GroundingResult:
    quotes: list[str]
    input_count: int
    exact_count: int
    normalized_count: int
    hint_step_count: int
    anchor_count: int
    fuzzy_count: int
    dropped_count: int


def _quote_list(quotes: Any) -> list[str]:
    if isinstance(quotes, str):
        return [quotes]
    if isinstance(quotes, Sequence) and not isinstance(quotes, (bytes, bytearray)):
        return [str(quote) for quote in quotes if quote is not None]
    return []


def _trim_wrappers(text: str) -> str:
    cleaned = text.strip()
    fence_match = _MARKDOWN_FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group("body").strip()
    if cleaned.startswith("- "):
        cleaned = cleaned[2:].strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'", "`"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _normalized_with_offsets(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    offsets: list[int] = []
    in_whitespace = False
    for index, char in enumerate(text):
        if char.isspace():
            if normalized and not in_whitespace:
                normalized.append(" ")
                offsets.append(index)
            in_whitespace = True
            continue
        normalized.append(char)
        offsets.append(index)
        in_whitespace = False
    if normalized and normalized[-1] == " ":
        normalized.pop()
        offsets.pop()
    return "".join(normalized), offsets


def _whitespace_match(
    text: str,
    raw_history: str,
    normalized_history: str,
    history_offsets: list[int],
) -> str | None:
    normalized_quote, _ = _normalized_with_offsets(text)
    if len(normalized_quote) < 24:
        return None
    start = normalized_history.find(normalized_quote)
    if start < 0 or normalized_history.find(normalized_quote, start + 1) >= 0:
        return None
    raw_start = history_offsets[start]
    raw_end = history_offsets[start + len(normalized_quote) - 1] + 1
    return raw_history[raw_start:raw_end]


def _informative_anchor(candidate: str) -> bool:
    stripped = candidate.strip()
    if len(stripped) < 32 or stripped.lower() in _GENERIC_LINES:
        return False
    tokens = re.findall(r"[A-Za-z0-9_./:@=-]+", stripped)
    if len(tokens) < 4:
        return False
    return len(set(token.lower() for token in tokens)) >= 4


def _bounded_excerpt(block: str, anchor: str | None = None, max_chars: int = 12_000) -> str:
    if len(block) <= max_chars:
        return block
    anchor_start = block.find(anchor) if anchor else -1
    if anchor_start < 0:
        return block[:max_chars]
    start = max(0, anchor_start - max_chars // 3)
    end = min(len(block), start + max_chars)
    start = max(0, end - max_chars)
    return block[start:end]


def _step_for_anchor(anchor: str, raw_steps: Mapping[int, str]) -> str | None:
    matches = [block for block in raw_steps.values() if anchor in block]
    if len(matches) != 1:
        return None
    return _bounded_excerpt(matches[0], anchor)


def _anchor_match(text: str, raw_steps: Mapping[int, str]) -> str | None:
    candidates = [line.strip() for line in text.splitlines()]
    candidates.extend(match.group(1).strip() for match in _INLINE_CODE_RE.finditer(text))
    candidates.extend(
        match.group(1).strip()
        for line in text.splitlines()
        if (match := re.match(r"^\s*\d+\s+(.+)$", line)) is not None
    )
    for candidate in sorted(set(candidates), key=len, reverse=True):
        if not _informative_anchor(candidate):
            continue
        match = _step_for_anchor(candidate, raw_steps)
        if match is not None:
            return match
    return None


def ground_extractor_quotes(
    quotes: Any,
    *,
    raw_history: str,
    raw_steps: Mapping[int, str],
) -> GroundingResult:
    """Map extractor selections to exact raw-history excerpts.

    Inserted provenance hints are never valid source evidence. A hint selection
    can only recover the corresponding raw action block by its explicit step ID.
    Ambiguous summaries and composites are dropped rather than reconstructed.
    """
    submitted = _quote_list(quotes)
    grounded: list[str] = []
    seen: set[str] = set()
    counts = {
        "exact": 0,
        "normalized": 0,
        "hint_step": 0,
        "anchor": 0,
        "fuzzy": 0,
        "dropped": 0,
    }
    normalized_history, history_offsets = _normalized_with_offsets(raw_history)

    def add(value: str) -> None:
        if not value or value not in raw_history:
            raise AssertionError("Grounded extractor quote is not contained in raw agent history")
        if value not in seen:
            seen.add(value)
            grounded.append(value)

    for submitted_quote in submitted:
        candidate = _trim_wrappers(submitted_quote)
        if not candidate:
            counts["dropped"] += 1
            continue
        if candidate.lower() in _GENERIC_LINES:
            counts["dropped"] += 1
            continue
        if candidate in raw_history:
            add(candidate)
            counts["exact"] += 1
            continue

        hint_steps = list(dict.fromkeys(int(step) for step in _HINT_STEP_RE.findall(candidate)))
        recovered = [raw_steps[step] for step in hint_steps if step in raw_steps]
        if recovered:
            for block in recovered:
                add(_bounded_excerpt(block))
            counts["hint_step"] += 1
            continue

        normalization_candidates = [candidate]
        if "\\n" in candidate or "\\t" in candidate or "\\r" in candidate:
            decoded_whitespace = candidate.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
            if decoded_whitespace != candidate:
                normalization_candidates.append(decoded_whitespace)
        normalized = None
        for normalization_candidate in normalization_candidates:
            if normalization_candidate in raw_history:
                normalized = normalization_candidate
                break
            normalized = _whitespace_match(
                normalization_candidate,
                raw_history,
                normalized_history,
                history_offsets,
            )
            if normalized is not None:
                break
        if normalized is not None:
            add(normalized)
            counts["normalized"] += 1
            continue

        anchored = _anchor_match(candidate, raw_steps)
        if anchored is not None:
            add(anchored)
            counts["anchor"] += 1
            continue

        counts["dropped"] += 1

    return GroundingResult(
        quotes=grounded,
        input_count=len(submitted),
        exact_count=counts["exact"],
        normalized_count=counts["normalized"],
        hint_step_count=counts["hint_step"],
        anchor_count=counts["anchor"],
        fuzzy_count=counts["fuzzy"],
        dropped_count=counts["dropped"],
    )
