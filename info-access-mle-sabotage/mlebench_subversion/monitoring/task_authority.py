"""Immutable canonical-task authority for exported monitoring samples."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping


CANONICAL_TASK_TEXT_KEY = "canonical_task_text"
CANONICAL_TASK_SHA256_KEY = "canonical_task_sha256"

_POST_HOC_MAIN_TASK_ENVELOPE_RE = re.compile(
    r"\AMain task test cases:\s*\n<test_cases>.*?"
    r"\nMain task test cases results:\s*\n<main_task_results>",
    re.DOTALL,
)
_POST_HOC_SIDE_TASK_ENVELOPE_RE = re.compile(
    r"\ASide task ['\"][^'\"\n]+['\"] (?:passed|failed):\s*\n.*?"
    r"(?:SECURITY VIOLATION:|No unauthorized sudo escalation detected)",
    re.DOTALL | re.IGNORECASE,
)


def canonical_task_sha256(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def validate_canonical_task_boundary(text: str, *, sample_id: Any = "UNKNOWN") -> str:
    """Reject known benchmark grader envelopes without policing ordinary task language."""

    normalized = str(text or "").strip()
    if _POST_HOC_MAIN_TASK_ENVELOPE_RE.search(normalized) or _POST_HOC_SIDE_TASK_ENVELOPE_RE.search(
        normalized
    ):
        first_line = normalized.splitlines()[0] if normalized else ""
        raise ValueError(
            f"Post-hoc benchmark grader envelope cannot be canonical task authority for sample "
            f"{sample_id}: {first_line!r}."
        )
    return normalized


def read_canonical_task(sample: Any) -> str:
    """Read and verify embedded task authority, with legacy-path fallback."""

    metadata = getattr(sample, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    embedded = metadata.get(CANONICAL_TASK_TEXT_KEY)
    if isinstance(embedded, str) and embedded.strip():
        text = embedded.strip()
        expected = metadata.get(CANONICAL_TASK_SHA256_KEY)
        actual = canonical_task_sha256(text)
        if not isinstance(expected, str) or expected != actual:
            raise ValueError(
                f"Canonical task hash mismatch for sample {getattr(sample, 'id', 'UNKNOWN')}: "
                f"expected {expected!r}, computed {actual}."
            )
        return validate_canonical_task_boundary(text, sample_id=getattr(sample, "id", "UNKNOWN"))

    # Existing readable archives already embed the same text used to create
    # task_description.md.  Prefer those frozen bytes to reopening the mutable
    # absolute path; subset manifests independently attest their hash.
    legacy_embedded = metadata.get("main_task_description")
    if isinstance(legacy_embedded, str) and legacy_embedded.strip():
        return validate_canonical_task_boundary(
            legacy_embedded, sample_id=getattr(sample, "id", "UNKNOWN")
        )

    # Backward compatibility for previously frozen logs.  New exports always
    # embed and hash the authority, so this path is provenance-only legacy.
    path_value = metadata.get("task_description_path")
    if isinstance(path_value, str) and path_value.strip():
        return validate_canonical_task_boundary(
            Path(path_value).read_text(encoding="utf-8"),
            sample_id=getattr(sample, "id", "UNKNOWN"),
        )
    raise ValueError(
        "Missing embedded canonical task authority and legacy "
        f"metadata['task_description_path'] for sample {getattr(sample, 'id', 'UNKNOWN')}."
    )
