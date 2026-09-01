from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, TypeVar
from uuid import uuid4

from mlebench_subversion.monitoring.model_compat import (
    begin_monitor_model_telemetry,
    end_monitor_model_telemetry,
    monitor_model_telemetry,
)


MAX_OPENAI_API_CALLS = 200
DEFAULT_MONITOR_SAMPLE_ATTEMPTS = 3
DEFAULT_MONITOR_PREFLIGHT_SAMPLES = 10
DEFAULT_MONITOR_SYSTEMIC_FAILURE_LIMIT = 1_000_000
_T = TypeVar("_T")
_MAX_HUMAN_ARTIFACT_BASENAME_CHARS = 160
_MAX_EVAL_ARTIFACT_BASENAME_CHARS = 220
def resume_output_path(default_path: Path) -> Path:
    """Use the stable output name, or discover one legacy hashed stage artifact."""

    if default_path.exists():
        return default_path
    candidates = sorted(default_path.parent.glob("*.eval")) if default_path.parent.is_dir() else []
    if len(candidates) > 1:
        raise ValueError(
            "Resume found multiple candidate stage artifacts in "
            f"{default_path.parent}: {[path.name for path in candidates]}"
        )
    return candidates[0] if candidates else default_path


def _identity_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _identity_jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_identity_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _identity_jsonable(model_dump(mode="json"))
    return str(value)


def monitor_sample_identity(sample: Any) -> tuple[str, Any]:
    """Cheap resume identity; content hashes are intentionally not computed."""

    return str(getattr(sample, "id", "")), getattr(sample, "epoch", None)


def validate_concurrency(value: int) -> int:
    if not 1 <= value <= MAX_OPENAI_API_CALLS:
        raise ValueError(f"Concurrency must be between 1 and {MAX_OPENAI_API_CALLS}, got {value}.")
    return value


def _monitor_integer_setting(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def monitor_sample_attempt_count() -> int:
    return _monitor_integer_setting(
        "MONITOR_SAMPLE_ATTEMPTS",
        DEFAULT_MONITOR_SAMPLE_ATTEMPTS,
        minimum=1,
    )


def monitor_preflight_sample_count() -> int:
    return _monitor_integer_setting(
        "MONITOR_PREFLIGHT_SAMPLES",
        DEFAULT_MONITOR_PREFLIGHT_SAMPLES,
        minimum=0,
    )


def monitor_systemic_failure_limit() -> int:
    return _monitor_integer_setting(
        "MONITOR_SYSTEMIC_FAILURE_LIMIT",
        DEFAULT_MONITOR_SYSTEMIC_FAILURE_LIMIT,
        minimum=1,
    )


@dataclass(frozen=True)
class MonitorFailurePolicy:
    code: str
    max_attempts: int
    systemic_candidate: bool
    abort_stage: bool = False


class MonitorValidationError(ValueError):
    """Typed model-output validation failure with bounded diagnostic context."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str,
        max_attempts: int = DEFAULT_MONITOR_SAMPLE_ATTEMPTS,
        diagnostic: dict[str, Any] | None = None,
        fallback_result: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.max_attempts = max_attempts
        self.diagnostic = dict(diagnostic or {})
        self.fallback_result = fallback_result


class MonitorOutputLimitError(RuntimeError):
    """A response exhausted its output/time allowance; one retry is permitted."""

    def __init__(self, message: str, *, diagnostic: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.failure_code = "model_output_limit"
        self.max_attempts = 2
        self.diagnostic = dict(diagnostic or {})


class SystemicMonitorFailure(RuntimeError):
    """A deterministic or repeated cross-sample failure tripped the stage circuit."""


def _exception_chain(exc: Exception) -> list[Exception]:
    chain: list[Exception] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def classify_monitor_failure(exc: Exception) -> MonitorFailurePolicy:
    chain = _exception_chain(exc)
    for item in chain:
        failure_code = getattr(item, "failure_code", None)
        max_attempts = getattr(item, "max_attempts", None)
        if failure_code and max_attempts:
            return MonitorFailurePolicy(
                code=str(failure_code),
                max_attempts=max(1, int(max_attempts)),
                systemic_candidate=True,
            )

    combined = " | ".join(str(item) for item in chain).lower()
    deterministic_markers = (
        "unresolved placeholder",
        "reviewed prompt document",
        "reviewed prompt template",
        "no reviewed prompt document",
        "unknown reviewed prompt stage",
        "unexpected forced-tool declaration",
        "reviewed prompt tool parameters",
        "removed test-score field",
        "cannot be combined",
        "does not attest the current",
        "different source/protocol",
        "existing output sample identities do not match",
        "multiple existing scores match",
        "exactly one canonical extractor score",
        "unknown output_mode",
        "is frozen for model",
        "requires evaluator_prompt_profile",
    )
    if any(marker in combined for marker in deterministic_markers):
        return MonitorFailurePolicy("configuration_or_protocol", 1, True, abort_stage=True)
    if any(isinstance(item, (TimeoutError, asyncio.TimeoutError)) for item in chain):
        return MonitorFailurePolicy("sample_timeout", 2, True)
    if "model length" in combined or "length limit" in combined or "max_tokens" in combined:
        return MonitorFailurePolicy("model_output_limit", 2, True)
    if "evidence_analysis did not ground" in combined or "evidence unit" in combined:
        return MonitorFailurePolicy("evidence_unit_coverage", 3, True)
    if "tool arguments were rejected" in combined or "required submit_score tool" in combined:
        return MonitorFailurePolicy("model_tool_protocol", 3, True)
    if "invalid suspicion score" in combined or "non-substantive" in combined:
        return MonitorFailurePolicy("model_output_validation", 3, True)
    if any(
        marker in combined
        for marker in ("rate limit", "connection", "transport", "temporar", "service unavailable")
    ):
        return MonitorFailurePolicy("transient_api", 3, False)
    normalized = re.sub(r"\bsample\s+[^\s,;:]+", "sample <id>", combined)
    normalized = re.sub(r"(?<![a-z0-9])/(?:[^\s,;:]+)", "<path>", normalized)
    normalized = re.sub(r"\b[0-9a-f]{12,}\b", "<hex>", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", normalized)
    signature = hashlib.sha256(normalized[:2000].encode("utf-8")).hexdigest()[:12]
    if isinstance(exc, ValueError):
        return MonitorFailurePolicy(f"model_or_sample_validation:{signature}", 3, True)
    return MonitorFailurePolicy(f"monitor_runtime:{signature}", 3, True)


def _bounded_diagnostic(value: Any, *, limit: int = 12000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...[truncated]"
    if isinstance(value, dict):
        return {str(key): _bounded_diagnostic(item, limit=limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bounded_diagnostic(item, limit=limit) for item in value[:100]]
    return _bounded_diagnostic(repr(value), limit=limit)


class MonitorFailureCircuitBreaker:
    def __init__(
        self,
        *,
        stage: str,
        diagnostic_path: Path,
        distinct_failure_limit: int | None = None,
    ) -> None:
        self.stage = stage
        self.diagnostic_path = diagnostic_path
        self.attempt_history_path = diagnostic_path.with_name(
            f"{diagnostic_path.stem}_attempts.jsonl"
        )
        self.invocation_id = str(uuid4())
        self.distinct_failure_limit = (
            monitor_systemic_failure_limit()
            if distinct_failure_limit is None
            else max(1, int(distinct_failure_limit))
        )
        self.phase = "preflight"
        self.records: list[dict[str, Any]] = []
        self._recorded_model_calls_by_sample: dict[str, int] = {}
        self._terminal_samples_by_code: dict[str, set[str]] = {}
        self._tripped_reason: str | None = None
        self._lock = asyncio.Lock()

    @property
    def tripped(self) -> bool:
        return self._tripped_reason is not None

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def as_exception(self) -> SystemicMonitorFailure:
        return SystemicMonitorFailure(self._tripped_reason or f"{self.stage} failure circuit tripped")

    async def raise_if_tripped(self) -> None:
        if self.tripped:
            raise self.as_exception()

    async def record_failure(
        self,
        *,
        sample_key: str,
        exc: Exception,
        attempt: int,
        terminal: bool,
    ) -> MonitorFailurePolicy:
        policy = classify_monitor_failure(exc)
        async with self._lock:
            chain = _exception_chain(exc)
            diagnostic: dict[str, Any] = {}
            for item in chain:
                diagnostic.update(dict(getattr(item, "diagnostic", {}) or {}))
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "invocation_id": self.invocation_id,
                "stage": self.stage,
                "phase": self.phase,
                "sample_key": sample_key,
                "attempt": attempt,
                "terminal": terminal,
                "failure_code": policy.code,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "exception_chain": [
                    {"type": type(item).__name__, "message": str(item)} for item in chain
                ],
                "diagnostic": _bounded_diagnostic(diagnostic),
            }
            telemetry = monitor_model_telemetry()
            cumulative_calls = list(telemetry.get("calls") or [])
            previous_call_count = self._recorded_model_calls_by_sample.get(sample_key, 0)
            attempt_calls = cumulative_calls[previous_call_count:]
            self._recorded_model_calls_by_sample[sample_key] = len(cumulative_calls)
            token_fields = (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "reasoning_tokens",
                "input_tokens_cache_read",
                "input_tokens_cache_write",
            )
            record["generation_count"] = len(attempt_calls)
            record["attempt_usage"] = {
                field: sum(int(call.get(field, 0) or 0) for call in attempt_calls)
                for field in token_fields
            }
            # These are complete, untruncated provider/model records for only
            # the calls newly made by this retry attempt. The JSONL ledger is
            # append-only across separate resume invocations.
            record["attempt_model_calls"] = attempt_calls
            # ``attempt_model_calls`` already preserves every complete call
            # newly made by this retry. Do not also attach the cumulative
            # trajectory telemetry to terminal records: doing so duplicates
            # earlier attempts and can make an append-only failure ledger grow
            # quadratically when provider events contain the full prompt.
            append_jsonl(self.attempt_history_path, record)
            # Keep the replaceable operator-facing snapshot compact. Complete
            # call payloads live in the durable ledger; repeatedly rewriting
            # all of them would make later failures progressively expensive.
            summary_record = dict(record)
            summary_record.pop("attempt_model_calls", None)
            summary_record.pop("monitor_model_telemetry", None)
            summary_record["attempt_history_recorded"] = True
            self.records.append(summary_record)
            if terminal and policy.systemic_candidate:
                affected = self._terminal_samples_by_code.setdefault(policy.code, set())
                affected.add(sample_key)
                if self.phase == "preflight":
                    self._tripped_reason = (
                        f"{self.stage} preflight failed: failure class {policy.code!r} exhausted its bounded "
                        f"attempts for {sample_key}. The full batch was not started. See {self.diagnostic_path}."
                    )
                elif len(affected) >= self.distinct_failure_limit:
                    self._tripped_reason = (
                        f"{self.stage} fail-fast circuit opened: failure class {policy.code!r} exhausted retries "
                        f"for {len(affected)} distinct trajectories (limit={self.distinct_failure_limit}). "
                        f"See {self.diagnostic_path}."
                    )
            if policy.abort_stage:
                self._tripped_reason = (
                    f"{self.stage} fail-fast circuit opened on deterministic failure class {policy.code!r} "
                    f"for {sample_key}. See {self.diagnostic_path}."
                )
            atomic_write_json(
                self.diagnostic_path,
                {
                    "stage": self.stage,
                    "phase": self.phase,
                    "distinct_failure_limit": self.distinct_failure_limit,
                    "tripped": self.tripped,
                    "tripped_reason": self._tripped_reason,
                    "attempt_history_path": str(self.attempt_history_path),
                    "records": self.records,
                },
            )
            if self.tripped:
                raise self.as_exception() from exc
        return policy


async def run_with_retries(
    worker: Callable[[], Awaitable[_T]],
    *,
    attempts: int | None = None,
    on_retry: Callable[[int, int, Exception], None] | None = None,
    sample_key: str = "unknown-sample",
    failure_guard: MonitorFailureCircuitBreaker | None = None,
) -> _T:
    """Retry a true raised failure without interpreting model analysis text."""

    attempt_limit = monitor_sample_attempt_count() if attempts is None else int(attempts)
    if attempt_limit < 1:
        raise ValueError("Retry attempts must be at least 1.")
    telemetry_tokens = begin_monitor_model_telemetry()
    try:
        for attempt in range(1, attempt_limit + 1):
            try:
                return await worker()
            except asyncio.CancelledError:
                raise
            except SystemicMonitorFailure:
                # Preserve the original circuit-break reason and diagnostic. An
                # outer retry layer must never reclassify or re-record it.
                raise
            except Exception as exc:
                policy = classify_monitor_failure(exc)
                effective_limit = min(attempt_limit, policy.max_attempts)
                terminal = attempt >= effective_limit
                if failure_guard is not None:
                    await failure_guard.record_failure(
                        sample_key=sample_key,
                        exc=exc,
                        attempt=attempt,
                        terminal=terminal,
                    )
                if terminal:
                    raise
                if on_retry is not None:
                    on_retry(attempt, effective_limit, exc)
        raise AssertionError("retry loop did not return or raise")
    finally:
        end_monitor_model_telemetry(telemetry_tokens)


async def map_concurrently(
    items: Iterable[_T],
    worker: Callable[[_T], Awaitable[Any]],
    *,
    limit: int,
    attempts: int | None = None,
    failure_guard: MonitorFailureCircuitBreaker | None = None,
) -> list[Any]:
    semaphore = asyncio.Semaphore(validate_concurrency(limit))
    success_marker = object()

    async def bounded(item: _T) -> Any:
        if failure_guard is not None:
            await failure_guard.raise_if_tripped()
        async with semaphore:
            if failure_guard is not None:
                await failure_guard.raise_if_tripped()
            return success_marker, await run_with_retries(
                lambda: worker(item),
                attempts=attempts,
                sample_key=str(item),
                failure_guard=failure_guard,
            )

    tasks = [asyncio.create_task(bounded(item)) for item in items]
    pending = set(tasks)
    while pending:
        _done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if failure_guard is not None and failure_guard.tripped:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise failure_guard.as_exception()
    gathered: list[Any] = []
    for task in tasks:
        if task.cancelled():
            gathered.append(asyncio.CancelledError())
            continue
        task_error = task.exception()
        gathered.append(task_error if task_error is not None else task.result())
    failures = [result for result in gathered if isinstance(result, BaseException)]
    if failures:
        first = failures[0]
        raise RuntimeError(
            f"{len(failures)} concurrent monitor sample(s) failed; completed samples were preserved for resume. "
            f"First failure: {type(first).__name__}: {first}"
        ) from first
    return [result[1] for result in gathered]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_value = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_value)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    """Durably append one record without replacing history from earlier resumes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def bounded_human_artifact_name(
    output_stem: str,
    safe_sample_id: str,
    stage: str,
    suffix: str,
) -> str:
    """Keep log filenames readable while leaving room for atomic-write temp suffixes."""
    base = f"{output_stem}__{safe_sample_id}__{stage}"
    if len(base) <= _MAX_HUMAN_ARTIFACT_BASENAME_CHARS:
        return f"{base}{suffix}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    fixed = f"__{digest}__{safe_sample_id[-24:]}__{stage}"
    readable_chars = max(24, _MAX_HUMAN_ARTIFACT_BASENAME_CHARS - len(fixed))
    return f"{output_stem[:readable_chars]}{fixed}{suffix}"


def bounded_eval_artifact_name(input_stem: str, score_key: str) -> str:
    """Build a stable, readable stage filename without content fingerprints."""

    full_name = f"{input_stem}__{score_key}.eval"
    if len(full_name) <= _MAX_EVAL_ARTIFACT_BASENAME_CHARS:
        return full_name
    fixed = "__truncated.eval"
    readable_budget = _MAX_EVAL_ARTIFACT_BASENAME_CHARS - len(fixed) - 2
    input_chars = min(len(input_stem), readable_budget // 2)
    score_chars = readable_budget - input_chars
    return f"{input_stem[:input_chars]}__{score_key[:score_chars]}{fixed}"


def write_human_monitor_log(
    *,
    output_eval_path: Path,
    source_eval_path: Path,
    sample: Any,
    score_key: str,
    score_obj: Any,
    stage: str,
) -> Path:
    metadata = dict(getattr(score_obj, "metadata", {}) or {})
    transcript = str(metadata.pop("_monitor_transcript", ""))
    upstream_transcript = str(metadata.pop("_upstream_monitor_transcript", ""))
    system_prompt = str(metadata.pop("_monitor_system_prompt", ""))
    model_calls = list(metadata.pop("monitor_model_calls", []) or [])
    sample_id = str(getattr(sample, "id", "UNKNOWN"))
    safe_sample_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    report_path = output_eval_path.parent / "human_logs" / bounded_human_artifact_name(
        output_eval_path.stem, safe_sample_id, stage, ".md"
    )
    transcript_path = output_eval_path.parent / "transcripts" / bounded_human_artifact_name(
        output_eval_path.stem, safe_sample_id, stage, ".txt"
    )
    model_calls_path = output_eval_path.parent / "model_calls" / bounded_human_artifact_name(
        output_eval_path.stem, safe_sample_id, stage, ".json"
    )
    if transcript:
        atomic_write_text(transcript_path, transcript)
    if model_calls:
        atomic_write_json(
            model_calls_path,
            {
                "sample_id": sample_id,
                "stage": stage,
                "score_key": score_key,
                "requested_model": metadata.get("monitor_requested_model"),
                "canonical_model": metadata.get("monitor_model"),
                "usage": metadata.get("monitor_usage", {}),
                "calls": model_calls,
            },
        )
        metadata["monitor_model_calls_path"] = str(model_calls_path)

    timing_rows = []
    for key, value in sorted(metadata.items()):
        if key.startswith("timing_") and key.endswith("_seconds"):
            label = key.removeprefix("timing_").removesuffix("_seconds").replace("_", " ").title()
            try:
                rendered = f"{float(value):.3f}"
            except (TypeError, ValueError):
                rendered = str(value)
            timing_rows.append(f"| {label} | {rendered} |")
    if not timing_rows:
        timing_rows.append("| No timing data recorded | N/A |")

    value = getattr(score_obj, "value", None)
    explanation = getattr(score_obj, "explanation", None) or ""
    reasoning = metadata.get("monitor_reasoning") or ""
    lines = [
        "# Monitoring Run",
        "",
        f"- Sample: `{sample_id}`",
        f"- Stage: `{stage}`",
        f"- Score key: `{score_key}`",
        f"- Monitor score: `{value}`",
        f"- Source eval: `{source_eval_path}`",
        f"- Output eval: `{output_eval_path}`",
        "",
        "## Summary",
        "",
        str(explanation) or "No summary was returned.",
        "",
        "## Monitor Reasoning",
        "",
        str(reasoning) or "No separate reasoning was returned.",
        "",
        "## Timings",
        "",
        "| Step | Seconds |",
        "| --- | ---: |",
        *timing_rows,
        "",
        "## Transcript Provided To Monitor",
        "",
        "````text",
        transcript or "No transcript was captured.",
        "````",
        "",
    ]
    if upstream_transcript:
        lines.extend(
            [
                "## Upstream Hint-Augmented Transcript",
                "",
                "````text",
                upstream_transcript,
                "````",
                "",
            ]
        )
    lines.extend([
        "## System Prompt",
        "",
        "````text",
        system_prompt or "No system prompt was captured.",
        "````",
        "",
        "## Structured Metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, ensure_ascii=True, default=str),
        "```",
        "",
    ])
    atomic_write_text(report_path, "\n".join(lines))

    # Large prompt material belongs in the human report, not duplicated in .eval metadata.
    score_obj.metadata.pop("_monitor_transcript", None)
    score_obj.metadata.pop("_upstream_monitor_transcript", None)
    score_obj.metadata.pop("_monitor_system_prompt", None)
    score_obj.metadata.pop("monitor_model_calls", None)
    score_obj.metadata["human_log_path"] = str(report_path)
    if transcript:
        score_obj.metadata["monitor_transcript_path"] = str(transcript_path)
    if model_calls:
        score_obj.metadata["monitor_model_calls_path"] = str(model_calls_path)
    return report_path
