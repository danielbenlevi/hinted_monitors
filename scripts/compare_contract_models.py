from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import task_contract  # noqa: E402


DEFAULT_MODELS = (
    "openai/gpt-5.4-mini",
    "openrouter/anthropic/claude-haiku-4.5",
    "openrouter/anthropic/claude-sonnet-5",
    "google/gemini-3.6-flash",
)
SEED = 20260811


@dataclass(frozen=True)
class TaskRef:
    split: str
    trajectory_id: str
    task_id: str
    task_path: Path
    task_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare contract models with the final GPT cache.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--count-per-model", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "runs" / "contract_model_comparison_10x4"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_text(path: Path) -> str:
    return task_contract.normalize_identity_text(path.read_text(encoding="utf-8"))


def make_ref(split: str, row: dict[str, Any], root: Path) -> TaskRef:
    trajectory_id = str(row["sample_uuid"])
    path = root / trajectory_id / "task_description.md"
    text = task_text(path)
    return TaskRef(
        split=split,
        trajectory_id=trajectory_id,
        task_id=str(row["task_id"]),
        task_path=path,
        task_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def choose_tasks(count: int) -> list[TaskRef]:
    if count < 2:
        raise ValueError("--count-per-model must be at least 2")
    benign_root = REPO_ROOT / "data" / "readable" / "benign_final"
    malicious_root = REPO_ROOT / "data" / "readable" / "malicious"
    benign_rows = load_jsonl(benign_root / "examples.jsonl")
    malicious_rows = load_jsonl(malicious_root / "examples.jsonl")
    dev_manifest = json.loads(
        (
            REPO_ROOT
            / "runs"
            / "quick_monitor_eval"
            / "gpt41mini_human_c12_full_v2"
            / "subset_inputs"
            / "malicious"
            / "subset_manifest.json"
        ).read_text(encoding="utf-8")
    )
    dev_ids = {Path(row["source_eval"]).parent.name for row in dev_manifest["files"]}
    malicious_rows = [
        row
        for row in malicious_rows
        if row.get("source_group") == "Refined" and str(row["sample_uuid"]) not in dev_ids
    ]

    def unique(split: str, rows: list[dict[str, Any]], root: Path) -> list[TaskRef]:
        refs: dict[str, TaskRef] = {}
        for row in rows:
            ref = make_ref(split, row, root)
            refs.setdefault(ref.task_sha256, ref)
        return list(refs.values())

    rng = random.Random(SEED)
    benign = unique("benign", benign_rows, benign_root)
    malicious = unique("malicious", malicious_rows, malicious_root)
    rng.shuffle(benign)
    rng.shuffle(malicious)
    benign_count = count // 2
    selected = benign[:benign_count]
    seen = {ref.task_sha256 for ref in selected}
    selected.extend(ref for ref in malicious if ref.task_sha256 not in seen)
    selected = selected[:count]
    if len(selected) != count:
        raise ValueError(f"Only found {len(selected)} unique matched tasks.")
    return selected


def load_gpt_contract(ref: TaskRef) -> dict[str, Any]:
    text = task_text(ref.task_path)
    identity = task_contract.contract_cache_identity(text, "gpt-4.1-mini")
    path = task_contract.contract_cache_path(text, "gpt-4.1-mini", identity=identity)
    cached = task_contract.load_cached_contract(path, expected_identity=identity)
    if cached is None:
        raise FileNotFoundError(f"Missing final GPT contract for {ref.task_id}: {path}")
    return task_contract.normalize_contract(cached, text, source="final-gpt-cache")


def operation_types(contract: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(row.get("op") or ""), str(row.get("domain") or ""))
        for row in contract.get("allowed_operations", [])
    }


def operation_objects(contract: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (
            str(row.get("op") or ""),
            str(row.get("domain") or ""),
            str(row.get("object") or "").casefold(),
        )
        for row in contract.get("allowed_operations", [])
    }


def symbols(contract: dict[str, Any]) -> set[tuple[str, str]]:
    values: set[tuple[str, str]] = set()
    for operation in contract.get("allowed_operations", []):
        for kind, items in (operation.get("symbols") or {}).items():
            values.update((str(kind), str(item)) for item in items or [])
    return values


def forbidden_types(contract: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(row.get("action") or ""), str(row.get("domain") or ""))
        for row in contract.get("forbidden_or_requires_justification", [])
        if not str(row.get("id") or "").startswith("deny_unlisted_")
        and not str(row.get("id") or "").startswith("deny_security_")
        and not str(row.get("id") or "").startswith("deny_out_of_scope_")
        and not str(row.get("id") or "").startswith("deny_test_")
    }


def prf(candidate: set[Any], reference: set[Any]) -> dict[str, float | int]:
    overlap = len(candidate & reference)
    precision = overlap / len(candidate) if candidate else float(not reference)
    recall = overlap / len(reference) if reference else float(not candidate)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "candidate": len(candidate),
        "reference": len(reference),
        "overlap": overlap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compare(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation_types": prf(operation_types(candidate), operation_types(reference)),
        "operation_objects": prf(operation_objects(candidate), operation_objects(reference)),
        "symbols": prf(symbols(candidate), symbols(reference)),
        "forbidden_types": prf(forbidden_types(candidate), forbidden_types(reference)),
        "candidate_operation_count": len(candidate.get("allowed_operations", [])),
        "reference_operation_count": len(reference.get("allowed_operations", [])),
    }


def slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "_")


def generate_one(
    model: str,
    ref: TaskRef,
    gpt_contract: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        contract = task_contract.create_task_contract(task_text(ref.task_path), model=model)
    except Exception as exc:
        return {
            "model": model,
            "split": ref.split,
            "task_id": ref.task_id,
            "trajectory_id": ref.trajectory_id,
            "task_sha256": ref.task_sha256,
            "success": False,
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    metadata = contract.get("metadata") or {}
    return {
        "model": model,
        "split": ref.split,
        "task_id": ref.task_id,
        "trajectory_id": ref.trajectory_id,
        "task_sha256": ref.task_sha256,
        "success": True,
        "seconds": time.perf_counter() - started,
        "semantic_attempts": metadata.get("generation_semantic_attempts"),
        "transport_attempts": metadata.get("generation_attempts"),
        "generation_seconds": metadata.get("generation_seconds"),
        "input_tokens": metadata.get("generation_input_tokens"),
        "output_tokens": metadata.get("generation_output_tokens"),
        "reasoning_tokens": metadata.get("generation_reasoning_tokens"),
        "resolved_model": metadata.get("generation_resolved_model"),
        "comparison": compare(contract, gpt_contract),
        "contract": contract,
        "gpt_contract": gpt_contract,
    }


def average(rows: list[dict[str, Any]], *path: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return statistics.mean(values) if values else None


def summarize(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["success"]]
    return {
        "model": model,
        "count": len(rows),
        "success_count": len(successes),
        "failure_count": len(rows) - len(successes),
        "first_attempt_success_count": sum(row.get("semantic_attempts") == 1 for row in successes),
        "mean_generation_seconds": average(successes, "generation_seconds"),
        "mean_input_tokens": average(successes, "input_tokens"),
        "mean_output_tokens": average(successes, "output_tokens"),
        "mean_reasoning_tokens": average(successes, "reasoning_tokens"),
        "mean_operation_type_precision_vs_gpt": average(
            successes, "comparison", "operation_types", "precision"
        ),
        "mean_operation_type_recall_vs_gpt": average(
            successes, "comparison", "operation_types", "recall"
        ),
        "mean_operation_type_f1_vs_gpt": average(
            successes, "comparison", "operation_types", "f1"
        ),
        "mean_operation_object_f1_vs_gpt": average(
            successes, "comparison", "operation_objects", "f1"
        ),
        "mean_symbol_precision_vs_gpt": average(
            successes, "comparison", "symbols", "precision"
        ),
        "mean_symbol_recall_vs_gpt": average(successes, "comparison", "symbols", "recall"),
        "mean_symbol_f1_vs_gpt": average(successes, "comparison", "symbols", "f1"),
        "mean_forbidden_type_f1_vs_gpt": average(
            successes, "comparison", "forbidden_types", "f1"
        ),
        "mean_candidate_operation_count": average(
            successes, "comparison", "candidate_operation_count"
        ),
        "mean_gpt_operation_count": average(
            successes, "comparison", "reference_operation_count"
        ),
        "errors": [row["error"] for row in rows if not row["success"]],
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.concurrency <= 10:
        raise ValueError("--concurrency must be between 1 and 10")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be positive")
    os.environ.pop("TASK_CONTRACT_REQUIRE_CACHE_HIT", None)
    os.environ.pop("TASK_CONTRACT_REASONING_EFFORT", None)
    os.environ.pop("TASK_CONTRACT_REASONING_TOKENS", None)
    os.environ["TASK_CONTRACT_MODEL_SNAPSHOT"] = "gpt-4.1-mini"
    os.environ["TASK_CONTRACT_CACHE_DIR"] = str(REPO_ROOT / "runs" / "final_run" / "contract_cache")
    refs = choose_tasks(args.count_per_model)
    gpt_contracts = {ref.task_sha256: load_gpt_contract(ref) for ref in refs}
    if args.max_output_tokens is not None:
        task_contract.CONTRACT_MAX_OUTPUT_TOKENS = args.max_output_tokens
    (output_dir / "selection.json").write_text(
        json.dumps(
            [
                {
                    **ref.__dict__,
                    "task_path": str(ref.task_path),
                }
                for ref in refs
            ],
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    summaries: list[dict[str, Any]] = []
    for model in args.models:
        task_contract.resolve_contract_model_spec(model)
        model_dir = output_dir / slug(model)
        os.environ["TASK_CONTRACT_CACHE_DIR"] = str(model_dir / "cache")
        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(generate_one, model, ref, gpt_contracts[ref.task_sha256]): ref
                for ref in refs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                print(
                    f"[{model}] {completed}/{len(refs)} task={row['task_id']} "
                    f"success={row['success']} seconds={row['seconds']:.2f}",
                    flush=True,
                )
        rows.sort(key=lambda row: (row["split"], row["task_id"], row["trajectory_id"]))
        (model_dir / "results.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        model_summary = summarize(model, rows)
        (model_dir / "summary.json").write_text(
            json.dumps(model_summary, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append(model_summary)
        print(json.dumps(model_summary, indent=2), flush=True)

    (output_dir / "summary.json").write_text(
        json.dumps({"models": summaries}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
