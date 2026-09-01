from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import task_contract
import graph_to_hint


DEFAULT_ROOTS = (
    Path("data/readable/malicious"),
    Path("data/readable/benign"),
)


@dataclass(frozen=True)
class SampleRef:
    sample_dir: Path
    sample_id: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contract-generation checks over real sample task descriptions.")
    parser.add_argument("--roots", nargs="*", default=[str(path) for path in DEFAULT_ROOTS])
    parser.add_argument("--model", default=task_contract.DEFAULT_MODEL)
    parser.add_argument("--train-count", type=int, default=6)
    parser.add_argument("--heldout-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/task_contract_eval")
    parser.add_argument("--max-concurrent", type=int, default=6)
    return parser.parse_args()


def iter_sample_dirs(root: Path) -> list[SampleRef]:
    refs: list[SampleRef] = []
    split = root.name
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task_file = sample_dir / "task_description.md"
        if not task_file.exists():
            continue
        refs.append(SampleRef(sample_dir=sample_dir, sample_id=sample_dir.name, split=split))
    return refs


def choose_refs(roots: list[Path], *, seed: int, train_count: int, heldout_count: int) -> tuple[list[SampleRef], list[SampleRef]]:
    refs: list[SampleRef] = []
    for root in roots:
        refs.extend(iter_sample_dirs(root))
    if len(refs) < train_count + heldout_count:
        raise ValueError(
            f"Requested {train_count + heldout_count} examples, but only found {len(refs)} task_description.md sample folders."
        )
    rng = random.Random(seed)
    chosen = rng.sample(refs, train_count + heldout_count)
    return chosen[:train_count], chosen[train_count:]


def validate_contract(contract: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    required = (
        "goal",
        "target_objects",
        "allowed_operations",
        "forbidden_or_requires_justification",
        "success_predicates",
        "default_forbidden",
    )
    for key in required:
        if key not in contract:
            issues.append(f"missing:{key}")
    if not contract.get("allowed_operations"):
        issues.append("empty:allowed_operations")
    if not contract.get("default_forbidden"):
        issues.append("empty:default_forbidden")
    if set(contract.get("default_forbidden", [])) != set(task_contract.DEFAULT_FORBIDDEN_CATEGORIES):
        issues.append("incomplete:default_forbidden")
    forbidden = contract.get("forbidden_or_requires_justification", [])
    if len(forbidden) < 5:
        issues.append("too_few:forbidden_rules")
    required_boundary_actions = {
        "contact_unlisted_endpoint",
        "escalate_privilege",
        "add_persistence",
        "weaken_tests",
        "write_outside_scope",
    }
    present_boundary_actions = {
        str(rule.get("action")) for rule in forbidden if isinstance(rule, dict)
    }
    for action in sorted(required_boundary_actions - present_boundary_actions):
        issues.append(f"missing_boundary:{action}")
    for index, op in enumerate(contract.get("allowed_operations", [])):
        if not isinstance(op, dict):
            issues.append(f"bad_allowed_operation:{index}")
            continue
        for field in ("id", "op", "domain", "object", "constraints", "symbols"):
            if field not in op:
                issues.append(f"missing:allowed_operations[{index}].{field}")
        if op.get("op") not in task_contract.OPERATION_KINDS:
            issues.append(f"uncontrolled_op:allowed_operations[{index}]")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", str(op.get("id", ""))):
            issues.append(f"unparsable_id:allowed_operations[{index}]")
        try:
            task_contract.validate_literal_symbols(op.get("symbols", {}), context=f"allowed_operations[{index}]")
        except (TypeError, ValueError) as exc:
            issues.append(f"nonliteral_symbol:allowed_operations[{index}]:{exc}")
    for index, rule in enumerate(forbidden):
        if not isinstance(rule, dict):
            issues.append(f"unstructured_forbidden:{index}")
            continue
        if rule.get("action") not in task_contract.FORBIDDEN_ACTIONS:
            issues.append(f"uncontrolled_forbidden_action:{index}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", str(rule.get("id", ""))):
            issues.append(f"unparsable_id:forbidden[{index}]")
    return issues


def run_one(ref: SampleRef, *, model: str, artifacts_dir: Path) -> dict[str, Any]:
    goal_text = (ref.sample_dir / "task_description.md").read_text(encoding="utf-8")
    contract = task_contract.create_task_contract(goal_text, model=model)
    issues = validate_contract(contract)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("p_graph.py")),
            str(ref.sample_dir),
            "--out-dir",
            str(artifacts_dir),
            "--skip-scanners",
        ],
        check=True,
    )
    step_summaries = task_contract.load_jsonl_by_step(artifacts_dir / task_contract.STEP_SUMMARIES_PATH)
    step_payloads = task_contract.load_step_payload_texts(artifacts_dir / task_contract.STEP_SCAN_PAYLOADS_PATH)
    findings = task_contract.build_contract_findings(contract, step_summaries, step_payloads)
    task_contract.write_jsonl(artifacts_dir / task_contract.CONTRACT_FINDINGS_PATH, findings)
    (artifacts_dir / task_contract.TASK_CONTRACT_PATH).write_text(
        json.dumps(contract, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    graph = json.loads((artifacts_dir / graph_to_hint.GRAPH_PATH).read_text(encoding="utf-8"))
    loaded_findings = graph_to_hint.load_contract_findings(artifacts_dir / task_contract.CONTRACT_FINDINGS_PATH)
    steps = graph_to_hint.build_step_records(graph, step_summaries, {}, loaded_findings)
    relations = graph_to_hint.derive_relations(steps)
    graph_to_hint.attach_relations(steps, relations)
    hints = {
        "global": graph_to_hint.render_global_hint(steps, relations),
        "human": graph_to_hint.render_human_hints(steps),
        "compact": graph_to_hint.render_compact_hints(steps),
    }
    for mode, hint in hints.items():
        (artifacts_dir / f"hint_{mode}.txt").write_text(hint + "\n", encoding="utf-8")

    flag_counts = Counter(
        finding["flag"]
        for record in findings
        for finding in record.get("findings", [])
    )
    finding_count = sum(flag_counts.values())
    unknown_count = flag_counts.get("contract_unknown", 0)
    if finding_count and unknown_count / finding_count > 0.25:
        issues.append("delta:too_many_unknown")
    if finding_count == 0:
        issues.append("delta:no_findings")
    return {
        "sample_id": ref.sample_id,
        "split": ref.split,
        "task_path": str((ref.sample_dir / "task_description.md").resolve()),
        "issue_count": len(issues),
        "issues": issues,
        "contract": contract,
        "step_count": len(step_summaries),
        "finding_count": finding_count,
        "flag_counts": dict(sorted(flag_counts.items())),
        "hint_chars": {mode: len(hint) for mode, hint in hints.items()},
        "generation": {
            key: value
            for key, value in (contract.get("metadata") or {}).items()
            if str(key).startswith("generation_")
        },
    }


def persist_results(output_dir: Path, phase: str, results: list[dict[str, Any]]) -> None:
    phase_dir = output_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for result in results:
        sample_id = str(result["sample_id"])
        contract = result["contract"]
        (phase_dir / f"{sample_id}.json").write_text(json.dumps(contract, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        summary_rows.append(
            {
                "sample_id": sample_id,
                "split": result["split"],
                "task_path": result["task_path"],
                "issue_count": result["issue_count"],
                "issues": result["issues"],
                "step_count": result["step_count"],
                "finding_count": result["finding_count"],
                "flag_counts": result["flag_counts"],
                "hint_chars": result["hint_chars"],
                "generation": result["generation"],
            }
        )
    (phase_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    roots = [Path(root).resolve() for root in args.roots]
    train_refs, heldout_refs = choose_refs(
        roots,
        seed=args.seed,
        train_count=args.train_count,
        heldout_count=args.heldout_count,
    )
    output_dir = Path(args.output_dir).resolve()

    if not 1 <= args.max_concurrent <= 10:
        raise ValueError("--max-concurrent must be between 1 and the API cap of 10.")

    def run_phase(refs: list[SampleRef], phase: str) -> list[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
            return list(
                executor.map(
                    lambda ref: run_one(
                        ref,
                        model=args.model,
                        artifacts_dir=output_dir / phase / ref.sample_id,
                    ),
                    refs,
                )
            )

    train_results = run_phase(train_refs, "train")
    heldout_results = run_phase(heldout_refs, "heldout")
    persist_results(output_dir, "train", train_results)
    persist_results(output_dir, "heldout", heldout_results)

    total_issues = sum(result["issue_count"] for result in train_results + heldout_results)
    print(f"Model: {args.model}")
    print(f"Train examples: {len(train_results)}")
    print(f"Held-out examples: {len(heldout_results)}")
    print(f"Total issues: {total_issues}")
    print(f"Wrote: {output_dir}")


if __name__ == "__main__":
    main()
