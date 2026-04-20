"""Aggregate baseline + treatment runs into one delta artifact.

Reads OpenHands' per-run ``output.critic_attempt_*.jsonl`` files from
both condition directories, emits a unified JSON artifact whose schema
mirrors operon's ``eval/results/swebench_phase2.json`` envelope where
applicable.

Usage::

    python scripts/collect_results.py \\
        --baseline eval/runs/baseline \\
        --treatment eval/runs/treatment \\
        --out eval/results/swebench_lite_delta.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _find_output_jsonl(run_dir: Path) -> Path:
    """Locate the benchmark runner's output JSONL in ``run_dir``.

    The benchmarks runner writes ``output.critic_attempt_{N}.jsonl`` under
    ``<output_dir>/<dataset_description>/<model>/...``. We take the first
    match we find; multi-attempt runs aren't in our scope.
    """
    candidates = sorted(run_dir.rglob("output.critic_attempt_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no output.critic_attempt_*.jsonl under {run_dir!s}")
    return candidates[0]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_result(record: dict[str, Any], condition: str) -> dict[str, Any]:
    """Project one ``EvalOutput`` record into our delta schema.

    Mirrors per-result field names from operon's
    ``eval/swebench_phase2.py::build_artifact`` where they apply.
    ``certificate_*`` fields are populated only on the treatment
    condition when stagnation fires.
    """
    test_result = record.get("test_result") or {}
    patch = test_result.get("git_patch") or ""
    history = record.get("history") or []

    eval_status = _infer_eval_status(record, patch, test_result)
    metadata = record.get("metrics") or {}
    total_tokens = metadata.get("total_tokens")

    out: dict[str, Any] = {
        "instance_id": record.get("instance_id"),
        "condition": condition,
        "eval_status": eval_status,
        "patch_extracted": bool(patch),
        "patch_size_bytes": len(patch.encode("utf-8")) if patch else 0,
        "n_turns": len(history),
        "total_tokens": total_tokens,
        "error_reason": record.get("error"),
    }

    # Certificates surface through CriticResult.metadata which the
    # benchmarks runner propagates into history events. We don't have
    # a cross-repo stable path for them yet, so leave placeholders —
    # the smoke run will confirm the exact shape.
    if condition == "operon_stagnation":
        out["certificate_emitted"] = None
        out["certificate_theorem"] = None
        out["cert_evidence_n"] = None
    return out


def _infer_eval_status(record: dict[str, Any], patch: str, test_result: dict[str, Any]) -> str:
    if record.get("error"):
        return "error"
    if not patch:
        return "empty_patch"
    # ``test_result.resolved`` is set by the SWE-bench harness pass
    # when it's been invoked. If the caller didn't run the harness,
    # we can't distinguish resolved vs unresolved; mark 'not_evaluated'
    # and let the caller fix up.
    resolved = test_result.get("resolved")
    if resolved is True:
        return "resolved"
    if resolved is False:
        return "unresolved"
    return "not_evaluated"


def _aggregate(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    resolved = sum(1 for r in results if r["eval_status"] == "resolved")
    unresolved = sum(1 for r in results if r["eval_status"] == "unresolved")
    empty = sum(1 for r in results if r["eval_status"] == "empty_patch")
    errors = sum(1 for r in results if r["eval_status"] == "error")
    not_eval = sum(1 for r in results if r["eval_status"] == "not_evaluated")
    n = len(results)
    pass_at_1 = resolved / n if n else 0.0
    mean_turns = sum(r["n_turns"] for r in results) / n if n else 0.0
    total_tokens = sum(r["total_tokens"] or 0 for r in results)
    return {
        "n": n,
        "resolved": resolved,
        "unresolved": unresolved,
        "empty_patch": empty,
        "error": errors,
        "not_evaluated": not_eval,
        "pass_at_1": round(pass_at_1, 4),
        "mean_turns": round(mean_turns, 2),
        "total_tokens": total_tokens,
    }


def build_artifact(
    baseline: Iterable[dict[str, Any]],
    treatment: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    baseline_rows = [_extract_result(r, "baseline") for r in baseline]
    treatment_rows = [_extract_result(r, "operon_stagnation") for r in treatment]

    return {
        "run_id": str(uuid.uuid4()),
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "conditions": ["baseline", "operon_stagnation"],
        "results": baseline_rows + treatment_rows,
        "summary": {
            "baseline": _aggregate(baseline_rows),
            "operon_stagnation": _aggregate(treatment_rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = _load_jsonl(_find_output_jsonl(args.baseline))
    treatment = _load_jsonl(_find_output_jsonl(args.treatment))

    artifact = build_artifact(baseline, treatment)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"output_json": str(args.out)}))


if __name__ == "__main__":
    main()
