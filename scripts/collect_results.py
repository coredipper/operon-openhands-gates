"""Aggregate baseline + treatment runs into one delta artifact.

Reads OpenHands' per-run ``output.critic_attempt_*.jsonl`` files from
both condition directories, emits a unified JSON artifact whose schema
mirrors operon's ``eval/results/swebench_phase2.json`` envelope where
applicable.

Validates that baseline and treatment contain the *same* set of
instance_ids before aggregation — a partial rerun, wrong ``--select``
file, or failed benchmark shard is a hard error, not a silent mismatch.

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
from typing import Any, cast

# Events whose ``metadata`` dict may carry the windowed-stability
# certificate emitted by ``OperonStagnationCritic``. The critic writes
# ``certificate_theorem`` keyed into the ``CriticResult.metadata``
# returned from ``evaluate()``; the benchmarks runner persists critic
# results somewhere in ``EvalOutput.history``. The exact event kind
# that carries this is not stable across benchmarks versions, so we
# do a best-effort recursive scan rather than keying on a specific
# event type — the field name is distinctive enough to avoid
# false positives.
_CERTIFICATE_KEY = "certificate_theorem"
_CERTIFICATE_SOURCE_KEY = "certificate_source"


def _find_output_jsonl(run_dir: Path) -> Path:
    """Locate the benchmark runner's output JSONL in ``run_dir``.

    Requires exactly one match. Multiple matches (e.g. from iterative
    runs that wrote multiple ``critic_attempt_N.jsonl`` files under the
    same directory, or from accidentally reusing an output dir across
    runs) is a hard error — the caller must point at the specific file
    they want, since lexicographic ordering isn't a meaningful
    "pick the freshest output" rule.
    """
    candidates = sorted(run_dir.rglob("output.critic_attempt_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no output.critic_attempt_*.jsonl under {run_dir!s}")
    if len(candidates) > 1:
        raise ValueError(
            f"ambiguous: {len(candidates)} output.critic_attempt_*.jsonl files "
            f"under {run_dir!s}:\n"
            + "\n".join(f"  {c}" for c in candidates)
            + "\n\nPoint at a more specific subdirectory, or delete stale output files."
        )
    return candidates[0]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _scan_certificate(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best-effort extraction of the windowed-stability certificate.

    Walks each event and any nested ``metadata``/``critic_result`` dicts
    looking for ``certificate_theorem``. Returns the enclosing metadata
    dict on first hit, or ``None`` if the critic never emitted. Safe to
    call on either condition — baseline runs simply won't have the key.
    """

    def _visit(node: Any) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if _CERTIFICATE_KEY in node:
                return node
            for v in node.values():
                hit = _visit(v)
                if hit is not None:
                    return hit
        elif isinstance(node, list):
            for item in node:
                hit = _visit(item)
                if hit is not None:
                    return hit
        return None

    for event in history or []:
        hit = _visit(event)
        if hit is not None:
            return hit
    return None


def _extract_result(record: dict[str, Any], condition: str) -> dict[str, Any]:
    """Project one ``EvalOutput`` record into our delta schema."""
    test_result = record.get("test_result") or {}
    patch = test_result.get("git_patch") or ""
    history = record.get("history") or []

    eval_status = _infer_eval_status(record, patch, test_result)
    metrics = record.get("metrics") or {}
    total_tokens = metrics.get("total_tokens") if isinstance(metrics, dict) else None

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

    if condition == "operon_stagnation":
        cert = _scan_certificate(history)
        out["certificate_emitted"] = cert is not None
        if cert is not None:
            out["certificate_theorem"] = cert.get(_CERTIFICATE_KEY)
            out["certificate_source"] = cert.get(_CERTIFICATE_SOURCE_KEY)
        else:
            out["certificate_theorem"] = None
            out["certificate_source"] = None
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


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = sum(1 for r in results if r["eval_status"] == "resolved")
    unresolved = sum(1 for r in results if r["eval_status"] == "unresolved")
    empty = sum(1 for r in results if r["eval_status"] == "empty_patch")
    errors = sum(1 for r in results if r["eval_status"] == "error")
    not_eval = sum(1 for r in results if r["eval_status"] == "not_evaluated")
    n = len(results)
    pass_at_1 = resolved / n if n else 0.0
    mean_turns = sum(r["n_turns"] for r in results) / n if n else 0.0
    total_tokens = sum(r["total_tokens"] or 0 for r in results)
    summary: dict[str, Any] = {
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
    # Certificate rollup is treatment-only; present only when the rows
    # actually carry the key.
    if results and "certificate_emitted" in results[0]:
        summary["certificates_emitted"] = sum(1 for r in results if r.get("certificate_emitted"))
    return summary


def _validate_matched_instances(
    baseline: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
) -> None:
    """Require identical instance_id sets between conditions.

    A partial run, wrong ``--select``, or shard failure should fail
    loudly — a summary that silently compares different task sets is
    worse than no summary. Reports missing/extra IDs on either side.
    """
    bids_raw = {r.get("instance_id") for r in baseline}
    tids_raw = {r.get("instance_id") for r in treatment}
    if None in bids_raw or None in tids_raw:
        raise ValueError("one or more records are missing ``instance_id``")
    bids = cast(set[str], bids_raw)
    tids = cast(set[str], tids_raw)
    if bids != tids:
        missing_in_treatment = sorted(bids - tids)
        missing_in_baseline = sorted(tids - bids)
        lines = [
            f"instance_id set mismatch: baseline has {len(bids)}, "
            f"treatment has {len(tids)} (intersection {len(bids & tids)})",
        ]
        if missing_in_treatment:
            lines.append(f"  missing in treatment ({len(missing_in_treatment)}):")
            lines.extend(f"    {i}" for i in missing_in_treatment)
        if missing_in_baseline:
            lines.append(f"  missing in baseline ({len(missing_in_baseline)}):")
            lines.extend(f"    {i}" for i in missing_in_baseline)
        raise ValueError("\n".join(lines))
    if len(baseline) != len(treatment):
        # Same IDs but different row counts implies a condition has
        # duplicate records. Not recoverable without caller intent.
        raise ValueError(
            f"row-count mismatch: baseline={len(baseline)}, treatment={len(treatment)} "
            f"(same instance_id set; one side has duplicates)"
        )


def build_artifact(
    baseline: Iterable[dict[str, Any]],
    treatment: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    baseline_records = list(baseline)
    treatment_records = list(treatment)
    _validate_matched_instances(baseline_records, treatment_records)

    baseline_rows = [_extract_result(r, "baseline") for r in baseline_records]
    treatment_rows = [_extract_result(r, "operon_stagnation") for r in treatment_records]

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


# Re-export for tests.
__all__ = [
    "_find_output_jsonl",
    "_scan_certificate",
    "_validate_matched_instances",
    "build_artifact",
]


if __name__ == "__main__":
    main()
