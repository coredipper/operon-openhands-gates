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
# Key the critic writes into CriticResult.metadata alongside the theorem
# name; equals ``critical_duration`` by construction. Kept as a constant
# so the collector and the critic agree on the contract and a single
# string-rename lands in one place on both sides.
_CERTIFICATE_EVIDENCE_N_KEY = "cert_evidence_n"


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
            out["cert_evidence_n"] = cert.get(_CERTIFICATE_EVIDENCE_N_KEY)
        else:
            out["certificate_theorem"] = None
            out["certificate_source"] = None
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


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = sum(1 for r in results if r["eval_status"] == "resolved")
    unresolved = sum(1 for r in results if r["eval_status"] == "unresolved")
    empty = sum(1 for r in results if r["eval_status"] == "empty_patch")
    errors = sum(1 for r in results if r["eval_status"] == "error")
    not_eval = sum(1 for r in results if r["eval_status"] == "not_evaluated")
    n = len(results)
    mean_turns = sum(r["n_turns"] for r in results) / n if n else 0.0
    total_tokens = sum(r["total_tokens"] or 0 for r in results)
    summary: dict[str, Any] = {
        "n": n,
        "resolved": resolved,
        "unresolved": unresolved,
        "empty_patch": empty,
        "error": errors,
        "not_evaluated": not_eval,
        "mean_turns": round(mean_turns, 2),
        "total_tokens": total_tokens,
    }
    # ``pass_at_1`` requires the SWE-bench patch-evaluation step to have
    # populated ``test_result.resolved`` on every row. If any row is
    # still ``not_evaluated``, emit ``null`` instead of a divide-by-n
    # that silently reports 0.0 — roborev #836 Medium: an unevaluated
    # run should not be presented as a real pass@1 of 0. Callers that
    # want to force-compute anyway should run eval first.
    if not_eval > 0:
        summary["pass_at_1"] = None
        summary["pass_at_1_note"] = (
            f"SWE-bench patch-evaluation not run ({not_eval}/{n} rows "
            "have ``resolved: None``); treat as unknown, not 0."
        )
    else:
        summary["pass_at_1"] = round(resolved / n, 4) if n else 0.0
    # Certificate rollup is treatment-only; present only when the rows
    # actually carry the key. NOTE: the underlying
    # ``OperonStagnationCritic.metadata`` is currently NOT serialized
    # by the openhands-sdk into ``history`` events — see
    # ``eval/results/swebench_lite_delta.md`` caveat 3. Until that gap
    # is closed (SDK patch or side-channel log), ``_scan_certificate``
    # will return ``None`` for every real run and this rollup reports
    # 0. Useful for unit tests (which inject synthetic history), not
    # for real treatment runs — use ``scripts/generate_delta_artifact.py``
    # for those, which infers critic firing from Attempt-2 retries.
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

    Also rejects duplicate ``instance_id`` rows within either condition
    independently: ``baseline=[a,a,b]`` vs ``treatment=[a,b,b]`` would
    pass a set/length comparison but double-count instances in the
    aggregates. Per-condition uniqueness is checked first so the failure
    names the offending condition + duplicate IDs.
    """
    _reject_duplicates("baseline", baseline)
    _reject_duplicates("treatment", treatment)

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


def _reject_duplicates(condition: str, records: list[dict[str, Any]]) -> None:
    """Fail if any ``instance_id`` appears more than once in ``records``.

    The per-condition check runs *before* the cross-condition set
    comparison so the error names the offending condition rather than
    a generic "row-count mismatch" that could leave the user guessing
    which side to inspect.
    """
    from collections import Counter

    ids = [r.get("instance_id") for r in records]
    counts = Counter(ids)
    dups = sorted(iid for iid, c in counts.items() if c > 1 and iid is not None)
    if dups:
        lines = [
            f"duplicate instance_id(s) in {condition} ({len(dups)}):",
            *(f"  {iid} ({counts[iid]}x)" for iid in dups),
        ]
        raise ValueError("\n".join(lines))


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
