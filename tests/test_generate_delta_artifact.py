"""Regression tests for ``scripts/generate_delta_artifact.py``.

Covers roborev #837:
- ``completed_retries`` counts total retry rounds (sum of
  ``max_attempt - 1``), not just instances-with-retry. Matters when
  an instance reaches ``attempt >= 3`` — a 2-retry instance now
  contributes 2 to the count, not 1.
- ``--aborted-treatment-retry`` IDs are validated: unknown IDs and
  IDs that already have a completed retry both raise rather than
  silently double-count into the headline rejection metric.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_delta_artifact.py"
_spec = importlib.util.spec_from_file_location("_generate_delta_artifact", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
gen = importlib.util.module_from_spec(_spec)
sys.modules["_generate_delta_artifact"] = gen
_spec.loader.exec_module(gen)


# ---- Row factories ---------------------------------------------------------


def _row(instance_id: str, attempt: int = 1, cost: float = 0.1) -> dict:
    return {
        "instance_id": instance_id,
        "attempt": attempt,
        "test_result": {"git_patch": "diff --git a b\n"},
        "history": [],
        "metrics": {
            "accumulated_cost": cost,
            "accumulated_token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "reasoning_tokens": 50,
            },
        },
    }


# ---- #837 Medium 1: retry counts scale with attempt number ----------------


def test_aggregate_splits_per_instance_vs_per_round_metrics() -> None:
    """Roborev #838: per-instance rate stays in [0, 1] even when an
    instance has attempt=3 (2 retry rounds). Per-round total grows
    with retry depth; per-instance count is capped at 1 per instance.
    """
    rows = [
        _row("a", attempt=1),
        _row("b", attempt=1),
        _row("b", attempt=2),
        _row("c", attempt=1),
        _row("c", attempt=2),
        _row("c", attempt=3),
    ]
    agg = gen._aggregate(rows, set())
    # Per-round: b contributes 1 (attempt 2), c contributes 2 (attempts 2+3)
    assert agg["total_completed_retry_rounds"] == 3
    assert agg["total_retry_rounds"] == 3  # no aborts
    # Per-instance: 2 of 3 instances (b, c) had ≥1 retry; a had none.
    assert agg["instances_with_completed_retry"] == 2
    assert agg["instances_with_rejection"] == 2
    assert agg["instances_with_rejection_rate"] == round(2 / 3, 3)  # 0.667, always ≤ 1
    # Histogram captures the depth distribution
    assert agg["max_attempt_histogram"] == {1: 1, 2: 1, 3: 1}


def test_aggregate_retry_counts_coincide_when_all_max_attempt_is_2() -> None:
    """The common case: every retried instance stopped at attempt=2 —
    per-round == per-instance.
    """
    rows = [
        _row("a", attempt=1),
        _row("b", attempt=1),
        _row("b", attempt=2),
        _row("c", attempt=1),
        _row("c", attempt=2),
    ]
    agg = gen._aggregate(rows, set())
    assert agg["total_completed_retry_rounds"] == 2
    assert agg["instances_with_completed_retry"] == 2
    assert agg["instances_with_rejection"] == 2


def test_aggregate_aborted_retries_count_per_instance_and_per_round() -> None:
    """An aborted retry adds 1 to both per-instance and per-round
    (we can't observe more than one aborted round per instance since
    no row is written).
    """
    rows = [_row("a", attempt=1), _row("b", attempt=1)]
    agg = gen._aggregate(rows, {"b"})
    assert agg["total_completed_retry_rounds"] == 0
    assert agg["aborted_retries"] == 1
    assert agg["total_retry_rounds"] == 1
    assert agg["instances_with_rejection"] == 1
    assert agg["instances_with_rejection_rate"] == 0.5
    assert agg["aborted_retry_instance_ids"] == ["b"]


# ---- #837 Medium 2: aborted-retry validation ------------------------------


def test_validate_aborted_retries_rejects_unknown_id() -> None:
    rows = [_row("a", attempt=1), _row("b", attempt=1)]
    with pytest.raises(ValueError, match="not found in treatment run"):
        gen._validate_aborted_retries(rows, {"nonexistent"})


def test_validate_aborted_retries_rejects_already_completed() -> None:
    """An ID with a completed retry row (attempt > 1) cannot also be
    marked aborted — that double-counts it into both categories.
    """
    rows = [_row("a", attempt=1), _row("a", attempt=2)]
    with pytest.raises(ValueError, match="already have a completed retry"):
        gen._validate_aborted_retries(rows, {"a"})


def test_validate_aborted_retries_accepts_valid_input() -> None:
    rows = [_row("a", attempt=1), _row("b", attempt=1)]
    # No exception raised.
    gen._validate_aborted_retries(rows, {"a"})
    gen._validate_aborted_retries(rows, set())  # empty set is fine too


def test_build_artifact_propagates_validation_error() -> None:
    baseline = [_row("a", attempt=1)]
    treatment = [_row("a", attempt=1)]
    with pytest.raises(ValueError, match="not found in treatment run"):
        gen.build_artifact(baseline, treatment, {"typo_id"}, model="openai/gpt-5")


# ---- Integration: full artifact build -------------------------------------


def test_build_artifact_happy_path() -> None:
    baseline = [_row("a", attempt=1), _row("b", attempt=1)]
    treatment = [
        _row("a", attempt=1),
        _row("a", attempt=2),
        _row("b", attempt=1),
    ]
    artifact = gen.build_artifact(baseline, treatment, set(), model="openai/gpt-5")
    assert artifact["conditions"] == ["baseline", "operon_stagnation"]
    assert artifact["summary"]["baseline"]["total_completed_retry_rounds"] == 0
    assert artifact["summary"]["operon_stagnation"]["total_completed_retry_rounds"] == 1
    assert artifact["summary"]["operon_stagnation"]["instances_with_completed_retry"] == 1
    assert artifact["summary"]["operon_stagnation"]["instances_with_rejection"] == 1
    assert artifact["scope"]["n_instances"] == 2


def test_build_artifact_rejects_instance_set_mismatch() -> None:
    baseline = [_row("a", attempt=1)]
    treatment = [_row("b", attempt=1)]  # different ID
    with pytest.raises(ValueError, match="instance_id set mismatch"):
        gen.build_artifact(baseline, treatment, set(), model="openai/gpt-5")


# ---- #838 Medium: markdown renders sane numbers under attempt >= 3 --------


# ---- Eval-report ingestion (roborev plan "publish v0.1.0a2 + close resolved caveat") ----


def _eval_report(
    resolved: list[str] | None = None,
    unresolved: list[str] | None = None,
    empty_patch: list[str] | None = None,
    error: list[str] | None = None,
) -> dict:
    """Synthetic swebench .report.json-shaped dict for tests."""
    return {
        "total_instances": 300,  # full dataset; irrelevant to per-instance lookup
        "submitted_instances": (
            len(resolved or []) + len(unresolved or []) + len(empty_patch or []) + len(error or [])
        ),
        "resolved_ids": list(resolved or []),
        "unresolved_ids": list(unresolved or []),
        "empty_patch_ids": list(empty_patch or []),
        "error_ids": list(error or []),
        "incomplete_ids": [],
        "completed_ids": list(resolved or []) + list(unresolved or []),
        "submitted_ids": (
            list(resolved or [])
            + list(unresolved or [])
            + list(empty_patch or [])
            + list(error or [])
        ),
        "schema_version": 2,
    }


def test_aggregate_populates_pass_at_1_with_eval_report() -> None:
    rows = [_row("a", attempt=1), _row("b", attempt=1), _row("c", attempt=1)]
    report = _eval_report(resolved=["a"], unresolved=["b", "c"])
    # frozenset-ify matching _load_eval_report behavior:
    for k in ("resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids", "incomplete_ids"):
        report[k] = frozenset(report[k])
    agg = gen._aggregate(rows, set(), eval_report=report)
    assert agg["pass_at_1"] == round(1 / 3, 4)
    assert agg["resolved_count"] == 1
    assert agg["unresolved_count"] == 2
    assert agg["incomplete_count"] == 0
    assert "pass_at_1_note" not in agg


def test_aggregate_pass_at_1_is_null_when_row_is_incomplete() -> None:
    """Mirror collect_results.py: if any instance wasn't scored by the
    harness (not in resolved/unresolved/empty/error sets), pass_at_1
    emits None with a note."""
    rows = [_row("a", attempt=1), _row("b", attempt=1)]
    report = _eval_report(resolved=["a"])  # b is missing — harness didn't complete it
    for k in ("resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids", "incomplete_ids"):
        report[k] = frozenset(report[k])
    agg = gen._aggregate(rows, set(), eval_report=report)
    assert agg["pass_at_1"] is None
    assert agg["incomplete_count"] == 1
    assert "pass_at_1_note" in agg


def test_aggregate_without_eval_report_omits_pass_at_1() -> None:
    """Back-compat: when no eval report is supplied, pass@1 fields don't appear."""
    rows = [_row("a", attempt=1)]
    agg = gen._aggregate(rows, set())
    assert "pass_at_1" not in agg
    assert "resolved_count" not in agg


def test_build_artifact_propagates_eval_reports() -> None:
    baseline = [_row("a", attempt=1), _row("b", attempt=1)]
    treatment = [_row("a", attempt=1), _row("a", attempt=2), _row("b", attempt=1)]
    b_report = _eval_report(resolved=["a"], unresolved=["b"])
    t_report = _eval_report(resolved=["a", "b"])
    for r in (b_report, t_report):
        for k in (
            "resolved_ids",
            "unresolved_ids",
            "empty_patch_ids",
            "error_ids",
            "incomplete_ids",
        ):
            r[k] = frozenset(r[k])

    artifact = gen.build_artifact(
        baseline,
        treatment,
        set(),
        model="openai/gpt-5",
        baseline_eval_report=b_report,
        treatment_eval_report=t_report,
    )
    assert artifact["summary"]["baseline"]["pass_at_1"] == 0.5
    assert artifact["summary"]["operon_stagnation"]["pass_at_1"] == 1.0
    # Per-instance rows carry the eval status as string.
    by_iid = {p["instance_id"]: p for p in artifact["per_instance"]}
    assert by_iid["a"]["baseline_eval_status"] == "resolved"
    assert by_iid["a"]["treatment_eval_status"] == "resolved"
    assert by_iid["b"]["baseline_eval_status"] == "unresolved"
    assert by_iid["b"]["treatment_eval_status"] == "resolved"


def test_load_eval_report_roundtrips_through_disk(tmp_path: Path) -> None:
    report_path = tmp_path / "x.report.json"
    report_path.write_text(
        json.dumps(
            {
                "resolved_ids": ["a"],
                "unresolved_ids": ["b"],
                "empty_patch_ids": [],
                "error_ids": [],
                "incomplete_ids": [],
            }
        )
    )
    loaded = gen._load_eval_report(report_path)
    # ids fields should be frozensets for O(1) membership:
    assert isinstance(loaded["resolved_ids"], frozenset)
    assert "a" in loaded["resolved_ids"]
    assert "b" not in loaded["resolved_ids"]


def test_markdown_renders_pass_at_1_row_and_retry_flip_accounting() -> None:
    baseline = [_row("a", attempt=1), _row("b", attempt=1), _row("c", attempt=1)]
    # Treatment: b retried, c retried.
    treatment = [
        _row("a", attempt=1),
        _row("b", attempt=1),
        _row("b", attempt=2),
        _row("c", attempt=1),
        _row("c", attempt=2),
    ]
    # b's retry "improved" (baseline unresolved → treatment resolved);
    # c's retry "broke"   (baseline resolved   → treatment unresolved).
    b_report = _eval_report(resolved=["a", "c"], unresolved=["b"])
    t_report = _eval_report(resolved=["a", "b"], unresolved=["c"])
    for r in (b_report, t_report):
        for k in (
            "resolved_ids",
            "unresolved_ids",
            "empty_patch_ids",
            "error_ids",
            "incomplete_ids",
        ):
            r[k] = frozenset(r[k])

    artifact = gen.build_artifact(
        baseline,
        treatment,
        set(),
        model="openai/gpt-5",
        baseline_eval_report=b_report,
        treatment_eval_report=t_report,
    )
    md = gen.generate_markdown(artifact)
    # pass@1 row exists; delta is 0 pp (2 resolved on each side)
    assert "**pass@1**" in md
    assert "+0 pp" in md
    # Retry-flip accounting
    assert "**1** improved" in md  # b flipped unresolved→resolved
    assert "**1** broke" in md  # c flipped resolved→unresolved
    # Instance table has resolved columns
    assert "| Base resolved | Treat resolved |" in md
    assert "✅" in md
    assert "❌" in md


def test_build_artifact_requires_both_eval_reports_together() -> None:
    """Roborev #844 Medium 2: supplying only one side would render an
    asymmetric half-eval markdown. Fail fast instead.
    """
    baseline = [_row("a", attempt=1)]
    treatment = [_row("a", attempt=1)]
    b_report = _eval_report(resolved=["a"])
    for k in ("resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids", "incomplete_ids"):
        b_report[k] = frozenset(b_report[k])
    with pytest.raises(ValueError, match="must be supplied together"):
        gen.build_artifact(
            baseline,
            treatment,
            set(),
            model="openai/gpt-5",
            baseline_eval_report=b_report,
            treatment_eval_report=None,
        )
    with pytest.raises(ValueError, match="must be supplied together"):
        gen.build_artifact(
            baseline,
            treatment,
            set(),
            model="openai/gpt-5",
            baseline_eval_report=None,
            treatment_eval_report=b_report,
        )


def test_build_artifact_rejects_eval_report_missing_row_ids() -> None:
    """Roborev #844 Medium 1: a mistyped/stale report that doesn't
    cover every row's instance_id would silently mark rows as
    ``incomplete`` and skew pass@1. Fail fast.
    """
    baseline = [_row("a", attempt=1), _row("b", attempt=1)]
    treatment = [_row("a", attempt=1), _row("b", attempt=1)]
    # Treatment report omits "b" entirely — simulates pointing at a
    # stale/wrong report.
    b_report = _eval_report(resolved=["a"], unresolved=["b"])
    t_report = _eval_report(resolved=["a"])  # missing "b"
    for r in (b_report, t_report):
        for k in (
            "resolved_ids",
            "unresolved_ids",
            "empty_patch_ids",
            "error_ids",
            "incomplete_ids",
            "submitted_ids",
            "completed_ids",
        ):
            if k in r:
                r[k] = frozenset(r[k])
    with pytest.raises(ValueError, match="treatment eval report is missing 1 instance"):
        gen.build_artifact(
            baseline,
            treatment,
            set(),
            model="openai/gpt-5",
            baseline_eval_report=b_report,
            treatment_eval_report=t_report,
        )


def test_validate_eval_report_accepts_matching_ids() -> None:
    rows = [_row("a", attempt=1), _row("b", attempt=1)]
    report = _eval_report(resolved=["a"], unresolved=["b"])
    for k in (
        "resolved_ids",
        "unresolved_ids",
        "empty_patch_ids",
        "error_ids",
        "incomplete_ids",
        "submitted_ids",
        "completed_ids",
    ):
        if k in report:
            report[k] = frozenset(report[k])
    # No exception.
    gen._validate_eval_report_covers_rows(report, rows, "baseline")


def test_markdown_does_not_claim_patch_evaluation_when_has_eval_is_false() -> None:
    """Roborev #846 Medium: the infer-only branch must not say "from
    inference through patch-evaluation" in the "Does" section. That
    contradicts the "`resolved` not computed" caveat.
    """
    baseline = [_row("a", attempt=1)]
    treatment = [_row("a", attempt=1)]
    artifact = gen.build_artifact(baseline, treatment, set(), model="openai/gpt-5")
    md = gen.generate_markdown(artifact)
    assert "from inference through patch-evaluation" not in md
    assert "inference only" in md
    # "Does not" must include the no-eval caveat and its link:
    assert "Establish whether retries improve patch correctness" in md
    # Next steps should lead with running the eval
    assert "Run SWE-bench evaluation" in md


def test_markdown_without_eval_report_keeps_old_schema() -> None:
    """Back-compat: when no eval report supplied, pass@1 row + resolved
    columns absent; `resolved` not computed caveat present."""
    baseline = [_row("a", attempt=1)]
    treatment = [_row("a", attempt=1)]
    artifact = gen.build_artifact(baseline, treatment, set(), model="openai/gpt-5")
    md = gen.generate_markdown(artifact)
    assert "**pass@1**" not in md
    assert "Base resolved" not in md
    assert "resolved` not computed" in md


def test_markdown_per_instance_rate_stays_in_range_with_attempt_3() -> None:
    """Roborev #838: a run with attempt >= 3 instances must not produce
    a ">100%" rejection-rate in the markdown. The per-instance rate
    (used for the headline prose + percentage) is capped by design;
    per-round totals live in a separate field.
    """
    baseline = [_row("a", attempt=1), _row("b", attempt=1), _row("c", attempt=1)]
    treatment = [
        _row("a", attempt=1),
        _row("b", attempt=1),
        _row("b", attempt=2),
        _row("c", attempt=1),
        _row("c", attempt=2),
        _row("c", attempt=3),
    ]
    artifact = gen.build_artifact(baseline, treatment, set(), model="openai/gpt-5")
    md = gen.generate_markdown(artifact)
    # 2 of 3 instances had rejections → 67%. Per-round total is 3.
    assert "67%" in md
    assert "rejected the first-attempt patch on 2 of 3 instances" in md
    assert "producing 3 total retry round(s)" in md
    # Rejection-rate row + prose should show 67%, not 100% — the old
    # bug was that the rate field was divided by n_instances but could
    # exceed n_instances under attempt>=3. Cost-overhead percentages
    # can legitimately exceed 100% and are not checked here.
    rate_row_lines = [ln for ln in md.splitlines() if "Per-instance rejection rate" in ln]
    assert rate_row_lines, "markdown missing per-instance rejection rate row"
    assert "67%" in rate_row_lines[0]
    assert "100%" not in rate_row_lines[0]
