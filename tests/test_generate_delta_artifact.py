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
