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


def test_aggregate_completed_retries_counts_retry_rounds_not_instances() -> None:
    """Instance with max_attempt=3 contributes 2 retries, not 1."""
    rows = [
        _row("a", attempt=1),
        _row("b", attempt=1),
        _row("b", attempt=2),
        _row("c", attempt=1),
        _row("c", attempt=2),
        _row("c", attempt=3),
    ]
    agg = gen._aggregate(rows, set())
    # b contributes 1 retry (attempt 2), c contributes 2 (attempts 2+3)
    assert agg["completed_retries"] == 3
    assert agg["instances_with_completed_retry"] == 2
    # critic_rejections = total retry rounds (aborted + completed)
    assert agg["critic_rejections"] == 3
    # rate is per-instance (retries/n_instances): 3/3 = 1.0 (every
    # instance... no wait, 2/3 had retries, but 3 rounds total
    # divided by 3 instances = 1.0 "rejections per instance on average")
    assert agg["critic_rejection_rate"] == 1.0


def test_aggregate_retry_counts_match_when_all_max_attempt_is_2() -> None:
    """The common case in this repo: every retried instance completed
    exactly one retry. ``completed_retries == instances_with_completed_retry``.
    """
    rows = [
        _row("a", attempt=1),
        _row("b", attempt=1),
        _row("b", attempt=2),
        _row("c", attempt=1),
        _row("c", attempt=2),
    ]
    agg = gen._aggregate(rows, set())
    assert agg["completed_retries"] == 2
    assert agg["instances_with_completed_retry"] == 2


def test_aggregate_aborted_retries_add_to_rejections() -> None:
    rows = [_row("a", attempt=1), _row("b", attempt=1)]
    agg = gen._aggregate(rows, {"b"})
    assert agg["completed_retries"] == 0
    assert agg["aborted_retries"] == 1
    assert agg["critic_rejections"] == 1
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
    assert artifact["summary"]["baseline"]["completed_retries"] == 0
    assert artifact["summary"]["operon_stagnation"]["completed_retries"] == 1
    assert artifact["summary"]["operon_stagnation"]["instances_with_completed_retry"] == 1
    assert artifact["scope"]["n_instances"] == 2


def test_build_artifact_rejects_instance_set_mismatch() -> None:
    baseline = [_row("a", attempt=1)]
    treatment = [_row("b", attempt=1)]  # different ID
    with pytest.raises(ValueError, match="instance_id set mismatch"):
        gen.build_artifact(baseline, treatment, set(), model="openai/gpt-5")
