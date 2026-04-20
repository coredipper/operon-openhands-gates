"""Regression tests for scripts/collect_results.py.

Covers the three fixes from roborev #811:

- High: instance-id set mismatch between baseline and treatment is a
  hard error (``_validate_matched_instances`` raises).
- Medium: ambiguous output JSONL selection is a hard error
  (``_find_output_jsonl`` raises when multiple matches exist).
- Medium: certificate metadata is extracted from history
  (``_scan_certificate`` finds the key, ``_extract_result`` propagates
  it into the output row).

Tests the scripts/ module by path import — it's not part of the
installed package and we don't want to move it there for test access.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_MODULE_PATH = _SCRIPTS_DIR / "collect_results.py"

# Load scripts/collect_results.py as a module without polluting sys.path
# with the whole scripts/ directory (which also contains runner scripts
# that import the external benchmarks package).
_spec = importlib.util.spec_from_file_location("_collect_results", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
collect_results = importlib.util.module_from_spec(_spec)
sys.modules["_collect_results"] = collect_results
_spec.loader.exec_module(collect_results)


# ---- Synthetic record factories --------------------------------------------


def _record(
    instance_id: str,
    *,
    resolved: bool | None = True,
    history: list[dict] | None = None,
    error: str | None = None,
) -> dict:
    """A synthetic ``EvalOutput``-shaped record."""
    test_result: dict = {"git_patch": "diff --git a b\n"}
    if resolved is not None:
        test_result["resolved"] = resolved
    return {
        "instance_id": instance_id,
        "attempt": 1,
        "test_result": test_result,
        "history": history or [],
        "metrics": {"total_tokens": 1000},
        "error": error,
    }


# ---- High: instance-set validation ------------------------------------------


def test_build_artifact_requires_matching_instance_sets() -> None:
    baseline = [_record("a"), _record("b")]
    treatment = [_record("a"), _record("c")]  # c missing from baseline, b missing
    with pytest.raises(ValueError, match="instance_id set mismatch"):
        collect_results.build_artifact(baseline, treatment)


def test_build_artifact_requires_no_missing_instance_ids() -> None:
    baseline = [_record("a"), {"attempt": 1, "test_result": {}}]  # missing id
    treatment = [_record("a"), _record("b")]
    with pytest.raises(ValueError, match="missing ``instance_id``"):
        collect_results.build_artifact(baseline, treatment)


def test_build_artifact_rejects_duplicate_rows_same_id_set() -> None:
    baseline = [_record("a"), _record("a")]  # duplicate
    treatment = [_record("a")]
    with pytest.raises(ValueError, match="row-count mismatch"):
        collect_results.build_artifact(baseline, treatment)


def test_build_artifact_happy_path_identical_sets() -> None:
    baseline = [_record("a"), _record("b")]
    treatment = [_record("a"), _record("b")]
    artifact = collect_results.build_artifact(baseline, treatment)
    assert artifact["dataset"] == "princeton-nlp/SWE-bench_Lite"
    assert artifact["conditions"] == ["baseline", "operon_stagnation"]
    assert len(artifact["results"]) == 4  # 2 × 2 conditions
    assert artifact["summary"]["baseline"]["n"] == 2
    assert artifact["summary"]["operon_stagnation"]["n"] == 2


# ---- Medium: unambiguous JSONL selection ------------------------------------


def test_find_output_jsonl_errors_on_zero_matches(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no output.critic_attempt"):
        collect_results._find_output_jsonl(tmp_path)


def test_find_output_jsonl_errors_on_multiple_matches(tmp_path: Path) -> None:
    (tmp_path / "output.critic_attempt_1.jsonl").write_text("")
    (tmp_path / "output.critic_attempt_2.jsonl").write_text("")
    with pytest.raises(ValueError, match="ambiguous"):
        collect_results._find_output_jsonl(tmp_path)


def test_find_output_jsonl_returns_sole_match(tmp_path: Path) -> None:
    (tmp_path / "output.critic_attempt_1.jsonl").write_text("")
    hit = collect_results._find_output_jsonl(tmp_path)
    assert hit.name == "output.critic_attempt_1.jsonl"


# ---- Medium: certificate extraction from history ----------------------------


def test_scan_certificate_returns_none_when_absent() -> None:
    history = [{"kind": "MessageEvent", "content": "hello"}]
    assert collect_results._scan_certificate(history) is None


def test_scan_certificate_finds_key_in_nested_metadata() -> None:
    # Mirror the shape OperonStagnationCritic writes: CriticResult.metadata
    # propagated into a history event's metadata dict.
    history = [
        {
            "kind": "CriticEvent",
            "metadata": {
                "certificate_theorem": "behavioral_stability_windowed",
                "certificate_source": "operon_openhands_gates.stagnation_critic",
            },
        }
    ]
    hit = collect_results._scan_certificate(history)
    assert hit is not None
    assert hit["certificate_theorem"] == "behavioral_stability_windowed"
    assert hit["certificate_source"] == "operon_openhands_gates.stagnation_critic"


def test_scan_certificate_finds_key_in_deeply_nested_structures() -> None:
    history = [
        {
            "kind": "ActionEvent",
            "nested": {
                "also_nested": [
                    {"irrelevant": 1},
                    {"critic_result": {"metadata": {"certificate_theorem": "x"}}},
                ]
            },
        }
    ]
    hit = collect_results._scan_certificate(history)
    assert hit is not None
    assert hit["certificate_theorem"] == "x"


def test_extract_result_populates_certificate_fields_for_treatment() -> None:
    history = [
        {"kind": "MessageEvent"},
        {
            "metadata": {
                "certificate_theorem": "behavioral_stability_windowed",
                "certificate_source": "src",
            }
        },
    ]
    record = _record("a", history=history)
    out = collect_results._extract_result(record, "operon_stagnation")
    assert out["certificate_emitted"] is True
    assert out["certificate_theorem"] == "behavioral_stability_windowed"
    assert out["certificate_source"] == "src"


def test_extract_result_certificate_emitted_false_when_absent() -> None:
    record = _record("a", history=[{"kind": "MessageEvent"}])
    out = collect_results._extract_result(record, "operon_stagnation")
    assert out["certificate_emitted"] is False
    assert out["certificate_theorem"] is None


def test_extract_result_omits_certificate_fields_on_baseline() -> None:
    history = [{"metadata": {"certificate_theorem": "behavioral_stability_windowed"}}]
    record = _record("a", history=history)
    out = collect_results._extract_result(record, "baseline")
    assert "certificate_emitted" not in out
    assert "certificate_theorem" not in out


def test_summary_includes_certificate_rollup_on_treatment_only() -> None:
    baseline = [_record("a"), _record("b")]
    treatment_history = [{"metadata": {"certificate_theorem": "behavioral_stability_windowed"}}]
    treatment = [
        _record("a", history=treatment_history),
        _record("b"),  # no cert
    ]
    artifact = collect_results.build_artifact(baseline, treatment)
    assert "certificates_emitted" not in artifact["summary"]["baseline"]
    assert artifact["summary"]["operon_stagnation"]["certificates_emitted"] == 1
