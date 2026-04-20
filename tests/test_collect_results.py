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


def test_build_artifact_rejects_duplicate_rows_within_baseline() -> None:
    baseline = [_record("a"), _record("a")]  # duplicate in baseline
    treatment = [_record("a")]
    with pytest.raises(ValueError, match="duplicate instance_id.*in baseline"):
        collect_results.build_artifact(baseline, treatment)


def test_build_artifact_rejects_mirror_duplicates_with_matching_sets() -> None:
    """Roborev #812 Medium: ``baseline=[a,a,b]`` vs ``treatment=[a,b,b]``
    has the same id set ``{a,b}`` and same length ``3``, but each side
    has a duplicate. Previously passed validation and silently
    double-counted. Must now error, naming the condition.
    """
    baseline = [_record("a"), _record("a"), _record("b")]
    treatment = [_record("a"), _record("b"), _record("b")]
    with pytest.raises(ValueError, match="duplicate instance_id.*in baseline"):
        collect_results.build_artifact(baseline, treatment)


def test_build_artifact_rejects_duplicate_rows_within_treatment() -> None:
    baseline = [_record("a"), _record("b")]
    treatment = [_record("a"), _record("b"), _record("b")]
    with pytest.raises(ValueError, match="duplicate instance_id.*in treatment"):
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


def _critic_event(
    theorem: str = "behavioral_stability_windowed",
    source: str = "operon_openhands_gates.stagnation_critic",
    cert_evidence_n: int | None = 3,
) -> dict:
    """History-event fixture that mirrors the real serialized CriticResult.

    ``OperonStagnationCritic`` writes ``certificate_theorem``,
    ``certificate_source``, and ``cert_evidence_n`` into
    ``CriticResult.metadata``. The benchmarks runner persists
    ``CriticResult`` records inside ``EvalOutput.history`` entries.
    ``_scan_certificate`` recursively finds the metadata dict, so the
    exact event wrapper shape doesn't matter — but the fixture mirrors
    the end-to-end structure (wrapper → critic_result → metadata)
    rather than inlining the metadata dict at the top level.
    """
    metadata: dict = {
        "severity": 0.85,
        "certificate_theorem": theorem,
        "certificate_source": source,
    }
    if cert_evidence_n is not None:
        metadata["cert_evidence_n"] = cert_evidence_n
    return {
        "kind": "AgentActionEvent",
        "source": "agent",
        "critic_result": {
            "score": 0.15,
            "message": "epiplexic_integral=0.150 threshold=0.2 streak=3 STAGNANT",
            "metadata": metadata,
        },
    }


def test_extract_result_populates_certificate_fields_for_treatment() -> None:
    """Regression for roborev #813 Medium: fixture mirrors real
    serialized history (wrapper event → ``critic_result`` → nested
    ``metadata``) rather than flattening the metadata dict at the top
    level. The recursive scan finds the certificate fields and the
    ``cert_evidence_n`` value the critic now emits into metadata.
    """
    history = [{"kind": "MessageEvent"}, _critic_event(cert_evidence_n=3)]
    record = _record("a", history=history)
    out = collect_results._extract_result(record, "operon_stagnation")
    assert out["certificate_emitted"] is True
    assert out["certificate_theorem"] == "behavioral_stability_windowed"
    assert out["certificate_source"] == "operon_openhands_gates.stagnation_critic"
    assert out["cert_evidence_n"] == 3


def test_extract_result_certificate_emitted_false_when_absent() -> None:
    record = _record("a", history=[{"kind": "MessageEvent"}])
    out = collect_results._extract_result(record, "operon_stagnation")
    assert out["certificate_emitted"] is False
    assert out["certificate_theorem"] is None
    # cert_evidence_n is a documented schema field; it stays present on
    # the row even when the critic didn't fire (value ``None``) so
    # downstream consumers can key on it without probing.
    assert out["cert_evidence_n"] is None


def test_extract_result_cert_evidence_n_none_when_runner_omits_field() -> None:
    """Roborev #813 Medium: older benchmarks versions (or a stripped
    metadata dict) may not carry ``cert_evidence_n``. Row still carries
    the key with ``None`` so downstream consumers can key on it without
    probing. ``certificate_emitted`` still flips True because the
    theorem name is present.
    """
    history = [_critic_event(cert_evidence_n=None)]
    out = collect_results._extract_result(_record("a", history=history), "operon_stagnation")
    assert out["certificate_emitted"] is True
    assert out["cert_evidence_n"] is None


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


# ---- Roborev #836 Medium: nullable pass_at_1 on unevaluated runs -----------


def test_pass_at_1_is_null_when_resolved_missing_on_any_row() -> None:
    """Roborev #836 Medium: if SWE-bench patch-evaluation step hasn't
    run, ``test_result.resolved`` is None on every row and the old
    ``resolved / n`` math silently reports 0.0 — presenting an
    unevaluated run as a real 0 pass rate. Summary must emit ``None``
    with an explanatory note instead.
    """
    baseline = [_record("a", resolved=None), _record("b", resolved=None)]
    treatment = [_record("a", resolved=None), _record("b", resolved=None)]
    artifact = collect_results.build_artifact(baseline, treatment)
    assert artifact["summary"]["baseline"]["pass_at_1"] is None
    assert "pass_at_1_note" in artifact["summary"]["baseline"]
    assert artifact["summary"]["operon_stagnation"]["pass_at_1"] is None


def test_pass_at_1_is_null_when_subset_of_rows_unevaluated() -> None:
    """Partial eval (some rows scored, others not) also triggers None —
    the headline metric would otherwise understate pass rate by
    counting unscored rows as failures.
    """
    baseline = [_record("a", resolved=True), _record("b", resolved=None)]
    treatment = [_record("a", resolved=True), _record("b", resolved=True)]
    artifact = collect_results.build_artifact(baseline, treatment)
    assert artifact["summary"]["baseline"]["pass_at_1"] is None
    assert artifact["summary"]["baseline"]["not_evaluated"] == 1
    # Treatment has all rows scored, so pass_at_1 is numeric.
    assert artifact["summary"]["operon_stagnation"]["pass_at_1"] == 1.0


def test_pass_at_1_is_numeric_when_all_rows_evaluated() -> None:
    """Happy path: both sides have resolved set. Note absent."""
    baseline = [_record("a", resolved=True), _record("b", resolved=False)]
    treatment = [_record("a", resolved=True), _record("b", resolved=True)]
    artifact = collect_results.build_artifact(baseline, treatment)
    assert artifact["summary"]["baseline"]["pass_at_1"] == 0.5
    assert "pass_at_1_note" not in artifact["summary"]["baseline"]
