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


# ---- Cert-fire log ingestion (new side-channel log from v0.1.0a3) ---------


def _write_docker_log_with_cert_fire(
    path: Path, payload: dict, *, noise_before: bool = True
) -> None:
    """Write a synthetic instance_*.output.log that mirrors what the
    benchmarks runner captures from the container's stdout.

    Real lines look like:
        [DOCKER] {"asctime": "...", "levelname": "INFO",
                  "name": "operon_openhands_gates.stagnation_critic",
                  "message": "[CERT-FIRE] {\"theorem\": ..., ...}"}
    with optional unrelated lines before/after.
    """
    lines = []
    if noise_before:
        lines.append(
            '[DOCKER] {"asctime": "2026-04-21 12:00:00,000", "levelname": "INFO", '
            '"name": "uvicorn.access", "message": "GET /api/conversations/foo"}'
        )
    lines.append(
        '[DOCKER] {"asctime": "2026-04-21 12:00:01,000", "levelname": "INFO", '
        '"name": "operon_openhands_gates.stagnation_critic", '
        '"message": "[CERT-FIRE] ' + json.dumps(payload).replace('"', '\\"') + '"}'
    )
    lines.append(
        '[DOCKER] {"asctime": "2026-04-21 12:00:02,000", "levelname": "INFO", '
        '"name": "uvicorn.access", "message": "GET /api/conversations/bar"}'
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_cert_fires_parses_docker_wrapped_lines(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    payload = {
        "theorem": "behavioral_stability_windowed",
        "source": "operon_openhands_gates.stagnation_critic",
        "cert_evidence_n": 3,
        "epiplexic_integral": 0.12,
        "severity": 0.88,
        "detection_index": 42,
    }
    _write_docker_log_with_cert_fire(logs_dir / "instance_django__django-11001.output.log", payload)
    _write_docker_log_with_cert_fire(
        logs_dir / "instance_astropy__astropy-12907.output.log", payload
    )
    result = gen._load_cert_fires_from_logs(logs_dir)
    assert set(result.keys()) == {"django__django-11001", "astropy__astropy-12907"}
    assert result["django__django-11001"]["theorem"] == "behavioral_stability_windowed"
    assert result["django__django-11001"]["cert_evidence_n"] == 3


def test_load_cert_fires_returns_empty_when_logs_dir_missing(tmp_path: Path) -> None:
    """Missing logs dir is not an error — treated same as no cert fires.
    Lets pre-instrumentation runs flow through the back-compat path.
    """
    result = gen._load_cert_fires_from_logs(tmp_path / "nonexistent")
    assert result == {}


def test_load_cert_fires_returns_empty_when_no_cert_lines(tmp_path: Path) -> None:
    """Logs dir exists but no [CERT-FIRE] lines — instance not keyed."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "instance_a.output.log").write_text(
        '[DOCKER] {"asctime": "x", "message": "nothing to see here"}\n',
        encoding="utf-8",
    )
    result = gen._load_cert_fires_from_logs(logs_dir)
    assert result == {}


def test_load_cert_fires_keeps_first_only(tmp_path: Path) -> None:
    """Defensive: two [CERT-FIRE] lines in a single file (shouldn't
    happen in practice — critic only fires on transition) — keep the
    first so downstream correlation is deterministic.
    """
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    log = logs_dir / "instance_a.output.log"
    first = {"theorem": "t1", "source": "s", "cert_evidence_n": 1}
    second = {"theorem": "t2", "source": "s", "cert_evidence_n": 2}
    log.write_text(
        '[DOCKER] {"asctime": "x", "levelname": "INFO", "name": "n", '
        '"message": "[CERT-FIRE] ' + json.dumps(first).replace('"', '\\"') + '"}\n'
        '[DOCKER] {"asctime": "y", "levelname": "INFO", "name": "n", '
        '"message": "[CERT-FIRE] ' + json.dumps(second).replace('"', '\\"') + '"}\n',
        encoding="utf-8",
    )
    result = gen._load_cert_fires_from_logs(logs_dir)
    assert result["a"]["theorem"] == "t1"


def test_build_artifact_populates_cert_fields_when_logs_supplied() -> None:
    baseline = [_row("a", attempt=1), _row("b", attempt=1)]
    treatment = [_row("a", attempt=1), _row("b", attempt=1)]
    # treatment has cert for "a", not "b"; baseline has no certs
    treatment_cert_fires = {"a": {"theorem": "behavioral_stability_windowed", "cert_evidence_n": 3}}
    artifact = gen.build_artifact(
        baseline,
        treatment,
        set(),
        model="openai/gpt-5",
        baseline_cert_fires={},
        treatment_cert_fires=treatment_cert_fires,
    )
    # Summary rollup present on both sides since both cert_fires were
    # passed (empty on baseline, one entry on treatment)
    assert artifact["summary"]["baseline"]["certificates_emitted"] == 0
    assert artifact["summary"]["operon_stagnation"]["certificates_emitted"] == 1
    # Per-instance rows carry the cert payload (or None)
    by_iid = {p["instance_id"]: p for p in artifact["per_instance"]}
    assert by_iid["a"]["treatment_certificate"]["theorem"] == "behavioral_stability_windowed"
    assert by_iid["b"]["treatment_certificate"] is None
    assert by_iid["a"]["baseline_certificate"] is None


def test_markdown_includes_cert_row_and_mutates_caveat_when_evidence_present() -> None:
    baseline = [_row("a", attempt=1), _row("b", attempt=1)]
    treatment = [_row("a", attempt=1), _row("b", attempt=1)]
    treatment_cert_fires = {"a": {"theorem": "behavioral_stability_windowed"}}
    artifact = gen.build_artifact(
        baseline,
        treatment,
        set(),
        model="openai/gpt-5",
        baseline_cert_fires={},
        treatment_cert_fires=treatment_cert_fires,
    )
    md = gen.generate_markdown(artifact)
    # Headline row for certificates
    assert "Certificates emitted" in md
    # Treatment-side cert column with one ✓ for "a" and · for "b"
    assert "Treat cert" in md
    assert "✓" in md
    # Caveat flipped to the on-disk evidence phrasing
    assert "Certificate evidence via side-channel log" in md
    assert "Certificate metadata not serialized" not in md


def test_validate_logs_dir_rejects_missing_directory(tmp_path: Path) -> None:
    """Roborev #848 Medium: a mistyped --*-logs-dir path must fail
    fast, not silently produce ``certificates_emitted: 0``.
    """
    rows = [_row("a", attempt=1)]
    missing = tmp_path / "nonexistent"
    with pytest.raises(ValueError, match="logs dir does not exist"):
        gen._validate_logs_dir_covers_rows(missing, rows, "baseline")


def test_validate_logs_dir_rejects_empty_dir(tmp_path: Path) -> None:
    """An existing but empty logs dir is the clearest case of a wrong
    path (e.g. user pointed at the parent). Fail with a hint.
    """
    empty = tmp_path / "logs"
    empty.mkdir()
    rows = [_row("a", attempt=1)]
    with pytest.raises(ValueError, match="no ``instance_"):
        gen._validate_logs_dir_covers_rows(empty, rows, "baseline")


def test_validate_logs_dir_rejects_missing_instance_file(tmp_path: Path) -> None:
    """A logs dir from a different slice (wrong instance coverage)
    must fail with the list of missing IDs.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "instance_a.output.log").write_text("")  # has "a" only
    rows = [_row("a", attempt=1), _row("b", attempt=1)]  # needs both
    with pytest.raises(ValueError, match="missing 1 ``instance_"):
        gen._validate_logs_dir_covers_rows(logs, rows, "treatment")


def test_validate_logs_dir_accepts_matching_files(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "instance_a.output.log").write_text("")
    (logs / "instance_b.output.log").write_text("")
    # Extra files beyond the row set are fine — only MISSING is an error.
    (logs / "instance_c.output.log").write_text("")
    rows = [_row("a", attempt=1), _row("b", attempt=1)]
    # No exception.
    gen._validate_logs_dir_covers_rows(logs, rows, "baseline")


def test_markdown_keeps_old_caveat_when_no_cert_evidence() -> None:
    baseline = [_row("a", attempt=1)]
    treatment = [_row("a", attempt=1)]
    artifact = gen.build_artifact(baseline, treatment, set(), model="openai/gpt-5")
    md = gen.generate_markdown(artifact)
    assert "Certificates emitted" not in md
    assert "Certificate metadata not serialized" in md
    assert "Treat cert" not in md
