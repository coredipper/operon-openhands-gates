# ruff: noqa: E501  # prose caveats embedded in markdown templates are intentionally long
"""Generate the SWE-bench-lite delta artifact from baseline + treatment runs.

Produces ``eval/results/swebench_lite_delta.json`` (and
``.md`` via :func:`generate_markdown`) from the benchmarks runner's
``output.jsonl`` files for each condition. Deduplicates to one row per
instance by keeping the max-attempt row (the final critic-accepted
state) but reports cumulative cost / tokens across all attempts so the
retry overhead is visible in the summary.

Use this when:

- The runs used the ``OperonStagnationCritic`` iterative-refinement
  loop, which writes multiple ``(instance_id, attempt)`` rows to
  ``output.jsonl`` — one per critic attempt.
- One or more aborted retries (e.g. Attempt-2 hit the per-attempt
  timeout ceiling) need to be counted as critic rejections in the
  headline without introducing cost-per-rejection bias.

Why not ``collect_results.py``:
  That script is scoped for the full n=30 baseline/treatment pair with
  ``pass_at_1``, ``mean_turns``, ``total_tokens`` aggregates — it
  assumes the SWE-bench patch-evaluation step was run (``resolved``
  populated). This script handles the "infer-only" case: no resolved
  counts, per-rejection + per-completed-retry cost breakdown,
  aborted-retry accounting.

Usage::

    python scripts/generate_delta_artifact.py \\
        --baseline eval/runs/baseline \\
        --treatment eval/runs/treatment \\
        --aborted-treatment-retry django__django-11019 \\
        --out-json eval/results/swebench_lite_delta.json \\
        --out-md eval/results/swebench_lite_delta.md

The ``--aborted-treatment-retry`` flag is repeatable. It accepts
instance IDs whose treatment critic rejected Attempt-1 and triggered
Attempt-2, but Attempt-2 did not complete (e.g. timed out and was
killed) so no Attempt-2 row appears in ``output.jsonl``. These IDs
are counted as critic rejections in the headline rate, but excluded
from the ``$/completed-retry`` cost metric (their spend would bias it
downward — roborev #835 Medium 1).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path


def _find_output_jsonl(run_dir: Path) -> Path:
    matches = sorted(run_dir.rglob("output.jsonl"))
    if not matches:
        raise FileNotFoundError(f"no output.jsonl under {run_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous: {len(matches)} output.jsonl files under {run_dir}:\n"
            + "\n".join(f"  {m}" for m in matches)
        )
    return matches[0]


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()]


def _load_all_attempt_rows(run_dir: Path) -> list[dict]:
    """Load rows spanning every critic attempt, scoped to successful instances.

    The benchmarks runner's ``output.jsonl`` writes differ across
    versions: some keep every ``(instance_id, attempt)`` row, others
    consolidate to one row per instance at run-end. When consolidated,
    the cross-attempt cumulative cost/tokens are lost if we only read
    ``output.jsonl``.

    Per-attempt rows are preserved in ``output.critic_attempt_<N>.jsonl``
    files. This helper unions those files but filters to instances
    present in ``output.jsonl`` — the critic_attempt files may include
    instances that errored out of the final set (e.g. Docker build
    failures), and including them would expand the scope beyond the
    successful slice.

    If no ``critic_attempt_*`` files exist, falls back to ``output.jsonl``
    (older runs had multi-attempt rows there directly).
    """
    output_jsonl = _find_output_jsonl(run_dir)
    parent = output_jsonl.parent
    attempt_files = sorted(parent.glob("output.critic_attempt_*.jsonl"))
    if not attempt_files:
        return _load_jsonl(output_jsonl)
    # Scope = successful instance set from output.jsonl.
    scope_ids = {r["instance_id"] for r in _load_jsonl(output_jsonl)}
    rows: list[dict] = []
    for f in attempt_files:
        for line in f.open():
            r = json.loads(line)
            if r.get("instance_id") in scope_ids:
                rows.append(r)
    return rows


def _by_instance(rows: list[dict]) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for r in rows:
        by[r["instance_id"]].append(r)
    return dict(by)


def _safe_agg(vals: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean": round(statistics.fmean(vals), 4) if vals else None,
        "median": round(statistics.median(vals), 4) if vals else None,
    }


def _cum_metric(rows: list[dict], dotted: str, default: float = 0) -> float:
    total = 0.0
    for r in rows:
        cur = r
        for part in dotted.split("."):
            cur = (cur or {}).get(part) if isinstance(cur, dict) else None
            if cur is None:
                break
        total += cur if isinstance(cur, (int, float)) else default
    return total


def _final_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: r.get("attempt", 1))


# Line prefix emitted by ``OperonStagnationCritic.evaluate()`` when a
# certificate transitions from None → fired. See the library change in
# ``src/operon_openhands_gates/stagnation_critic.py``. The benchmarks
# runner captures the container's stdout into
# ``logs/instance_<iid>.output.log`` with the openhands JSON-log wrapper
# (``[DOCKER] {"asctime": ..., "message": "...the line above..."}``);
# this parser extracts the inner ``[CERT-FIRE] {json}`` payload.
_CERT_FIRE_PREFIX = "[CERT-FIRE] "
_DOCKER_LOG_PREFIX = "[DOCKER] "
_INSTANCE_LOG_STEM_PREFIX = "instance_"
_INSTANCE_LOG_SUFFIX = ".output.log"


def _load_cert_fires_from_logs(logs_dir: Path) -> dict[str, dict]:
    """Parse ``[CERT-FIRE]`` log lines from per-instance container logs.

    Walks ``logs_dir`` for ``instance_<iid>.output.log`` files, scans
    each for lines containing the ``[CERT-FIRE]`` prefix (either bare
    or wrapped in the openhands JSON log envelope
    ``[DOCKER] {"asctime": ..., "message": "...[CERT-FIRE] {...}..."}``),
    and returns a dict keyed by instance_id (extracted from the filename).

    Keeps the **first** cert-fire payload per instance. The critic's
    emission guard fires once per critic instance per conversation, so
    duplicates shouldn't occur; if they do (e.g. a retried Attempt-2
    also fires), we favor the Attempt-1 record since that's the one
    correlated with the baseline comparison.

    Missing ``logs_dir`` is not an error — returns an empty dict so
    callers can treat "no logs" identically to "logs but no cert
    fires," which is the pre-instrumentation regime.
    """
    if not logs_dir.is_dir():
        return {}

    cert_fires: dict[str, dict] = {}
    for log_path in sorted(logs_dir.glob(f"{_INSTANCE_LOG_STEM_PREFIX}*{_INSTANCE_LOG_SUFFIX}")):
        stem = log_path.name
        if not stem.startswith(_INSTANCE_LOG_STEM_PREFIX) or not stem.endswith(
            _INSTANCE_LOG_SUFFIX
        ):
            continue
        instance_id = stem[len(_INSTANCE_LOG_STEM_PREFIX) : -len(_INSTANCE_LOG_SUFFIX)]

        for raw_line in log_path.open(encoding="utf-8", errors="replace"):
            # Two shapes to handle:
            #   1. Bare line: "...[CERT-FIRE] {...json...}..."
            #   2. Docker-wrapped: "[DOCKER] {\"message\": \"...[CERT-FIRE] {...}...\"}"
            # For the wrapped case we parse the outer JSON first and
            # then search the ``message`` field. For the bare case we
            # search the raw line. Either way, once we find the
            # ``[CERT-FIRE] `` anchor we parse the trailing JSON.
            search_text = raw_line
            if raw_line.lstrip().startswith(_DOCKER_LOG_PREFIX):
                stripped = raw_line.lstrip()[len(_DOCKER_LOG_PREFIX) :]
                try:
                    outer = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                search_text = outer.get("message", "")
                if not isinstance(search_text, str):
                    continue
            anchor = search_text.find(_CERT_FIRE_PREFIX)
            if anchor < 0:
                continue
            payload_text = search_text[anchor + len(_CERT_FIRE_PREFIX) :].strip()
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            if instance_id not in cert_fires:
                cert_fires[instance_id] = payload
            # Continue scanning other lines, but this instance is locked.
    return cert_fires


def _load_eval_report(path: Path) -> dict:
    """Load a SWE-bench harness ``.report.json`` and normalize the id fields.

    The report is emitted by ``swebench.harness.reporting.make_run_report``
    (see .venv-experiment/.../swebench/harness/reporting.py:127-143) with
    fields ``resolved_ids``, ``unresolved_ids``, ``empty_patch_ids``,
    ``error_ids``, ``incomplete_ids``, ``submitted_ids``, plus the
    corresponding counts.

    Returns the dict with ``*_ids`` promoted to frozensets for fast
    ``iid in report["resolved_ids"]`` membership checks.
    """
    data = json.loads(path.read_text())
    for key in (
        "resolved_ids",
        "unresolved_ids",
        "empty_patch_ids",
        "error_ids",
        "incomplete_ids",
        "submitted_ids",
        "completed_ids",
    ):
        if key in data:
            data[key] = frozenset(data[key])
    return data


def _validate_logs_dir_covers_rows(
    logs_dir: Path,
    rows: list[dict],
    condition_label: str,
) -> None:
    """Require the logs dir to contain ``instance_<iid>.output.log`` for
    every row's ``instance_id``.

    Roborev #848 Medium. A mistyped ``--baseline-logs-dir`` /
    ``--treatment-logs-dir`` path or a stale log directory would
    silently return an empty cert dict, making the artifact report
    ``certificates_emitted: 0`` without any error — symmetric to how
    eval reports are validated. Fail fast with a list of missing
    ``instance_<iid>.output.log`` filenames so the user can fix the
    flag rather than the artifact.

    An empty logs dir (no ``instance_*.output.log`` at all) is also
    an error — it's the clearest case of a wrong path.
    """
    if not logs_dir.is_dir():
        raise ValueError(
            f"{condition_label} logs dir does not exist: {logs_dir!s}\n"
            "  Expected a directory containing ``instance_<iid>.output.log`` "
            "files (created by the benchmarks runner under "
            "``eval/runs/<cond>/.../logs/``)."
        )
    row_ids = {r["instance_id"] for r in rows}
    present_ids = {
        p.name[len(_INSTANCE_LOG_STEM_PREFIX) : -len(_INSTANCE_LOG_SUFFIX)]
        for p in logs_dir.glob(f"{_INSTANCE_LOG_STEM_PREFIX}*{_INSTANCE_LOG_SUFFIX}")
    }
    if not present_ids:
        raise ValueError(
            f"{condition_label} logs dir {logs_dir!s} has no "
            f"``instance_*.output.log`` files. "
            "Point at the directory the benchmarks runner wrote to, "
            "not its parent."
        )
    missing = sorted(row_ids - present_ids)
    if missing:
        raise ValueError(
            f"{condition_label} logs dir is missing {len(missing)} "
            f"``instance_<iid>.output.log`` file(s) for the "
            f"{condition_label} run rows: "
            + ", ".join(missing)
            + f"\n  Logs dir scanned: {logs_dir!s}. "
            "The logs dir is probably from a different slice — point at "
            "the one that matches this run's ``output.jsonl``."
        )


def _validate_eval_report_covers_rows(
    report: dict,
    rows: list[dict],
    condition_label: str,
) -> None:
    """Require every row's ``instance_id`` to appear in the report.

    Roborev #844 Medium 1. A mistyped ``--baseline-eval-report`` /
    ``--treatment-eval-report`` path, or a stale report from a
    different slice, would silently mark every uncovered row as
    ``"incomplete"`` and skew ``pass_at_1`` + resolved counts. Fail
    fast instead: the report must contain an entry (in ``submitted_ids``
    or any of the status id sets) for every instance in the run.
    """
    covered: frozenset[str] = frozenset().union(
        *(
            report.get(k, frozenset())
            for k in (
                "submitted_ids",
                "resolved_ids",
                "unresolved_ids",
                "empty_patch_ids",
                "error_ids",
                "incomplete_ids",
                "completed_ids",
            )
        )
    )
    row_ids = {r["instance_id"] for r in rows}
    missing = sorted(row_ids - covered)
    if missing:
        raise ValueError(
            f"{condition_label} eval report is missing {len(missing)} "
            f"instance_id(s) from the {condition_label} run rows: "
            + ", ".join(missing)
            + "\n  The report probably points at the wrong run or a stale slice. "
            "Regenerate via ``benchmarks.swebench.eval_infer`` on the "
            "matching output.jsonl."
        )


def _eval_status_for(iid: str, report: dict | None) -> str | None:
    """Classify an instance against the eval report.

    Returns one of ``"resolved"``, ``"unresolved"``, ``"empty_patch"``,
    ``"error"``, ``"incomplete"``, or ``None`` if no report was supplied.
    Instances that appear in none of the report's id sets fall back to
    ``"incomplete"`` — they were in the predictions file but the harness
    didn't complete them for some reason.
    """
    if report is None:
        return None
    if iid in report.get("resolved_ids", frozenset()):
        return "resolved"
    if iid in report.get("unresolved_ids", frozenset()):
        return "unresolved"
    if iid in report.get("empty_patch_ids", frozenset()):
        return "empty_patch"
    if iid in report.get("error_ids", frozenset()):
        return "error"
    return "incomplete"


def _aggregate(
    rows: list[dict],
    aborted_retries: set[str],
    eval_report: dict | None = None,
    cert_fires: dict[str, dict] | None = None,
) -> dict:
    by_iid = _by_instance(rows)
    finals = [_final_row(rs) for rs in by_iid.values()]

    cum_costs = [_cum_metric(rs, "metrics.accumulated_cost") for rs in by_iid.values()]
    cum_prompt = [
        _cum_metric(rs, "metrics.accumulated_token_usage.prompt_tokens") for rs in by_iid.values()
    ]
    cum_comp = [
        _cum_metric(rs, "metrics.accumulated_token_usage.completion_tokens")
        for rs in by_iid.values()
    ]
    cum_reason = [
        _cum_metric(rs, "metrics.accumulated_token_usage.reasoning_tokens")
        for rs in by_iid.values()
    ]

    patch_lens = [len((r.get("test_result") or {}).get("git_patch") or "") for r in finals]
    n_events = [len(r.get("history") or []) for r in finals]
    max_attempts = [max(r.get("attempt", 1) for r in rs) for rs in by_iid.values()]

    # Retry accounting — roborev #837/#838. Two orthogonal metrics:
    #
    # - **Per-instance**: how many *instances* had ≥1 rejection (either
    #   a completed retry or an aborted one)? This is the headline
    #   rate the writeup quotes as "rejected X of N first-attempt
    #   patches (Y%)"; always in [0, 1].
    #
    # - **Per-round**: how many total retry *rounds* happened across
    #   the slice — ``sum(max_attempt - 1)`` for completed rounds plus
    #   one per aborted retry. This is the denominator for the
    #   per-retry cost metric. Under attempt >= 3 it exceeds the
    #   per-instance count (a 2-retry instance contributes 2 to rounds
    #   but 1 to instances).
    #
    # The old ``critic_rejections`` / ``critic_rejection_rate`` was
    # overloaded to mean rounds, but the markdown treated it as a
    # per-instance rate (producing nonsensical ">100%" under heavy
    # retry). Split cleanly now.
    aborted_in_scope = sorted(aborted_retries & set(by_iid.keys()))
    aborted_set = set(aborted_in_scope)

    # Per-instance rejection metric (used for headline rate + prose):
    # an instance is "rejected" iff it had a completed retry OR is in
    # the aborted-retry list. Each instance contributes at most 1.
    instances_with_rejection = sum(
        1
        for iid, rows_for in by_iid.items()
        if max(r.get("attempt", 1) for r in rows_for) > 1 or iid in aborted_set
    )

    # Per-round totals (used for the $/retry cost denominator):
    total_completed_retry_rounds = sum(a - 1 for a in max_attempts if a > 1)
    total_retry_rounds = total_completed_retry_rounds + len(aborted_in_scope)

    instances_with_completed_retry = sum(1 for a in max_attempts if a > 1)

    summary = {
        "n_instances": len(by_iid),
        "final_patch_len": _safe_agg(patch_lens),
        "final_n_events": _safe_agg(n_events),
        "cumulative_cost_usd": {"total": round(sum(cum_costs), 2), **_safe_agg(cum_costs)},
        "cumulative_prompt_tokens": {"total": sum(cum_prompt), **_safe_agg(cum_prompt)},
        "cumulative_completion_tokens": {"total": sum(cum_comp), **_safe_agg(cum_comp)},
        "cumulative_reasoning_tokens": {"total": sum(cum_reason), **_safe_agg(cum_reason)},
        # Per-instance (headline rate — always in [0, 1]):
        "instances_with_rejection": instances_with_rejection,
        "instances_with_rejection_rate": round(instances_with_rejection / len(by_iid), 3),
        # Per-round (cost-denominator):
        "total_retry_rounds": total_retry_rounds,
        "total_completed_retry_rounds": total_completed_retry_rounds,
        # Breakdown:
        "instances_with_completed_retry": instances_with_completed_retry,
        "aborted_retries": len(aborted_in_scope),
        "aborted_retry_instance_ids": aborted_in_scope,
        "max_attempt_histogram": dict(Counter(max_attempts)),
    }

    # pass@1 + eval-status rollup (optional — only populated when the
    # caller supplied a SWE-bench ``.report.json``). Mirrors
    # ``scripts/collect_results.py._aggregate``'s convention of emitting
    # ``pass_at_1: None`` with a note when any instance is unscored,
    # rather than silently reporting 0.0.
    if eval_report is not None:
        statuses = [_eval_status_for(iid, eval_report) for iid in by_iid]
        n_resolved = sum(1 for s in statuses if s == "resolved")
        n_unresolved = sum(1 for s in statuses if s == "unresolved")
        n_empty = sum(1 for s in statuses if s == "empty_patch")
        n_error = sum(1 for s in statuses if s == "error")
        n_incomplete = sum(1 for s in statuses if s == "incomplete")
        summary["resolved_count"] = n_resolved
        summary["unresolved_count"] = n_unresolved
        summary["empty_patch_count"] = n_empty
        summary["error_count"] = n_error
        summary["incomplete_count"] = n_incomplete
        if n_incomplete > 0:
            # Same rule as collect_results.py: any unscored row ⇒
            # pass@1 is unknown, not silently 0.
            summary["pass_at_1"] = None
            summary["pass_at_1_note"] = (
                f"{n_incomplete}/{len(by_iid)} instances were not "
                "scored by the eval harness (incomplete)."
            )
        else:
            summary["pass_at_1"] = round(n_resolved / len(by_iid), 4)

    # Cert-fire rollup (optional — populated only when a logs dir with
    # ``[CERT-FIRE]`` lines was supplied). Each in-scope instance
    # either has a cert record or doesn't; the count and rate are
    # direct on-disk evidence rather than the retry-count proxy.
    if cert_fires is not None:
        n_with_cert = sum(1 for iid in by_iid if iid in cert_fires)
        summary["certificates_emitted"] = n_with_cert
        summary["certificates_emitted_rate"] = round(n_with_cert / len(by_iid), 3)

    return summary


def _validate_aborted_retries(
    treatment_rows: list[dict],
    aborted_retries: set[str],
) -> None:
    """Reject misspelled or inconsistent ``--aborted-treatment-retry`` IDs.

    Roborev #837 Medium 2. A silently-dropped typo or an ID that already
    has a completed retry row would double-count into both
    ``completed_retries`` and ``aborted_retries``, inflating the
    headline rejection metric. Fail fast with a message that names
    the offending ID so the user fixes the CLI flag rather than the
    artifact.
    """
    by_treat = _by_instance(treatment_rows)
    treat_ids = set(by_treat.keys())
    missing = sorted(aborted_retries - treat_ids)
    if missing:
        raise ValueError(
            "--aborted-treatment-retry IDs not found in treatment run: "
            + ", ".join(missing)
            + f"\n  available IDs: {sorted(treat_ids)}"
        )
    # Instances that already have a completed retry (attempt > 1) can't
    # also be marked aborted — those double-count.
    has_completed = {
        iid for iid in aborted_retries if max(r.get("attempt", 1) for r in by_treat[iid]) > 1
    }
    if has_completed:
        raise ValueError(
            "--aborted-treatment-retry IDs that already have a "
            "completed retry row (attempt > 1 in output.jsonl): "
            + ", ".join(sorted(has_completed))
            + "\n  these would be double-counted. Remove them from the flag."
        )


def build_artifact(
    baseline_rows: list[dict],
    treatment_rows: list[dict],
    aborted_treatment_retries: set[str],
    model: str,
    baseline_eval_report: dict | None = None,
    treatment_eval_report: dict | None = None,
    baseline_cert_fires: dict[str, dict] | None = None,
    treatment_cert_fires: dict[str, dict] | None = None,
) -> dict:
    base_ids = set(r["instance_id"] for r in baseline_rows)
    treat_ids = set(r["instance_id"] for r in treatment_rows)
    if base_ids != treat_ids:
        raise ValueError(
            f"instance_id set mismatch: "
            f"baseline-only={sorted(base_ids - treat_ids)}, "
            f"treatment-only={sorted(treat_ids - base_ids)}"
        )

    _validate_aborted_retries(treatment_rows, aborted_treatment_retries)

    # Roborev #844 Medium 2: eval reports must be supplied as a pair.
    # A one-sided artifact (baseline scored, treatment not — or vice
    # versa) would render the markdown in a broken half-state because
    # the ``has_eval`` check keys off one side. Fail fast instead.
    if (baseline_eval_report is None) != (treatment_eval_report is None):
        raise ValueError(
            "--baseline-eval-report and --treatment-eval-report must be "
            "supplied together (or neither). Got "
            f"baseline={'set' if baseline_eval_report else 'unset'}, "
            f"treatment={'set' if treatment_eval_report else 'unset'}."
        )

    # Roborev #844 Medium 1: validate eval reports cover every row's
    # instance_id. A mistyped/stale report would silently mark
    # uncovered rows as "incomplete" and skew pass@1.
    if baseline_eval_report is not None:
        _validate_eval_report_covers_rows(baseline_eval_report, baseline_rows, "baseline")
    if treatment_eval_report is not None:
        _validate_eval_report_covers_rows(treatment_eval_report, treatment_rows, "treatment")

    by_base = _by_instance(baseline_rows)
    by_treat = _by_instance(treatment_rows)

    return {
        "run_id": str(uuid.uuid4()),
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "model": model,
        "conditions": ["baseline", "operon_stagnation"],
        "scope": {
            "n_instances": len(base_ids),
            "instance_ids": sorted(base_ids),
        },
        "summary": {
            "baseline": _aggregate(baseline_rows, set(), baseline_eval_report, baseline_cert_fires),
            "operon_stagnation": _aggregate(
                treatment_rows,
                aborted_treatment_retries,
                treatment_eval_report,
                treatment_cert_fires,
            ),
        },
        "per_instance": [
            {
                "instance_id": iid,
                "baseline_max_attempt": max(r.get("attempt", 1) for r in by_base[iid]),
                "treatment_max_attempt": max(r.get("attempt", 1) for r in by_treat[iid]),
                "treatment_retry_aborted": iid in aborted_treatment_retries,
                "baseline_cumulative_cost_usd": round(
                    _cum_metric(by_base[iid], "metrics.accumulated_cost"), 4
                ),
                "treatment_cumulative_cost_usd": round(
                    _cum_metric(by_treat[iid], "metrics.accumulated_cost"), 4
                ),
                "baseline_final_patch_len": len(
                    (_final_row(by_base[iid]).get("test_result") or {}).get("git_patch") or ""
                ),
                "treatment_final_patch_len": len(
                    (_final_row(by_treat[iid]).get("test_result") or {}).get("git_patch") or ""
                ),
                # Eval status — populated only when reports are supplied.
                # ``None`` signals "not scored" (vs False = unresolved).
                "baseline_eval_status": _eval_status_for(iid, baseline_eval_report),
                "treatment_eval_status": _eval_status_for(iid, treatment_eval_report),
                # Certificate records — populated when a logs dir carrying
                # ``[CERT-FIRE]`` lines was supplied. ``None`` means the
                # logs-dir wasn't passed or the instance had no cert fire.
                # Baseline runs don't use OperonStagnationCritic so these
                # will always be None for baseline; schema stays symmetric.
                "baseline_certificate": (baseline_cert_fires or {}).get(iid),
                "treatment_certificate": (treatment_cert_fires or {}).get(iid),
            }
            for iid in sorted(base_ids)
        ],
    }


def generate_markdown(artifact: dict, extra_caveats: list[str] | None = None) -> str:
    s = artifact["summary"]
    bs, ts = s["baseline"], s["operon_stagnation"]
    scope = artifact["scope"]
    delta_total_dollars = round(
        ts["cumulative_cost_usd"]["total"] - bs["cumulative_cost_usd"]["total"], 2
    )
    delta_pct = round(
        100
        * (ts["cumulative_cost_usd"]["total"] - bs["cumulative_cost_usd"]["total"])
        / bs["cumulative_cost_usd"]["total"]
    )
    # Per-completed-retry cost: excludes aborted retries from the
    # denominator to avoid bias from their undercounted spend
    # (roborev #835 Medium 1). Uses total retry rounds (not
    # instances-with-retry) so an attempt=3 instance contributes 2
    # rounds, not 1 (roborev #837 Medium 1). Report "—" if no
    # completed retry rounds.
    per_round = (
        round(delta_total_dollars / ts["total_completed_retry_rounds"], 2)
        if ts["total_completed_retry_rounds"] > 0
        else None
    )
    per_round_str = (
        f"**${per_round:.2f}** per completed retry round "
        f"(= total cost delta / total_completed_retry_rounds; aborted retries excluded to avoid bias from undercounted spend)"
        if per_round is not None
        else "—"
    )

    repo_breakdown = Counter(iid.split("__")[0] for iid in scope["instance_ids"])
    repo_sorted = sorted(repo_breakdown.items(), key=lambda kv: -kv[1])
    repo_phrase = " + ".join(f"{n} {r}" for r, n in repo_sorted)
    dominant_repo = repo_sorted[0][0] if repo_sorted else "unknown"

    # Whether the eval step was run (affects columns + prose).
    # ``build_artifact`` enforces that eval reports are paired, but
    # the markdown renderer checks both sides defensively — if one is
    # missing, it falls back to the infer-only layout rather than
    # rendering an asymmetric half-eval view (roborev #844 Medium 2).
    has_eval = "resolved_count" in bs and "resolved_count" in ts
    # Whether on-disk cert records were supplied. Keyed on the
    # treatment summary alone because baseline uses finish_with_patch
    # and will always have 0 cert fires — a "treatment had cert log
    # ingestion but baseline didn't" state is meaningful. Specifically:
    # the summary field is only added by ``_aggregate`` when
    # ``cert_fires`` is non-None (empty dict is fine; it just gives 0),
    # so treating "key present at all" as the signal works.
    has_cert_evidence = "certificates_emitted" in ts

    def _status_emoji(status: str | None) -> str:
        if status is None:
            return "—"
        return {
            "resolved": "✅",
            "unresolved": "❌",
            "empty_patch": "∅",
            "error": "⚠️",
            "incomplete": "?",
        }.get(status, f"?{status}")

    rows = []
    for p in artifact["per_instance"]:
        t_att = p["treatment_max_attempt"]
        t_aborted = p.get("treatment_retry_aborted", False)
        att_disp = (
            "**1 → aborted 2**" if t_aborted else (f"**{t_att}**" if t_att > 1 else str(t_att))
        )
        cost_disp = (
            f"**${p['treatment_cumulative_cost_usd']:.2f}**"
            if t_att > 1 or t_aborted
            else f"${p['treatment_cumulative_cost_usd']:.2f}"
        )
        line = (
            f"| `{p['instance_id']}` | {p['baseline_max_attempt']} | {att_disp} "
            f"| ${p['baseline_cumulative_cost_usd']:.2f} | {cost_disp} "
        )
        if has_eval:
            line += (
                f"| {_status_emoji(p.get('baseline_eval_status'))} "
                f"| {_status_emoji(p.get('treatment_eval_status'))} "
            )
        if has_cert_evidence:
            treat_cert = p.get("treatment_certificate")
            cert_marker = "✓" if treat_cert else "·"
            line += f"| {cert_marker} "
        line += "|"
        rows.append(line)

    if has_cert_evidence:
        cert_caveat_text = (
            f"**Certificate evidence via side-channel log.** "
            f"``OperonStagnationCritic`` emits a ``[CERT-FIRE]`` line on its "
            f"stdout when a certificate fires (v0.1.0a3+). The benchmarks "
            f"runner captures those lines into "
            f"``logs/instance_<iid>.output.log``; "
            f"``scripts/generate_delta_artifact.py --*-logs-dir`` parses "
            f"them to populate ``treatment_certificate`` per instance + "
            f"``certificates_emitted`` summary. **{ts['certificates_emitted']} "
            f"of {ts['n_instances']} treatment instances have on-disk cert "
            f"records** on this run. "
            f"The openhands-sdk still doesn't serialize ``CriticResult.metadata`` "
            f"into ``MessageEvent`` / ``ActionEvent`` history; the stdout log "
            f"is the deliberate workaround."
        )
    else:
        cert_caveat_text = '**Certificate metadata not serialized.** `OperonStagnationCritic` emits `CriticResult.metadata` with `certificate_theorem="behavioral_stability_windowed"` on stagnation fires. That metadata field is **not captured** in the serialized event history (`critic_result` field on `MessageEvent` / `ActionEvent` is `null` throughout). Known openhands-sdk serialization gap, not a critic bug. Critic firing is inferred from Attempt-2 presence (for completed retries) or from the pinned aborted-retry list (for timed-out ones), not from on-disk certificate records. Fixing this requires either patching the SDK or adding a side-channel log from inside `OperonStagnationCritic.evaluate()`.'

    caveats = [
        f"**Sample bias.** n={bs['n_instances']}, {repo_phrase}. Original plan called for n=30 alphabetic slice. The Docker-Desktop-on-Mac apt-signature build bug blocked all matplotlib/sympy/flask/pytest instances, leaving a {dominant_repo}-heavy subset. Not representative of full SWE-bench-lite; generalizability limited. Run on a Linux host or OpenHands remote runtime for an unbiased slice.",
        cert_caveat_text,
    ]
    # Only keep the "`resolved` not computed" caveat when the eval
    # step hasn't been run. When it has, the artifact carries real
    # pass@1 numbers and the caveat is obsolete.
    if not has_eval:
        caveats.insert(
            1,
            "**`resolved` not computed.** Both conditions produced non-empty `git_patch` outputs, but the SWE-bench patch-evaluation step (which applies patches and runs `FAIL_TO_PASS` / `PASS_TO_PASS` tests) was not run. The critic-retry delta is a behavioral signal; the underlying pass@1 comparison requires the evaluation pipeline.",
        )
    if ts["aborted_retries"] > 0:
        aborted_ids = ", ".join(f"`{iid}`" for iid in ts["aborted_retry_instance_ids"])
        caveats.append(
            f"**Aborted retries ({ts['aborted_retries']}).** "
            f"Critic rejected Attempt-1 for {aborted_ids}, triggered Attempt-2 "
            f"that hit the 60-min per-attempt ceiling and was killed. Counted as "
            f"critic rejection(s) in the headline rate (`aborted_retries: "
            f"{ts['aborted_retries']}`), but no Attempt-2 row exists in "
            f"`output.jsonl` — their cumulative cost figures undercount what a "
            f"clean retry would have produced. The per-retry cost figure "
            f"excludes them from the denominator for this reason."
        )
    caveats.append(
        "**Single model.** Paper 4's `threshold=0.2` epiplexic-integral cutoff was validated on different model families (sentence-MiniLM embeddings at n=300 trials). Behavior on the experiment model's trajectories may have different stagnation dynamics than the validation set."
    )
    caveats.extend(extra_caveats or [])

    caveats_md = "\n\n".join(f"{i + 1}. {c}" for i, c in enumerate(caveats))

    # Optional pass@1 row + correctness prose. Only rendered when the
    # eval step was run. Using a helper instead of an inline ternary
    # because the pass_at_1 formatting needs to handle the "None"
    # (some-rows-unscored) branch from _aggregate.
    def _fmt_passk(cond: dict) -> str:
        p = cond.get("pass_at_1")
        if p is None:
            note = cond.get("pass_at_1_note", "not computed")
            return f"— ({note})"
        return f"{p * 100:.0f}%"

    if has_eval:
        # Retry-flip rollup: did treatment retries change the outcome
        # per instance vs baseline? (improved = unresolved → resolved,
        # broke = resolved → unresolved).
        retried_per_instance = [
            p
            for p in artifact["per_instance"]
            if p["treatment_max_attempt"] > 1 or p.get("treatment_retry_aborted")
        ]
        flipped_improved = sum(
            1
            for p in retried_per_instance
            if p.get("baseline_eval_status") == "unresolved"
            and p.get("treatment_eval_status") == "resolved"
        )
        flipped_broke = sum(
            1
            for p in retried_per_instance
            if p.get("baseline_eval_status") == "resolved"
            and p.get("treatment_eval_status") == "unresolved"
        )
        # pass@1 delta — only meaningful when both sides have numeric
        # pass_at_1 values (not None).
        bp = bs.get("pass_at_1")
        tp = ts.get("pass_at_1")
        if bp is None or tp is None:
            pass_at_1_delta = "—"
        else:
            diff_pp = round((tp - bp) * 100)
            pass_at_1_delta = f"**{diff_pp:+d} pp**"
        base_resolved = bs.get("resolved_count", "—")
        treat_resolved = ts.get("resolved_count", "—")
        base_unresolved = bs.get("unresolved_count", "—")
        treat_unresolved = ts.get("unresolved_count", "—")
        pass_at_1_row = (
            "\n"
            f"| **pass@1**                      | {_fmt_passk(bs):>8} | {_fmt_passk(ts):>17} | {pass_at_1_delta} |\n"
            f"| &nbsp;&nbsp;&nbsp;resolved       | {base_resolved:>8} | {treat_resolved:>17} | — |\n"
            f"| &nbsp;&nbsp;&nbsp;unresolved     | {base_unresolved:>8} | {treat_unresolved:>17} | — |"
        )
        n_retried = len(retried_per_instance)
        retry_flip_line = (
            f"\n\nRetry-flip accounting (treatment retried {n_retried} instance(s)): "
            f"**{flipped_improved}** improved (unresolved → resolved), "
            f"**{flipped_broke}** broke (resolved → unresolved), "
            f"{n_retried - flipped_improved - flipped_broke} unchanged."
        )
    else:
        pass_at_1_row = ""
        retry_flip_line = ""

    # Optional cert-emission headline row. ``_aggregate`` already
    # computed the count when ``cert_fires`` was non-None; here we
    # just format. Baseline will always be 0 on finish_with_patch
    # runs since that critic doesn't emit certificates.
    if has_cert_evidence:
        cert_row = (
            f"\n| Certificates emitted            | {bs.get('certificates_emitted', 0):>8} | "
            f"{ts['certificates_emitted']:>17} | "
            f"**+{ts['certificates_emitted'] - bs.get('certificates_emitted', 0)}** |"
        )
    else:
        cert_row = ""

    # Instance-level table header — adds a "Resolved?" column pair
    # when the eval step was run, and a treatment-side "Cert" column
    # when on-disk cert evidence is available.
    header_cols = ["Instance", "Base attempts", "Treat attempts", "Base cost", "Treat cost"]
    align_cols = ["", "--------------:", "---------------:", "----------:", "-----------:"]
    if has_eval:
        header_cols += ["Base resolved", "Treat resolved"]
        align_cols += [":-------------:", ":--------------:"]
    if has_cert_evidence:
        header_cols.append("Treat cert")
        align_cols.append(":----------:")
    instance_table_header = (
        "| " + " | ".join(header_cols) + " |\n|----------|" + "|".join(align_cols[1:]) + "|"
    )

    # "Does/Does not prove" + "Next steps" — gate claims on ``has_eval``
    # (roborev #846 Medium). Unconditional "from inference through
    # patch-evaluation" prose contradicts the caveats when the eval
    # step wasn't run.
    critic_rate_prose = (
        f"The structural critic exhibits a measurably different rejection pattern "
        f"from the default LLM critic "
        f"({ts['instances_with_rejection_rate'] * 100:.0f}% of instances vs "
        f"{bs['instances_with_rejection_rate'] * 100:.0f}%)."
    )
    overhead_prose = (
        f"Per-retry-round overhead: {per_round_str}. Full-slice overhead: **+{delta_pct}%**."
    )
    # Which caveat number covers certificates depends on whether the
    # "`resolved` not computed" caveat is present.
    cert_caveat_num = 2 if has_eval else 3
    # When on-disk cert evidence is present, "does not provide cert
    # evidence" is a false claim — flip to "does".
    if has_cert_evidence:
        cert_evidence_does = (
            f"- Surface {ts['certificates_emitted']} on-disk certificate record(s) "
            f"via the ``[CERT-FIRE]`` stdout log parsed from "
            f"``logs/instance_<iid>.output.log`` (see caveat {cert_caveat_num})."
        )
        cert_evidence_doesnot = ""
    else:
        cert_evidence_does = ""
        cert_evidence_doesnot = (
            f"\n- Provide on-disk certificate evidence "
            f"(openhands-sdk serialization gap; see caveat {cert_caveat_num})."
        )
    if has_eval:
        does_does_not_section = (
            "**Does:**\n"
            "- The harness runs end-to-end with `CodeActAgent` + `OperonStagnationCritic` on real SWE-bench instances, from inference through patch-evaluation.\n"
            f"- {critic_rate_prose}\n"
            f"- {overhead_prose}"
            + (f"\n{cert_evidence_does}" if cert_evidence_does else "")
            + "\n\n"
            "**Does not:**\n"
            "- Generalize beyond the selection-biased instance subset (see caveat 1)."
            + cert_evidence_doesnot
        )
        if has_cert_evidence:
            next_steps_section = (
                "1. **Rerun on a Linux host or `--workspace remote`** to get an unbiased 30-instance slice.\n"
                "2. **Broader scope:** once the harness is stable, repeat on Claude Sonnet 4.6 (original plan default) and on SWE-bench-Verified."
            )
        else:
            next_steps_section = (
                "1. **Fix the certificate-metadata serialization** so downstream consumers can populate `certificate_emitted`, `certificate_theorem`, `cert_evidence_n` without relying on the retry-count proxy.\n"
                "2. **Rerun on a Linux host or `--workspace remote`** to get an unbiased 30-instance slice.\n"
                "3. **Broader scope:** once the harness is stable, repeat on Claude Sonnet 4.6 (original plan default) and on SWE-bench-Verified."
            )
    else:
        does_does_not_section = (
            "**Does:**\n"
            "- The harness runs end-to-end with `CodeActAgent` + `OperonStagnationCritic` on real SWE-bench instances (inference only; the SWE-bench patch-evaluation step was not run on this artifact).\n"
            f"- {critic_rate_prose}\n"
            f"- {overhead_prose}"
            + (f"\n{cert_evidence_does}" if cert_evidence_does else "")
            + "\n\n"
            "**Does not:**\n"
            "- Establish whether retries improve patch correctness — that needs the SWE-bench evaluation step (see caveat 2).\n"
            "- Generalize beyond the selection-biased instance subset (see caveat 1)."
            + cert_evidence_doesnot
        )
        next_steps_section = (
            "1. **Run SWE-bench evaluation** on both `output.jsonl` files (dedupe treatment first via `scripts/dedupe_for_eval.py`), then pass the resulting `.report.json` files to this script via `--baseline-eval-report` / `--treatment-eval-report` to populate `pass_at_1`.\n"
            "2. **Fix the certificate-metadata serialization** so downstream consumers can populate `certificate_emitted`, `certificate_theorem`, `cert_evidence_n` without relying on the retry-count proxy.\n"
            "3. **Rerun on a Linux host or `--workspace remote`** to get an unbiased 30-instance slice.\n"
            "4. **Broader scope:** once the harness is stable, repeat on Claude Sonnet 4.6 (original plan default) and on SWE-bench-Verified."
        )

    return f"""# SWE-bench-lite delta: baseline vs OperonStagnationCritic

**Scope:** n={bs["n_instances"]}, {artifact["model"]}, SWE-bench-lite `test` split, local Docker workspace on Apple Silicon.
**Conditions:**
- `baseline`: OpenHands `CodeActAgent` + default `AgentFinishedCritic` (finish_with_patch preset).
- `operon_stagnation`: same agent + `OperonStagnationCritic(threshold=0.2, window=10, critical_duration=3)`.

## Headline numbers

| Metric                          | Baseline | Operon stagnation | Δ          |
|---------------------------------|---------:|------------------:|-----------:|
| Instances                       | {bs["n_instances"]:>8} | {ts["n_instances"]:>17} | —          |
| Instances with ≥1 rejection     | {bs["instances_with_rejection"]:>8} | {ts["instances_with_rejection"]:>17} | **+{ts["instances_with_rejection"] - bs["instances_with_rejection"]}** |
| &nbsp;&nbsp;&nbsp;with completed retry | — | {ts["instances_with_completed_retry"]:>17} | — |
| &nbsp;&nbsp;&nbsp;with aborted retry (timeout) | — | {ts["aborted_retries"]:>17} | — |
| Per-instance rejection rate     | {bs["instances_with_rejection_rate"] * 100:>7.0f}% | {ts["instances_with_rejection_rate"] * 100:>16.0f}% | **+{(ts["instances_with_rejection_rate"] - bs["instances_with_rejection_rate"]) * 100:.0f} pp** |
| Total retry rounds              | {bs["total_retry_rounds"]:>8} | {ts["total_retry_rounds"]:>17} | **+{ts["total_retry_rounds"] - bs["total_retry_rounds"]}** |
| Cumulative cost (USD)           | ${bs["cumulative_cost_usd"]["total"]:>6.2f} | ${ts["cumulative_cost_usd"]["total"]:>16.2f} | **+{delta_pct}%** (+${delta_total_dollars:.2f}) |
| Mean final patch length (chars) | {bs["final_patch_len"]["mean"]:>8.0f} | {ts["final_patch_len"]["mean"]:>17.0f} | — |
| Mean final history events       | {bs["final_n_events"]["mean"]:>8.1f} | {ts["final_n_events"]["mean"]:>17.1f} | — |{pass_at_1_row}{cert_row}

Per-round budget overhead: {per_round_str}.{retry_flip_line}

**Raw artifact:** [`swebench_lite_delta.json`](./swebench_lite_delta.json). Reproduce via `scripts/generate_delta_artifact.py`.

## What the critic did

On the {bs["n_instances"]}-instance shared slice, the default `finish_with_patch` critic accepted every first-attempt patch. `OperonStagnationCritic` rejected the first-attempt patch on {ts["instances_with_rejection"]} of {ts["n_instances"]} instances ({ts["instances_with_rejection_rate"] * 100:.0f}%), producing {ts["total_retry_rounds"]} total retry round(s): {ts["total_completed_retry_rounds"]} completed, {ts["aborted_retries"]} aborted on timeout.

That's a **measurable behavioral delta** — the structural critic reaches different "done" decisions than an LLM-judged critic on the same agent trajectories. The cost is a **+{delta_pct}% budget overhead** for the slice.

## Instance-level table

All numbers read directly from [`swebench_lite_delta.json`](./swebench_lite_delta.json) `per_instance[*]`.

{instance_table_header}
{chr(10).join(rows)}

`1 → aborted 2` = critic rejected Attempt-1, Attempt-2 started but timed out before completion (per-instance `treatment_retry_aborted: true` in the JSON).

## Caveats (read these before citing)

{caveats_md}

## What this does and does not prove

{does_does_not_section}

## Next steps

{next_steps_section}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline run directory (contains output.jsonl in a nested model subdir).",
    )
    parser.add_argument(
        "--treatment", type=Path, required=True, help="Path to treatment run directory."
    )
    parser.add_argument(
        "--aborted-treatment-retry",
        action="append",
        default=[],
        help="Instance ID whose treatment Attempt-2 aborted (timed out etc.). Repeatable.",
    )
    parser.add_argument(
        "--model", default="openai/gpt-5", help="Model name recorded in the artifact."
    )
    parser.add_argument(
        "--baseline-eval-report",
        type=Path,
        default=None,
        help=(
            "Path to the baseline's SWE-bench ``.report.json`` "
            "(produced by ``benchmarks.swebench.eval_infer``). When "
            "supplied, the artifact's summary carries ``pass_at_1`` + "
            "``resolved_count`` + ``unresolved_count`` and per-instance "
            "rows carry ``baseline_eval_status``. "
            "MUST be supplied alongside ``--treatment-eval-report`` — "
            "one-sided usage is rejected to avoid asymmetric markdown."
        ),
    )
    parser.add_argument(
        "--treatment-eval-report",
        type=Path,
        default=None,
        help=(
            "Path to the treatment's SWE-bench ``.report.json``. "
            "See ``--baseline-eval-report``. MUST be supplied together "
            "with ``--baseline-eval-report``."
        ),
    )
    parser.add_argument(
        "--baseline-logs-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to the baseline run's ``logs/`` directory "
            "(contains ``instance_<iid>.output.log`` files). If provided, "
            "the script parses ``[CERT-FIRE]`` lines emitted by "
            "``OperonStagnationCritic.evaluate()`` to populate "
            "``baseline_certificate`` per-instance and the "
            "``certificates_emitted`` summary rollup. Baseline runs use "
            "``finish_with_patch`` and will normally yield zero cert "
            "fires — the flag is symmetric for schema consistency."
        ),
    )
    parser.add_argument(
        "--treatment-logs-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to the treatment run's ``logs/`` directory. "
            "See ``--baseline-logs-dir``. Provides the on-disk cert "
            "evidence that replaces the retry-count proxy."
        ),
    )
    parser.add_argument(
        "--out-json", type=Path, required=True, help="Output path for the JSON artifact."
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional output path for the markdown writeup. If omitted, no md is written.",
    )
    args = parser.parse_args()

    # Argparse-level pair check so the error surfaces at CLI parse time
    # with argparse's standard "error: ..." exit code 2, instead of
    # deep inside ``build_artifact`` (roborev #846 Low).
    if bool(args.baseline_eval_report) != bool(args.treatment_eval_report):
        parser.error(
            "--baseline-eval-report and --treatment-eval-report must be "
            "supplied together (or neither)."
        )

    # Load cross-attempt rows so cumulative cost/tokens are correct
    # even when the runner consolidates ``output.jsonl`` to max-attempt
    # rows. See ``_load_all_attempt_rows`` docstring.
    baseline_rows = _load_all_attempt_rows(args.baseline)
    treatment_rows = _load_all_attempt_rows(args.treatment)

    baseline_eval_report = (
        _load_eval_report(args.baseline_eval_report) if args.baseline_eval_report else None
    )
    treatment_eval_report = (
        _load_eval_report(args.treatment_eval_report) if args.treatment_eval_report else None
    )

    # Cert-fire logs are independently optional (no pair requirement —
    # it's perfectly fine to have one side's logs and not the other's,
    # and the markdown handles each condition's summary separately).
    #
    # Roborev #848 Medium: when a logs dir IS supplied, validate it
    # against the run rows. A stale or mistyped directory would
    # silently degrade to ``certificates_emitted: 0`` without any
    # error — same bug class that eval-report validation catches.
    if args.baseline_logs_dir is not None:
        _validate_logs_dir_covers_rows(args.baseline_logs_dir, baseline_rows, "baseline")
    if args.treatment_logs_dir is not None:
        _validate_logs_dir_covers_rows(args.treatment_logs_dir, treatment_rows, "treatment")

    baseline_cert_fires = (
        _load_cert_fires_from_logs(args.baseline_logs_dir) if args.baseline_logs_dir else None
    )
    treatment_cert_fires = (
        _load_cert_fires_from_logs(args.treatment_logs_dir) if args.treatment_logs_dir else None
    )

    artifact = build_artifact(
        baseline_rows,
        treatment_rows,
        set(args.aborted_treatment_retry),
        model=args.model,
        baseline_eval_report=baseline_eval_report,
        treatment_eval_report=treatment_eval_report,
        baseline_cert_fires=baseline_cert_fires,
        treatment_cert_fires=treatment_cert_fires,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2) + "\n")

    if args.out_md is not None:
        args.out_md.write_text(generate_markdown(artifact))

    print(
        json.dumps(
            {"out_json": str(args.out_json), "out_md": str(args.out_md) if args.out_md else None}
        )
    )


if __name__ == "__main__":
    main()
