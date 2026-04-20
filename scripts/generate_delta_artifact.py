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


def _aggregate(
    rows: list[dict],
    aborted_retries: set[str],
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

    # Retry accounting — roborev #837 Medium 1. Distinguish:
    # - ``instances_with_completed_retry``: count of instances where
    #   max_attempt > 1 (at least one critic-rejection-and-retry loop
    #   closed successfully). This is what scales the retry rate.
    # - ``total_completed_retries``: actual count of retry rounds
    #   across instances — ``(max_attempt - 1)`` summed. Matters when
    #   an instance reaches attempt=3 or higher (2+ retries on that
    #   single instance). In this slice every retried instance
    #   completed exactly one retry, so the two numbers coincide,
    #   but the denominator for the per-retry cost figure should
    #   divide by total retries, not instances-with-retry, to stay
    #   correct under higher-attempt runs.
    instances_with_completed_retry = sum(1 for a in max_attempts if a > 1)
    total_completed_retries = sum(a - 1 for a in max_attempts if a > 1)
    aborted_in_scope = sorted(aborted_retries & set(by_iid.keys()))
    # ``critic_rejections`` = total retry rounds (completed + aborted).
    # Each aborted retry is one rejection; we can't count multiple
    # aborted rounds per instance since no row is written.
    critic_rejections = total_completed_retries + len(aborted_in_scope)

    return {
        "n_instances": len(by_iid),
        "final_patch_len": _safe_agg(patch_lens),
        "final_n_events": _safe_agg(n_events),
        "cumulative_cost_usd": {"total": round(sum(cum_costs), 2), **_safe_agg(cum_costs)},
        "cumulative_prompt_tokens": {"total": sum(cum_prompt), **_safe_agg(cum_prompt)},
        "cumulative_completion_tokens": {"total": sum(cum_comp), **_safe_agg(cum_comp)},
        "cumulative_reasoning_tokens": {"total": sum(cum_reason), **_safe_agg(cum_reason)},
        "critic_rejections": critic_rejections,
        "critic_rejection_rate": round(critic_rejections / len(by_iid), 3),
        "completed_retries": total_completed_retries,
        "instances_with_completed_retry": instances_with_completed_retry,
        "aborted_retries": len(aborted_in_scope),
        "aborted_retry_instance_ids": aborted_in_scope,
        "max_attempt_histogram": dict(Counter(max_attempts)),
    }


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
            "baseline": _aggregate(baseline_rows, set()),
            "operon_stagnation": _aggregate(treatment_rows, aborted_treatment_retries),
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
    # (roborev #835 Medium 1). If there are no completed retries,
    # report "—" rather than divide by zero.
    per_completed = (
        round(delta_total_dollars / ts["completed_retries"], 2)
        if ts["completed_retries"] > 0
        else None
    )
    per_completed_str = (
        f"**${per_completed:.2f}** per completed retry "
        f"(= total cost delta / completed_retries; aborted retries excluded to avoid bias from undercounted spend)"
        if per_completed is not None
        else "—"
    )

    repo_breakdown = Counter(iid.split("__")[0] for iid in scope["instance_ids"])
    repo_sorted = sorted(repo_breakdown.items(), key=lambda kv: -kv[1])
    repo_phrase = " + ".join(f"{n} {r}" for r, n in repo_sorted)
    dominant_repo = repo_sorted[0][0] if repo_sorted else "unknown"

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
        rows.append(
            f"| `{p['instance_id']}` | {p['baseline_max_attempt']} | {att_disp} "
            f"| ${p['baseline_cumulative_cost_usd']:.2f} | {cost_disp} |"
        )

    caveats = [
        f"**Sample bias.** n={bs['n_instances']}, {repo_phrase}. Original plan called for n=30 alphabetic slice. The Docker-Desktop-on-Mac apt-signature build bug blocked all matplotlib/sympy/flask/pytest instances, leaving a {dominant_repo}-heavy subset. Not representative of full SWE-bench-lite; generalizability limited. Run on a Linux host or OpenHands remote runtime for an unbiased slice.",
        "**`resolved` not computed.** Both conditions produced non-empty `git_patch` outputs, but the SWE-bench patch-evaluation step (which applies patches and runs `FAIL_TO_PASS` / `PASS_TO_PASS` tests) was not run. The critic-retry delta is a behavioral signal; the underlying pass@1 comparison requires the evaluation pipeline.",
        '**Certificate metadata not serialized.** `OperonStagnationCritic` emits `CriticResult.metadata` with `certificate_theorem="behavioral_stability_windowed"` on stagnation fires. That metadata field is **not captured** in the serialized event history (`critic_result` field on `MessageEvent` / `ActionEvent` is `null` throughout). Known openhands-sdk serialization gap, not a critic bug. Critic firing is inferred from Attempt-2 presence (for completed retries) or from the pinned aborted-retry list (for timed-out ones), not from on-disk certificate records. Fixing this requires either patching the SDK or adding a side-channel log from inside `OperonStagnationCritic.evaluate()`.',
    ]
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

    return f"""# SWE-bench-lite delta: baseline vs OperonStagnationCritic

**Scope:** n={bs["n_instances"]}, {artifact["model"]}, SWE-bench-lite `test` split, local Docker workspace on Apple Silicon.
**Conditions:**
- `baseline`: OpenHands `CodeActAgent` + default `AgentFinishedCritic` (finish_with_patch preset).
- `operon_stagnation`: same agent + `OperonStagnationCritic(threshold=0.2, window=10, critical_duration=3)`.

## Headline numbers

| Metric                          | Baseline | Operon stagnation | Δ          |
|---------------------------------|---------:|------------------:|-----------:|
| Instances                       | {bs["n_instances"]:>8} | {ts["n_instances"]:>17} | —          |
| Critic rejections               | {bs["critic_rejections"]:>8} | {ts["critic_rejections"]:>17} | **+{ts["critic_rejections"]}** |
| &nbsp;&nbsp;&nbsp;completed retries | — | {ts["completed_retries"]:>17} | — |
| &nbsp;&nbsp;&nbsp;aborted retries (timeout) | — | {ts["aborted_retries"]:>17} | — |
| Critic-rejection rate           | {bs["critic_rejection_rate"] * 100:>7.0f}% | {ts["critic_rejection_rate"] * 100:>16.0f}% | **+{(ts["critic_rejection_rate"] - bs["critic_rejection_rate"]) * 100:.0f} pp** |
| Cumulative cost (USD)           | ${bs["cumulative_cost_usd"]["total"]:>6.2f} | ${ts["cumulative_cost_usd"]["total"]:>16.2f} | **+{delta_pct}%** (+${delta_total_dollars:.2f}) |
| Mean final patch length (chars) | {bs["final_patch_len"]["mean"]:>8.0f} | {ts["final_patch_len"]["mean"]:>17.0f} | — |
| Mean final history events       | {bs["final_n_events"]["mean"]:>8.1f} | {ts["final_n_events"]["mean"]:>17.1f} | — |

Per-rejection budget overhead: {per_completed_str}.

**Raw artifact:** [`swebench_lite_delta.json`](./swebench_lite_delta.json). Reproduce via `scripts/generate_delta_artifact.py`.

## What the critic did

On the {bs["n_instances"]}-instance shared slice, the default `finish_with_patch` critic accepted every first-attempt patch. `OperonStagnationCritic` rejected {ts["critic_rejections"]} of {ts["n_instances"]} first-attempt patches ({ts["critic_rejection_rate"] * 100:.0f}%): {ts["completed_retries"]} triggered a second refinement iteration that completed, and {ts["aborted_retries"]} triggered a retry that timed out before completing.

That's a **measurable behavioral delta** — the structural critic reaches different "done" decisions than an LLM-judged critic on the same agent trajectories. The cost is a **+{delta_pct}% budget overhead** for the slice. Whether retries *improve* correctness against the gold fix is not measured here; see caveats.

## Instance-level table

All numbers read directly from [`swebench_lite_delta.json`](./swebench_lite_delta.json) `per_instance[*]`.

| Instance | Baseline attempts | Treatment attempts | Baseline cost | Treatment cost |
|----------|------------------:|-------------------:|--------------:|---------------:|
{chr(10).join(rows)}

`1 → aborted 2` = critic rejected Attempt-1, Attempt-2 started but timed out before completion (per-instance `treatment_retry_aborted: true` in the JSON).

## Caveats (read these before citing)

{caveats_md}

## What this does and does not prove

**Does:**
- The harness runs end-to-end with `CodeActAgent` + `OperonStagnationCritic` on real SWE-bench instances.
- The structural critic exhibits a measurably different rejection pattern from the default LLM critic ({ts["critic_rejection_rate"] * 100:.0f}% vs {bs["critic_rejection_rate"] * 100:.0f}%).
- Per-completed-retry overhead: {per_completed_str}. Full-slice overhead: **+{delta_pct}%**.

**Does not:**
- Establish whether retries improve patch correctness — that needs the SWE-bench evaluation step.
- Generalize beyond the selection-biased instance subset (see caveat 1).
- Provide on-disk certificate evidence (openhands-sdk serialization gap; pending follow-up).

## Next steps

1. **Run SWE-bench evaluation** on both `output.jsonl` files to compute per-condition pass@1, resolved counts, `FAIL_TO_PASS` breakdown. This is the number the plan's success criteria calls for.
2. **Fix the certificate-metadata serialization** so downstream consumers can populate `certificate_emitted`, `certificate_theorem`, `cert_evidence_n` without relying on the retry-count proxy.
3. **Rerun on a Linux host or `--workspace remote`** to get an unbiased 30-instance slice.
4. **Broader scope:** once the harness is stable, repeat on Claude Sonnet 4.6 (original plan default) and on SWE-bench-Verified.
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
        "--out-json", type=Path, required=True, help="Output path for the JSON artifact."
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional output path for the markdown writeup. If omitted, no md is written.",
    )
    args = parser.parse_args()

    baseline_rows = _load_jsonl(_find_output_jsonl(args.baseline))
    treatment_rows = _load_jsonl(_find_output_jsonl(args.treatment))

    artifact = build_artifact(
        baseline_rows,
        treatment_rows,
        set(args.aborted_treatment_retry),
        model=args.model,
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
