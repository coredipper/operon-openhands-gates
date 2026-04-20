# SWE-bench-lite delta: baseline vs OperonStagnationCritic

**Scope:** n=10, openai/gpt-5, SWE-bench-lite `test` split, local Docker workspace on Apple Silicon.
**Conditions:**
- `baseline`: OpenHands `CodeActAgent` + default `AgentFinishedCritic` (finish_with_patch preset).
- `operon_stagnation`: same agent + `OperonStagnationCritic(threshold=0.2, window=10, critical_duration=3)`.

## Headline numbers

| Metric                          | Baseline | Operon stagnation | Δ          |
|---------------------------------|---------:|------------------:|-----------:|
| Instances                       |       10 |                10 | —          |
| Instances with ≥1 rejection     |        0 |                 6 | **+6** |
| &nbsp;&nbsp;&nbsp;with completed retry | — |                 5 | — |
| &nbsp;&nbsp;&nbsp;with aborted retry (timeout) | — |                 1 | — |
| Per-instance rejection rate     |       0% |               60% | **+60 pp** |
| Total retry rounds              |        0 |                 6 | **+6** |
| Cumulative cost (USD)           | $  4.21 | $            6.58 | **+56%** (+$2.37) |
| Mean final patch length (chars) |     1839 |              1712 | — |
| Mean final history events       |     80.3 |              80.6 | — |

Per-round budget overhead: **$0.47** per completed retry round (= total cost delta / total_completed_retry_rounds; aborted retries excluded to avoid bias from undercounted spend).

**Raw artifact:** [`swebench_lite_delta.json`](./swebench_lite_delta.json). Reproduce via `scripts/generate_delta_artifact.py`.

## What the critic did

On the 10-instance shared slice, the default `finish_with_patch` critic accepted every first-attempt patch. `OperonStagnationCritic` rejected the first-attempt patch on 6 of 10 instances (60%), producing 6 total retry round(s): 5 completed, 1 aborted on timeout.

That's a **measurable behavioral delta** — the structural critic reaches different "done" decisions than an LLM-judged critic on the same agent trajectories. The cost is a **+56% budget overhead** for the slice. Whether retries *improve* correctness against the gold fix is not measured here; see caveats.

## Instance-level table

All numbers read directly from [`swebench_lite_delta.json`](./swebench_lite_delta.json) `per_instance[*]`.

| Instance | Baseline attempts | Treatment attempts | Baseline cost | Treatment cost |
|----------|------------------:|-------------------:|--------------:|---------------:|
| `astropy__astropy-12907` | 1 | 1 | $0.29 | $0.25 |
| `django__django-11001` | 1 | **2** | $0.40 | **$1.03** |
| `django__django-11019` | 1 | **1 → aborted 2** | $0.68 | **$0.66** |
| `django__django-11099` | 1 | **2** | $0.24 | **$0.65** |
| `django__django-11283` | 1 | **2** | $0.40 | **$0.66** |
| `django__django-11564` | 1 | 1 | $0.81 | $0.63 |
| `django__django-11815` | 1 | **2** | $0.43 | **$0.94** |
| `django__django-11848` | 1 | 1 | $0.33 | $0.37 |
| `django__django-11964` | 1 | 1 | $0.39 | $0.39 |
| `django__django-11999` | 1 | **2** | $0.24 | **$1.00** |

`1 → aborted 2` = critic rejected Attempt-1, Attempt-2 started but timed out before completion (per-instance `treatment_retry_aborted: true` in the JSON).

## Caveats (read these before citing)

1. **Sample bias.** n=10, 9 django + 1 astropy. Original plan called for n=30 alphabetic slice. The Docker-Desktop-on-Mac apt-signature build bug blocked all matplotlib/sympy/flask/pytest instances, leaving a django-heavy subset. Not representative of full SWE-bench-lite; generalizability limited. Run on a Linux host or OpenHands remote runtime for an unbiased slice.

2. **`resolved` not computed.** Both conditions produced non-empty `git_patch` outputs, but the SWE-bench patch-evaluation step (which applies patches and runs `FAIL_TO_PASS` / `PASS_TO_PASS` tests) was not run. The critic-retry delta is a behavioral signal; the underlying pass@1 comparison requires the evaluation pipeline.

3. **Certificate metadata not serialized.** `OperonStagnationCritic` emits `CriticResult.metadata` with `certificate_theorem="behavioral_stability_windowed"` on stagnation fires. That metadata field is **not captured** in the serialized event history (`critic_result` field on `MessageEvent` / `ActionEvent` is `null` throughout). Known openhands-sdk serialization gap, not a critic bug. Critic firing is inferred from Attempt-2 presence (for completed retries) or from the pinned aborted-retry list (for timed-out ones), not from on-disk certificate records. Fixing this requires either patching the SDK or adding a side-channel log from inside `OperonStagnationCritic.evaluate()`.

4. **Aborted retries (1).** Critic rejected Attempt-1 for `django__django-11019`, triggered Attempt-2 that hit the 60-min per-attempt ceiling and was killed. Counted as critic rejection(s) in the headline rate (`aborted_retries: 1`), but no Attempt-2 row exists in `output.jsonl` — their cumulative cost figures undercount what a clean retry would have produced. The per-retry cost figure excludes them from the denominator for this reason.

5. **Single model.** Paper 4's `threshold=0.2` epiplexic-integral cutoff was validated on different model families (sentence-MiniLM embeddings at n=300 trials). Behavior on the experiment model's trajectories may have different stagnation dynamics than the validation set.

## What this does and does not prove

**Does:**
- The harness runs end-to-end with `CodeActAgent` + `OperonStagnationCritic` on real SWE-bench instances.
- The structural critic exhibits a measurably different rejection pattern from the default LLM critic (60% of instances vs 0%).
- Per-retry-round overhead: **$0.47** per completed retry round (= total cost delta / total_completed_retry_rounds; aborted retries excluded to avoid bias from undercounted spend). Full-slice overhead: **+56%**.

**Does not:**
- Establish whether retries improve patch correctness — that needs the SWE-bench evaluation step.
- Generalize beyond the selection-biased instance subset (see caveat 1).
- Provide on-disk certificate evidence (openhands-sdk serialization gap; pending follow-up).

## Next steps

1. **Run SWE-bench evaluation** on both `output.jsonl` files to compute per-condition pass@1, resolved counts, `FAIL_TO_PASS` breakdown. This is the number the plan's success criteria calls for.
2. **Fix the certificate-metadata serialization** so downstream consumers can populate `certificate_emitted`, `certificate_theorem`, `cert_evidence_n` without relying on the retry-count proxy.
3. **Rerun on a Linux host or `--workspace remote`** to get an unbiased 30-instance slice.
4. **Broader scope:** once the harness is stable, repeat on Claude Sonnet 4.6 (original plan default) and on SWE-bench-Verified.
