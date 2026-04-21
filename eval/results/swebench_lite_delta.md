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
| &nbsp;&nbsp;&nbsp;with completed retry | — |                 6 | — |
| &nbsp;&nbsp;&nbsp;with aborted retry (timeout) | — |                 0 | — |
| Per-instance rejection rate     |       0% |               60% | **+60 pp** |
| Total retry rounds              |        0 |                12 | **+12** |
| Cumulative cost (USD)           | $  4.21 | $            9.94 | **+136%** (+$5.73) |
| Mean final patch length (chars) |     1839 |              1466 | — |
| Mean final history events       |     80.3 |              75.2 | — |
| **pass@1**                      |      60% |               70% | **+10 pp** |
| &nbsp;&nbsp;&nbsp;resolved       |        6 |                 7 | — |
| &nbsp;&nbsp;&nbsp;unresolved     |        4 |                 3 | — |
| Certificates emitted            |        0 |                 0 | **+0** |

Per-round budget overhead: **$0.48** per completed retry round (= total cost delta / total_completed_retry_rounds; aborted retries excluded to avoid bias from undercounted spend).

Retry-flip accounting (treatment retried 6 instance(s)): **1** improved (unresolved → resolved), **0** broke (resolved → unresolved), 5 unchanged.

**Raw artifact:** [`swebench_lite_delta.json`](./swebench_lite_delta.json). Reproduce via `scripts/generate_delta_artifact.py`.

## What the critic did

On the 10-instance shared slice, the default `finish_with_patch` critic accepted every first-attempt patch. `OperonStagnationCritic` rejected the first-attempt patch on 6 of 10 instances (60%), producing 12 total retry round(s): 12 completed, 0 aborted on timeout.

That's a **measurable behavioral delta** — the structural critic reaches different "done" decisions than an LLM-judged critic on the same agent trajectories. The cost is a **+136% budget overhead** for the slice.

## Instance-level table

All numbers read directly from [`swebench_lite_delta.json`](./swebench_lite_delta.json) `per_instance[*]`.

| Instance | Base attempts | Treat attempts | Base cost | Treat cost | Base resolved | Treat resolved | Treat cert |
|----------|--------------:|---------------:|----------:|-----------:|:-------------:|:--------------:|:----------:|
| `astropy__astropy-12907` | 1 | 1 | $0.29 | $0.20 | ✅ | ✅ | · |
| `django__django-11001` | 1 | 1 | $0.40 | $0.52 | ✅ | ✅ | · |
| `django__django-11019` | 1 | **3** | $0.68 | **$3.05** | ❌ | ❌ | · |
| `django__django-11099` | 1 | **3** | $0.24 | **$0.66** | ✅ | ✅ | · |
| `django__django-11283` | 1 | **3** | $0.40 | **$1.29** | ❌ | ❌ | · |
| `django__django-11564` | 1 | 1 | $0.81 | $0.66 | ❌ | ❌ | · |
| `django__django-11815` | 1 | **3** | $0.43 | **$1.12** | ❌ | ✅ | · |
| `django__django-11848` | 1 | **3** | $0.33 | **$1.15** | ✅ | ✅ | · |
| `django__django-11964` | 1 | 1 | $0.39 | $0.23 | ✅ | ✅ | · |
| `django__django-11999` | 1 | **3** | $0.24 | **$1.06** | ✅ | ✅ | · |

`1 → aborted 2` = critic rejected Attempt-1, Attempt-2 started but timed out before completion (per-instance `treatment_retry_aborted: true` in the JSON).

## Caveats (read these before citing)

1. **Sample bias.** n=10, 9 django + 1 astropy. Original plan called for n=30 alphabetic slice. The Docker-Desktop-on-Mac apt-signature build bug blocked all matplotlib/sympy/flask/pytest instances, leaving a django-heavy subset. Not representative of full SWE-bench-lite; generalizability limited. Run on a Linux host or OpenHands remote runtime for an unbiased slice.

2. **Certificate evidence via side-channel log.** ``OperonStagnationCritic`` emits a ``[CERT-FIRE]`` line on its stdout when a certificate fires (v0.1.0a3+). The benchmarks runner captures those lines into ``logs/instance_<iid>.output.log``; ``scripts/generate_delta_artifact.py --*-logs-dir`` parses them to populate ``treatment_certificate`` per instance + ``certificates_emitted`` summary. **0 of 10 treatment instances have on-disk cert records** on this run. The openhands-sdk still doesn't serialize ``CriticResult.metadata`` into ``MessageEvent`` / ``ActionEvent`` history; the stdout log is the deliberate workaround.

3. **Single model.** Paper 4's `threshold=0.2` epiplexic-integral cutoff was validated on different model families (sentence-MiniLM embeddings at n=300 trials). Behavior on the experiment model's trajectories may have different stagnation dynamics than the validation set.

## What this does and does not prove

**Does:**
- The harness runs end-to-end with `CodeActAgent` + `OperonStagnationCritic` on real SWE-bench instances, from inference through patch-evaluation.
- The structural critic exhibits a measurably different rejection pattern from the default LLM critic (60% of instances vs 0%).
- Per-retry-round overhead: **$0.48** per completed retry round (= total cost delta / total_completed_retry_rounds; aborted retries excluded to avoid bias from undercounted spend). Full-slice overhead: **+136%**.
- Surface 0 on-disk certificate record(s) via the ``[CERT-FIRE]`` stdout log parsed from ``logs/instance_<iid>.output.log`` (see caveat 2).

**Does not:**
- Generalize beyond the selection-biased instance subset (see caveat 1).

## Next steps

1. **Rerun on a Linux host or `--workspace remote`** to get an unbiased 30-instance slice.
2. **Broader scope:** once the harness is stable, repeat on Claude Sonnet 4.6 (original plan default) and on SWE-bench-Verified.
