# SWE-bench-lite delta: baseline vs OperonStagnationCritic

**Scope:** n=10, OpenAI GPT-5, SWE-bench-lite `test` split, local Docker workspace on Apple Silicon.
**Conditions:**
- `baseline`: OpenHands `CodeActAgent` + default `AgentFinishedCritic` (finish_with_patch preset).
- `operon_stagnation`: same agent + `OperonStagnationCritic(threshold=0.2, window=10, critical_duration=3)`.

## Headline numbers

| Metric                    | Baseline | Operon stagnation | Δ        |
|---------------------------|---------:|------------------:|---------:|
| Instances                 |       10 |                10 |        — |
| Critic-triggered retries  |        0 |                 5 |   **+5** |
| Max-attempt = 2 rate      |       0% |              50%  | **+50%** |
| Cumulative cost (USD)     |    $4.21 |             $6.58 |   **+56%** |
| Mean final patch length   |   1,839  |            1,712  |   −7%    |
| Mean history events       |     80.3 |              80.6 |     ~0%  |

**Raw artifact:** [`swebench_lite_delta.json`](./swebench_lite_delta.json).

## What the critic did

On the 10-instance shared slice, the default `finish_with_patch` critic accepted every first-attempt patch. `OperonStagnationCritic` rejected 5 of 10 first-attempt patches and triggered a second refinement iteration. That's a **measurable behavioral delta** — the structural critic reaches different "done" decisions than an LLM-judged critic on the same agent trajectories.

The cost is a **56% budget overhead** for the slice (~$0.20/instance retried, twice the tokens of a no-retry instance). Whether the retry *improves* the patch against the gold fix is not measured here; see caveats.

## Instance-level table

| Instance                     | Baseline attempts | Treatment attempts | Baseline cost | Treatment cost |
|------------------------------|------------------:|-------------------:|--------------:|---------------:|
| `astropy__astropy-12907`     | 1                 | 1                  | $0.28         | $0.25          |
| `django__django-11001`       | 1                 | **2**              | $0.44         | **$1.03**      |
| `django__django-11019`       | 1                 | 1 *(see caveat)*   | $0.37         | $0.66          |
| `django__django-11099`       | 1                 | **2**              | $0.25         | **$0.65**      |
| `django__django-11283`       | 1                 | **2**              | $0.29         | **$0.65**      |
| `django__django-11564`       | 1                 | 1                  | $0.60         | $0.63          |
| `django__django-11815`       | 1                 | **2**              | $0.41         | **$0.94**      |
| `django__django-11848`       | 1                 | 1                  | $0.41         | $0.37          |
| `django__django-11964`       | 1                 | 1                  | $0.28         | $0.39          |
| `django__django-11999`       | 1                 | **2**              | $0.88         | **$1.00**      |

## Caveats (read these before citing)

1. **Sample bias.** n=10, 9 django + 1 astropy. Original plan called for n=30 alphabetic slice. The Docker-Desktop-on-Mac apt-signature build bug blocked all matplotlib/sympy/flask/pytest instances, leaving a django-heavy subset. Not representative of full SWE-bench-lite; generalizability limited. Run on a Linux host or OpenHands remote runtime for an unbiased slice.

2. **`resolved` not computed.** Both conditions produced non-empty `git_patch` outputs, but the SWE-bench patch-evaluation step (which applies patches and runs `FAIL_TO_PASS` / `PASS_TO_PASS` tests) was not run. The critic-retry delta is a behavioral signal; the underlying pass@1 comparison requires the evaluation pipeline.

3. **Certificate metadata not serialized.** `OperonStagnationCritic` emits `CriticResult.metadata` with `certificate_theorem="behavioral_stability_windowed"` on stagnation fires. That metadata field is **not captured** in the serialized event history (`critic_result` field on `MessageEvent` / `ActionEvent` is `null` throughout). This is an openhands-sdk serialization gap, not a critic bug. Critic firing is inferred from Attempt-2 presence, not from on-disk certificate records. Fixing this requires either patching the SDK or adding a side-channel log from inside `OperonStagnationCritic.evaluate()`.

4. **`django__django-11019` Attempt-2 timeout.** This instance's critic rejected Attempt-1 and triggered retry. The retry ran 38+ minutes and hit the per-attempt 60-min ceiling. The run was killed to avoid a multi-hour retry cascade. Treatment's cumulative cost for this instance is therefore a lower bound; the true cost for a clean retry path would be higher.

5. **Single model (GPT-5).** Paper 4's `threshold=0.2` epiplexic-integral cutoff was validated on different model families (sentence-MiniLM embeddings at n=300 trials). Behavior on GPT-5 trajectories may have different stagnation dynamics than the validation set.

## What this does and does not prove

**Does:**
- The harness runs end-to-end with `CodeActAgent` + `OperonStagnationCritic` on real SWE-bench instances.
- The structural critic exhibits a measurably different retry pattern from the default LLM critic (50% retry rate vs 0%).
- Retries cost ~2× per-instance and roll up to a 56% overall overhead at this slice.

**Does not:**
- Establish whether retries improve patch correctness — that needs the SWE-bench evaluation step.
- Generalize beyond django-heavy instances on GPT-5.
- Provide on-disk certificate evidence (openhands-sdk serialization gap; pending follow-up).

## Next steps

1. **Run SWE-bench evaluation** on both `output.jsonl` files to compute per-condition pass@1, resolved counts, `FAIL_TO_PASS` breakdown. This is the number the plan's success criteria calls for.
2. **Fix the certificate-metadata serialization** so `scripts/collect_results.py`'s `_scan_certificate` can populate `certificate_emitted`, `certificate_theorem`, `cert_evidence_n` without relying on the retry-count proxy.
3. **Rerun on a Linux host or `--workspace remote`** to get an unbiased 30-instance slice.
4. **Broader scope:** once the harness is stable, repeat on Claude Sonnet 4.6 (original plan default) and on SWE-bench-Verified.
