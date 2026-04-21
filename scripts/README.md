# SWE-bench-lite delta harness

Two-condition experiment: OpenHands `CodeActAgent` with the default `AgentFinishedCritic` vs the same agent with `OperonStagnationCritic`. Fixed 30-instance slice (`scripts/instances.txt` — first 30 of SWE-bench-lite sorted lexicographically by `instance_id`).

## Prerequisites

- Python 3.12+
- Docker (local daemon, used by the benchmarks runner's `--workspace docker`)
- Anthropic API key (or OpenAI / other) exported as an environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) — the LLM config JSON does not carry the secret; LiteLLM reads it from env
- A separate venv for the experiment — the benchmarks repo pulls in `modal`, `commit0`, `swebench==4.1.0`, `docker`, etc. Don't pollute the dev venv.

## One-time setup

```bash
# From operon-openhands-gates repo root
python3.12 -m venv .venv-experiment
.venv-experiment/bin/pip install --upgrade pip
.venv-experiment/bin/pip install -e .

# openhands-benchmarks' pyproject ships a broken setuptools
# ``packages.find`` pattern (``include = ["benchmarks"]`` excludes
# subpackages). A wheel install gives you an empty ``benchmarks``
# package. Work around by cloning + editable install, which uses
# the on-disk layout directly.
mkdir -p .vendor
git clone https://github.com/OpenHands/benchmarks.git .vendor/benchmarks
git -C .vendor/benchmarks checkout 00182bf07968d2e9eb0a57f76ae58c9155e66f32
git -C .vendor/benchmarks submodule update --init --recursive --depth 1
.venv-experiment/bin/pip install -e .vendor/benchmarks

# Editable-install the vendored SDK subpackages to override the
# PyPI wheels. ``openhands.agent_server.docker.build`` rejects paths
# inside ``site-packages`` as "installed, not a source checkout" —
# the wheel-installed copies can't resolve the UV workspace root.
# Editable installs make ``__file__`` point into the vendor source
# tree, where the workspace-root climb succeeds.
for pkg in openhands-sdk openhands-tools openhands-workspace openhands-agent-server; do
  .venv-experiment/bin/pip install -e ".vendor/benchmarks/vendor/software-agent-sdk/$pkg"
done
# Commit SHA pinned to current main at 2026-04-19.
# Update deliberately — upstream may rename CRITIC_NAME_TO_CLASS or
# otherwise break the registration mechanism.
```

Copy `scripts/llm.json.example` → `scripts/llm.json` (gitignored). LiteLLM reads the API key from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.), so the config file itself carries no secrets — but the filename is on the ignore list anyway in case you add one:

```bash
cp scripts/llm.json.example scripts/llm.json
```

Then `export ANTHROPIC_API_KEY=...` (or source your existing env) before running.

## Smoke test (1 instance, ≤ $0.50)

```bash
.venv-experiment/bin/python scripts/run_swebench_lite.py \
  --condition operon_stagnation \
  --llm-config scripts/llm.json \
  --output-dir eval/runs/smoke \
  --select scripts/smoke_1.txt \
  --n-limit 1
```

Expected: one resolved (or unresolved — the point isn't pass@1, it's that the pipeline runs end-to-end). Verify:

1. Docker built the astropy image.
2. `eval/runs/smoke/.../output.critic_attempt_1.jsonl` exists and has one record.
3. If `operon_stagnation` was selected and the critic fired, `CriticResult.metadata` in the history should include `certificate_theorem: "behavioral_stability_windowed"`.

## Full runs (baseline + treatment, ~$20–80 total)

```bash
# Baseline (30 instances, ~3–5 hr wall-clock)
.venv-experiment/bin/python scripts/run_swebench_lite.py \
  --condition baseline \
  --llm-config scripts/llm.json \
  --output-dir eval/runs/baseline \
  --select scripts/instances.txt \
  --n-limit 30

# Treatment (30 instances, ~3–5 hr wall-clock)
.venv-experiment/bin/python scripts/run_swebench_lite.py \
  --condition operon_stagnation \
  --llm-config scripts/llm.json \
  --output-dir eval/runs/treatment \
  --select scripts/instances.txt \
  --n-limit 30
```

## Aggregate

Two aggregators, pick by whether you've run the SWE-bench patch-evaluation stage:

**`collect_results.py`** — for the full pass@1 comparison (requires `test_result.resolved` populated via a separate SWE-bench eval invocation that's not wired into this harness yet):

```bash
.venv-experiment/bin/python scripts/collect_results.py \
  --baseline eval/runs/baseline \
  --treatment eval/runs/treatment \
  --out eval/results/swebench_lite_delta.json
```

`summary` reports `pass_at_1`, `mean_turns`, `total_tokens` per condition. **If eval wasn't run, `pass_at_1` emits as `null` with an explanatory `pass_at_1_note`** — don't treat a missing eval step as a 0.0 pass rate.

**`generate_delta_artifact.py`** — for infer-only runs (what this repo produces today) where the behavioral delta you care about is the critic-retry pattern, not pass@1:

```bash
.venv-experiment/bin/python scripts/generate_delta_artifact.py \
  --baseline eval/runs/baseline \
  --treatment eval/runs/treatment \
  --aborted-treatment-retry django__django-11019 \
  --baseline-eval-report eval/runs/baseline/.../output.report.json \
  --treatment-eval-report eval/runs/treatment/.../output.dedup.report.json \
  --out-json eval/results/swebench_lite_delta.json \
  --out-md eval/results/swebench_lite_delta.md
```

Eval-report flags are optional; when supplied (produced by `benchmarks.swebench.eval_infer`; see the **Eval step** section below), the artifact adds `pass_at_1`, per-condition resolved/unresolved counts, per-instance `baseline_eval_status` + `treatment_eval_status`, and a retry-flip accounting line to the markdown. When omitted, output falls back to the infer-only schema.

Emits the markdown writeup alongside a JSON artifact whose `summary` block carries two orthogonal retry metrics:

- **Per-instance** (counts + a normalized rate; the `_rate` field is always in `[0, 1]`, the counts grow with `n_instances`):
  - `instances_with_rejection` / `instances_with_rejection_rate` — count / fraction of instances with at least one critic rejection (completed or aborted retry). The rate is the normalized headline number for prose.
  - `instances_with_completed_retry` — subset with `max_attempt > 1`.
  - `aborted_retries` / `aborted_retry_instance_ids` — instances pinned via `--aborted-treatment-retry`.
- **Per-round** (grows with retry depth — use for cost denominators):
  - `total_retry_rounds` — `total_completed_retry_rounds + aborted_retries`.
  - `total_completed_retry_rounds` — `Σ (max_attempt - 1)` across instances. Denominator for `$/retry-round` overhead.

Per-instance `per_instance[*]` rows carry `baseline_max_attempt`, `treatment_max_attempt`, `treatment_retry_aborted` (bool), cumulative costs, and final patch lengths.

`--aborted-treatment-retry` is repeatable and marks instances whose treatment Attempt-2 started but didn't complete (e.g. timeout). Unknown IDs or IDs already having a completed retry row raise a validation error (they would otherwise double-count).

### Certificate evidence (populates `certificates_emitted`, v0.1.0a3+)

`OperonStagnationCritic.evaluate()` emits a `[CERT-FIRE] {json}` stdout log line on the cert-transition turn. The benchmarks runner captures the container's stdout into `logs/instance_<iid>.output.log` (JSON-wrapped under `[DOCKER]`). `generate_delta_artifact.py --baseline-logs-dir` / `--treatment-logs-dir` parse those logs and correlate to instances by filename:

```bash
.venv-experiment/bin/python scripts/generate_delta_artifact.py \
  ... \
  --baseline-logs-dir eval/runs/baseline/.../logs \
  --treatment-logs-dir eval/runs/treatment/.../logs \
  --out-json ... --out-md ...
```

When cert records are found: `summary.certificates_emitted` + `per_instance[*].treatment_certificate` carry the on-disk payload (theorem, source, `cert_evidence_n`, `epiplexic_integral`, `severity`, `detection_index`). Markdown gets a `Certificates emitted` headline row, a `Treat cert` column in the instance table, and the "cert metadata not serialized" caveat flips to a "side-channel log" variant that links back to the library emission.

The logs-dir flags are independently optional and pre-instrumentation logs (v0.1.0a2 and earlier) parse to zero cert records — backward compatible.

### Eval step (populates `pass_at_1`)

SWE-bench's inference pipeline produces patches; the patch-evaluation step runs them against the instance tests. `benchmarks/swebench/eval_infer.py` is the entry point:

```bash
# Baseline (10 unique instances, use as-is):
BASE=eval/runs/baseline/princeton-nlp__SWE-bench_Lite-test/openai/gpt-5_sdk_3e0a3a0_maxiter_500
.venv-experiment/bin/python -m benchmarks.swebench.eval_infer \
  "$BASE/output.jsonl" \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --run-id operon-baseline-n10 \
  --no-modal --timeout 3600 --workers 4

# Treatment (retries produce duplicate rows; dedupe first):
TREAT=eval/runs/treatment/princeton-nlp__SWE-bench_Lite-test/openai/gpt-5_sdk_3e0a3a0_maxiter_500
.venv-experiment/bin/python scripts/dedupe_for_eval.py "$TREAT/output.jsonl" "$TREAT/output.dedup.jsonl"
.venv-experiment/bin/python -m benchmarks.swebench.eval_infer \
  "$TREAT/output.dedup.jsonl" \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --run-id operon-treatment-n10 \
  --no-modal --timeout 3600 --workers 4
```

Each run writes `$INPUT.report.json` next to the input with the harness's resolved/unresolved id lists. Feed both reports back into `generate_delta_artifact.py` via `--baseline-eval-report` / `--treatment-eval-report` to get pass@1 + resolved flags in the artifact.

`scripts/dedupe_for_eval.py` exists because `eval_infer.py` doesn't dedupe rows with the same `instance_id`; under the iterative-refinement loop, the treatment `output.jsonl` carries one row per (instance_id, attempt) pair, and we only want the final critic-accepted patch evaluated.

### Note on certificate fields

`collect_results.py`'s `certificates_emitted` rollup relies on `_scan_certificate` walking the serialized event history for `certificate_theorem`. **The openhands-sdk currently does NOT serialize `CriticResult.metadata` into `MessageEvent` / `ActionEvent` records** — the field comes back as `null`. Until that gap is closed (SDK patch or side-channel log from `OperonStagnationCritic.evaluate()`), the rollup will report 0 for every real run even when the critic fired. `generate_delta_artifact.py` sidesteps this by inferring critic firings from Attempt-2 presence + the `--aborted-treatment-retry` list. See `eval/results/swebench_lite_delta.md` caveat 3.

## Overrides

- `--llm-model <name>` — pass through to the benchmark runner to swap models at run time (Claude Haiku, GPT-4o, etc.).
- `--workspace remote` — switch to OpenHands' remote runtime API (requires `RUNTIME_API_KEY`). Use if Docker-on-Mac flakes on specific repos.
- `--num-workers <N>` — parallelize instance execution; start at 4 to avoid Anthropic rate-limit throttling, scale if no 429s.

## Reproduction expectations

The instance slice is deterministic. The model config is in git. The benchmarks commit SHA is pinned. Given the same LLM API endpoint, two runs should produce the same distribution of turns and pass@1 — but **individual instances may flap** due to model non-determinism. Report aggregate metrics, not per-instance.
