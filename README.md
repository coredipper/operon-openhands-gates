# operon-openhands-gates

> **In-loop** structural reliability critics for the [OpenHands Agent SDK](https://github.com/OpenHands/software-agent-sdk) — drop-in, cert-emitting.

OpenHands' own docs flag an architectural gap in iterative refinement:

> *"the current implementation relies solely on threshold/iteration limits rather than monitoring improvement velocity or convergence rates — suggesting this is an architectural gap where monitoring logic could plug in."*
> — https://docs.openhands.dev/sdk/guides/iterative-refinement

This package ships the missing monitor as a `CriticBase` subclass. It replaces an LLM-judged success score with a **Bayesian stagnation signal** computed over the conversation's message history. When the agent goes in circles, the critic's score drops below threshold, iterative refinement terminates, and a replayable `behavioral_stability` certificate is emitted.

**At a glance:**

- `OperonStagnationCritic` — `epiplexic_integral`-based detection (Paper 4 §4.3, 0.960 convergence accuracy with real embeddings) that plugs directly into `Agent(critic=...)`.
- One certificate per detection transition, self-verifiable via `certificate.verify()`.
- Zero-dep `NGramEmbedder` default — bring your own neural embedder for paraphrase-robust detection.

## Install

```bash
pip install operon-openhands-gates
```

Requires `operon-ai>=0.34.4` and `openhands-sdk>=1.15`.

## Quickstart

```python
from openhands.sdk import Agent, Conversation, LLM
from openhands.sdk.critic.base import IterativeRefinementConfig
from operon_openhands_gates import OperonStagnationCritic

critic = OperonStagnationCritic(
    threshold=0.2,
    critical_duration=3,
    iterative_refinement=IterativeRefinementConfig(
        success_threshold=0.2,  # match the critic's threshold
        max_iterations=5,
    ),
)

agent = Agent(llm=LLM(model="anthropic/claude-sonnet-4-5"), tools=[...], critic=critic)
conversation = Conversation(agent=agent, workspace=workspace)
conversation.send_message("Fix the failing test in ...")
conversation.run()  # iterative refinement terminates on sustained stagnation

if critic.certificate is not None:
    # Replayable evidence of what the gate saw.
    verification = critic.certificate.verify()
    assert verification.holds
```

### Why the non-default `success_threshold`

OpenHands' default `success_threshold=0.6` is tuned for LLM probability-of-success scores. `OperonStagnationCritic` returns the `epiplexic_integral` directly — in [0, 1] where low = stagnant. Paper 4 §4.3 uses δ=0.2 as the stagnation threshold, so match it on the refinement config.

## Sibling package

- [`operon-langgraph-gates`](https://github.com/coredipper/operon-langgraph-gates) — same Paper 4 substrate, same `behavioral_stability_windowed` certificate, targeted at LangGraph's `StateGraph` with `.wrap()` / `.edge()` node APIs. Two packages, one core — this is the framework-portability claim from Paper 5 §3 in code.

## Certificate theorem name and verification

Certificates emitted by this package carry the theorem name `behavioral_stability_windowed` (not the core's shared `behavioral_stability`). The two differ in how they verify:

- `behavioral_stability` (shared core): `mean(severities) < threshold`. Loses the per-window structure that rolling-integral detection operates on.
- `behavioral_stability_windowed` (this package): `max(per_window_severity_means) <= stability_threshold`. Mirrors detection exactly.

The windowed verifier is registered against `operon_ai.core.certificate`'s `_VERIFY_REGISTRY` at package import time. **In-process verification is transparent**: `certificate.verify()` resolves to the correct verifier as long as this package has been imported.

**Cross-process limitation**: deserializing a `behavioral_stability_windowed` cert in a process that has `operon_ai` installed but has NOT imported `operon_openhands_gates` will fail to resolve the verifier. The canonical fix is upstreaming `_verify_window_max_stability` into `operon_ai.core.certificate` as a registered theorem path; tracked as a follow-up. In the meantime, any process that consumes these certs must import `operon_openhands_gates` (which is already a runtime requirement for producing them).

### Breaking change from pre-alpha prototypes

Earlier pre-release builds emitted certificates with theorem name `behavioral_stability` (the shared core name), bound to a locally-attached `_verify_fn`. That shape was semantically wrong — the shared verifier is flat-mean-based, so any cert round-tripped through serialization would silently revert to the wrong replay logic. Consumers that key on `certificate.theorem == "behavioral_stability"` or `metadata["certificate_theorem"] == "behavioral_stability"` must update to `"behavioral_stability_windowed"`. No migration path is provided; alpha.

## Citations

Backed by [Paper 4 §4.3](https://github.com/coredipper/operon/blob/main/article/paper4/main.pdf): convergence/false-stagnation accuracy **0.960** with real sentence embeddings (all-MiniLM-L6-v2, N = 300 trials). Full numbers and reproduction commands in the Operon repo at `eval/results/benchmarks_real_embeddings/multi_model_summary.json`. [Paper 5 §3](https://github.com/coredipper/operon/blob/main/article/paper5/main.pdf) establishes the preservation-under-compilation framework that the certificate follows.

## Status

**Alpha.** API may change before `0.1.0` stable. Feedback welcome via Issues.

## License

MIT — see [LICENSE](./LICENSE).
