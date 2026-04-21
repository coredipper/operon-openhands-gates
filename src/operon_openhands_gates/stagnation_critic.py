"""StagnationCritic — Bayesian stagnation detection for OpenHands iterative refinement.

Backed by Operon Paper 4 §4.3 "Epiplexity: Embedding Quality Determines
Outcome". With real sentence embeddings (all-MiniLM-L6-v2, N = 3 seeds
x 100 trials = 300), the biological two-signal monitor achieves:

- convergence discrimination: 0.960 accuracy (vs 0.401 naive baseline)
- false-stagnation rejection: 0.960 accuracy, 0.000 FP rate
  (vs 0.020 accuracy, 0.980 FP rate for the naive baseline)

Authoritative source for these numbers (path relative to the Operon
repo root): ``eval/results/benchmarks_real_embeddings/multi_model_summary.json``
at ``github.com/coredipper/operon``.

This module plugs into OpenHands' iterative-refinement loop through the
``CriticBase`` seam. When ``IterativeRefinementConfig`` is attached, the
conversation retries while the critic's score is below
``success_threshold``. An LLM-judged critic decides by opinion; this
critic decides by structural signal: the epiplexic integral over a
sliding window of agent messages. Low integral sustained over
``critical_duration`` measurements triggers a ``behavioral_stability``
certificate, emitted once per critic instance on the transition to the
stagnant state.

Sibling package ``operon-langgraph-gates`` ships the equivalent primitive
for LangGraph. Both wrap the same ``EpiplexityMonitor`` substrate and
emit the same ``behavioral_stability`` certificate. This is the
framework-portability claim in code.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from openhands.sdk.critic.base import CriticBase
from openhands.sdk.critic.result import CriticResult
from operon_ai.core.certificate import Certificate
from operon_ai.health.epiplexity import EpiplexityMonitor
from pydantic import ConfigDict, Field, PrivateAttr

from .embedders import NGramEmbedder

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openhands.sdk.event.base import LLMConvertibleEvent


class OperonStagnationCritic(CriticBase):
    """Structural stagnation critic for OpenHands iterative refinement.

    Replaces an LLM-judged success score with ``EpiplexityMonitor``'s
    Bayesian convergence signal. Returns ``score = epiplexic_integral``,
    which is in [0, 1] where low values indicate stagnation — aligned with
    OpenHands' ``CriticResult.score`` semantic of "predicted probability
    of success". A low integral means the agent is going in circles,
    which is low success.

    Pair with ``IterativeRefinementConfig(success_threshold=0.2)`` to make
    the refinement loop terminate on sustained stagnation. OpenHands'
    default ``success_threshold=0.6`` is tuned for LLM probability-of-
    success scores and is too high for the epiplexic integral (which
    Paper 4 treats as stagnant below 0.2).

    One instance per conversation. ``CriticBase.evaluate`` is not handed
    the conversation object, so per-conversation state must live on the
    critic itself. Reusing a single critic instance across conversations
    is unsupported and will leak state.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Epiplexic-integral threshold below which a measurement counts "
            "as stagnant. Paper 4 uses 0.2."
        ),
    )
    window: int = Field(
        default=10,
        ge=2,
        description="Window size for the epiplexic integral (mean of recent measures).",
    )
    critical_duration: int = Field(
        default=3,
        ge=1,
        description=(
            "Consecutive below-threshold measurements required before the "
            "critic flips to stagnant and emits a certificate."
        ),
    )
    embedder: Any = Field(
        default=None,
        description=(
            "Optional embedding provider (duck-typed; must expose "
            "``embed(text: str) -> list[float]``). Defaults to the zero-dep "
            "``NGramEmbedder`` — sufficient for verbatim-repeat detection; "
            "swap in a neural embedder for robustness to paraphrase."
        ),
    )

    _monitor: EpiplexityMonitor | None = PrivateAttr(default=None)
    _severities: list[float] = PrivateAttr(default_factory=list)
    _integrals: list[float] = PrivateAttr(default_factory=list)
    _low_integral_streak: int = PrivateAttr(default=0)
    _is_stagnant: bool = PrivateAttr(default=False)
    _certificate: Certificate | None = PrivateAttr(default=None)

    def _get_monitor(self) -> EpiplexityMonitor:
        if self._monitor is None:
            self._monitor = EpiplexityMonitor(
                embedding_provider=self.embedder or NGramEmbedder(),
                window_size=self.window,
                threshold=self.threshold,
            )
        return self._monitor

    def evaluate(
        self,
        events: Sequence[LLMConvertibleEvent],
        git_patch: str | None = None,
    ) -> CriticResult:
        text = _extract_last_agent_text(events)
        result = self._get_monitor().measure(text)

        # Severity for the certificate payload: low epiplexity = high severity.
        # (Mirrors the ``behavioral_stability`` verify semantics used by
        # operon-langgraph-gates and the Operon core verifier.)
        severity = max(0.0, min(1.0, 1.0 - float(result.epiplexity)))
        self._severities.append(severity)

        integral = float(result.epiplexic_integral)
        self._integrals.append(integral)

        # Detect on sustained low integral — the monitor's built-in status
        # classifier relies on a perplexity approximation that varies with
        # text shape and can mask stagnation. The integral is the stable
        # signal. (Choice validated empirically in operon-langgraph-gates.)
        if integral < self.threshold:
            self._low_integral_streak += 1
        else:
            self._low_integral_streak = 0

        was_stagnant = self._is_stagnant
        should_be_stagnant = self._low_integral_streak >= self.critical_duration

        if should_be_stagnant and not was_stagnant:
            # Evidence = the exact aggregates detection operates on.
            # Stagnation fires when each of the last ``critical_duration``
            # rolling-window integrals is below the detection threshold.
            # Flattening those overlapping windows into a single mean loses
            # the per-window predicate: under ``window=2, cd=2`` with
            # severities ``[0.61, 1.0, 0.61]``, both window means are
            # ``0.805`` (detection fires against stability threshold
            # ``0.8``), but the flattened mean is only ``0.74`` (replay
            # would say stability held). Store the per-window severity
            # means themselves — one per violating window — and verify
            # with a max-based check (``max(window_means) < threshold``)
            # that mirrors detection's "every window violates" semantic.
            violating_integrals = self._integrals[-self.critical_duration :]
            window_severity_means = tuple(1.0 - i for i in violating_integrals)
            # Threshold-domain translation: detection fires on
            # ``integral < self.threshold`` ⟺ ``window_mean_severity > 1 - self.threshold``.
            # Store the stability threshold ``1 - self.threshold`` so the
            # verifier's ``max < stored_threshold`` matches detection at
            # every threshold value in [0, 1], not just <= 0.5.
            #
            # Emit the certificate *before* flipping ``_is_stagnant`` so a
            # failure (e.g. unregistered theorem) leaves the critic in its
            # prior non-stagnant state and a later ``evaluate()`` call can
            # retry emission once the underlying problem is corrected. If
            # we flipped state first, the ``was_stagnant/not was_stagnant``
            # guard below would suppress retry and the critic would be
            # permanently stuck in ``is_stagnant=True / certificate=None``.
            self._certificate = _emit_certificate(
                window_severity_means=window_severity_means,
                threshold=1.0 - self.threshold,
                detection_index=len(self._severities),
            )
            # Side-channel log so downstream consumers can find on-disk
            # evidence of the certificate emission. The openhands-sdk
            # currently does NOT serialize ``CriticResult.metadata``
            # into ``MessageEvent`` / ``ActionEvent`` records, so the
            # retry-count proxy was the only available signal before
            # this. With the line below, the critic's stdout carries a
            # stable ``[CERT-FIRE] {json}`` marker that the benchmarks
            # runner captures into ``logs/instance_<iid>.output.log``;
            # ``scripts/generate_delta_artifact.py`` parses those logs
            # to populate ``certificate_emitted`` / ``certificate_theorem``
            # / ``cert_evidence_n`` in the delta artifact.
            #
            # Fires exactly once per critic instance per conversation
            # (same transition guard as the cert emission itself), so
            # the log doesn't accumulate duplicates under sustained
            # stagnation.
            logger.info(
                "[CERT-FIRE] %s",
                json.dumps(
                    {
                        "theorem": self._certificate.theorem,
                        "source": self._certificate.source,
                        "cert_evidence_n": len(window_severity_means),
                        "epiplexic_integral": integral,
                        "severity": severity,
                        "detection_index": len(self._severities),
                    },
                    separators=(",", ":"),
                ),
            )
        self._is_stagnant = should_be_stagnant

        status = "STAGNANT" if self._is_stagnant else "healthy"
        metadata: dict[str, Any] = {"severity": severity}
        if self._certificate is not None:
            metadata["certificate_theorem"] = self._certificate.theorem
            metadata["certificate_source"] = self._certificate.source
            # Carry the evidence-window length alongside the theorem/source
            # so downstream artifact collectors (e.g.
            # operon-openhands-gates/scripts/collect_results.py) don't have
            # to reconstruct it from the certificate's parameters. Equals
            # ``self.critical_duration`` by construction (the cert is built
            # from exactly that many violating rolling-window means).
            signal_values = self._certificate.parameters.get("signal_values", ())
            metadata["cert_evidence_n"] = len(signal_values)
        return CriticResult(
            score=integral,
            message=(
                f"epiplexic_integral={integral:.3f} threshold={self.threshold} "
                f"streak={self._low_integral_streak} {status}"
            ),
            metadata=metadata,
        )

    @property
    def certificate(self) -> Certificate | None:
        """The ``behavioral_stability`` certificate, or None if never fired."""
        return self._certificate

    @property
    def is_stagnant(self) -> bool:
        return self._is_stagnant


def _emit_certificate(
    *,
    window_severity_means: tuple[float, ...],
    threshold: float,
    detection_index: int,
) -> Certificate:
    """Emit a ``behavioral_stability`` certificate backed by per-window means.

    ``signal_values`` is the sequence of mean severities over each violating
    rolling window — one value per window, ``critical_duration`` values
    total. The replay check is ``max(signal_values) < threshold``, which
    mirrors detection's "every one of the last ``critical_duration``
    rolling windows was violating" predicate.

    ``detection_index`` is the total number of evaluations the critic had
    performed when stagnation was detected, used only in the human-readable
    conclusion text. Pass ``len(self._severities)`` at emit time — after a
    long healthy prefix, this is much larger than the evidence-slice length
    and keeps the conclusion accurate.

    ``window_severity_means`` must be non-empty. A stagnation certificate
    without evidence is a contradiction in terms, so emitting one is a
    programmer error rather than a runtime condition to tolerate.
    """
    if not window_severity_means:
        raise ValueError(
            "window_severity_means must be non-empty; "
            "a stagnation certificate requires at least one violating window"
        )
    return Certificate.from_theorem(
        theorem=_WINDOWED_THEOREM,
        parameters=MappingProxyType(
            {
                "signal_values": tuple(window_severity_means),
                "threshold": float(threshold),
            }
        ),
        conclusion=(
            f"Stagnation detected after {detection_index} measurements; "
            f"{len(window_severity_means)} violating rolling windows "
            f"captured for replay verification."
        ),
        source="operon_openhands_gates.stagnation_critic",
    )


def _extract_last_agent_text(events: Sequence[LLMConvertibleEvent]) -> str:
    """Extract text from the most recent agent-sourced event.

    Walks events in reverse. Prefers ``MessageEvent`` from the agent;
    falls back to any textual content on recent events. Returns the
    empty string if nothing extractable is found — the monitor tolerates
    that and simply records a zero-novelty sample.
    """
    # Imported lazily so the module imports cleanly even if the SDK's
    # event subpackage shape changes across minor versions.
    from openhands.sdk.event.llm_convertible.message import MessageEvent
    from openhands.sdk.llm import content_to_str

    for event in reversed(events):
        if isinstance(event, MessageEvent) and _is_agent(event):
            try:
                parts = content_to_str(event.llm_message.content)
            except Exception:
                parts = []
            # ``content_to_str`` in ``openhands-sdk==1.17`` returns ``list[str]``
            # (see openhands-sdk/openhands/sdk/llm/message.py:697). Branch
            # defensively in case a future SDK version returns a plain string
            # — iterating a string would join characters with spaces and
            # corrupt the monitor's input.
            if isinstance(parts, str):
                joined = parts.strip()
            else:
                joined = " ".join(p for p in parts if p).strip()
            if joined:
                return joined
    return ""


def _is_agent(event: Any) -> bool:
    # ``MessageEvent.source`` is a ``SourceType`` enum-like — compare by string
    # to stay tolerant across SDK versions that may change the literal.
    return str(getattr(event, "source", "")).lower().endswith("agent")


# A theorem name distinct from the shared ``behavioral_stability`` (which is
# flat-mean-based). Upstream ``operon_ai.core.certificate`` registers this
# theorem in its theorem registry, so ``Certificate.from_theorem`` resolves
# the correct verifier without this package binding to any upstream symbol
# beyond the public ``Certificate`` class itself (operon-ai>=0.36.1).
_WINDOWED_THEOREM = "behavioral_stability_windowed"
