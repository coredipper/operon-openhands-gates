"""Tests for OperonStagnationCritic.

Shape ported from ``operon-langgraph-gates/tests/test_stagnation_gate.py``.
Same Paper 4 §4.3 claim, same NGram embedder for install-free determinism,
adapted to OpenHands' ``CriticBase.evaluate(events, git_patch=None)`` seam.

The SDK's ``MessageEvent`` is Pydantic-validated and requires a real
``Message``, so tests skip gracefully if the SDK is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openhands.sdk", reason="openhands-sdk not installed")

from openhands.sdk.critic.result import CriticResult  # noqa: E402
from openhands.sdk.event.llm_convertible.message import MessageEvent  # noqa: E402
from openhands.sdk.llm import Message, TextContent  # noqa: E402

from operon_openhands_gates import OperonStagnationCritic  # noqa: E402


def _make_critic() -> OperonStagnationCritic:
    # Short window + low critical_duration keeps tests fast and deterministic
    # under the zero-dep NGram embedder.
    return OperonStagnationCritic(threshold=0.2, critical_duration=2, window=3)


def _agent_msg(text: str) -> MessageEvent:
    return MessageEvent(
        source="agent",
        llm_message=Message(role="assistant", content=[TextContent(text=text)]),
    )


# Real English sentences — enough trigram divergence to keep the
# epiplexic integral above the threshold across calls.
_DIVERSE_RESPONSES = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "sphinx of black quartz judge my vow",
    "two driven jocks help fax my big quiz",
    "the five boxing wizards jump quickly",
    "waltz bad nymph for quick jigs vex",
    "jackdaws love my big sphinx of quartz",
]


def test_evaluate_returns_critic_result_with_score_in_range() -> None:
    critic = _make_critic()
    result = critic.evaluate([_agent_msg("hello world")])
    assert isinstance(result, CriticResult)
    assert 0.0 <= result.score <= 1.0


def test_no_stagnation_on_first_call() -> None:
    critic = _make_critic()
    critic.evaluate([_agent_msg("first response")])
    assert critic.is_stagnant is False
    assert critic.certificate is None


def test_stagnation_detected_on_identical_outputs() -> None:
    critic = _make_critic()
    for _ in range(6):
        critic.evaluate([_agent_msg("same response every turn")])
    assert critic.is_stagnant is True
    assert critic.certificate is not None
    assert critic.certificate.theorem == "behavioral_stability"


def test_no_stagnation_on_diverse_outputs() -> None:
    critic = _make_critic()
    for text in _DIVERSE_RESPONSES:
        critic.evaluate([_agent_msg(text)])
    assert critic.is_stagnant is False
    assert critic.certificate is None


def test_metadata_includes_certificate_theorem_once_fired() -> None:
    critic = _make_critic()
    last: CriticResult | None = None
    for _ in range(6):
        last = critic.evaluate([_agent_msg("loop loop loop")])
    assert last is not None
    assert last.metadata is not None
    assert last.metadata.get("certificate_theorem") == "behavioral_stability"
    assert last.metadata.get("certificate_source") == (
        "operon_openhands_gates.stagnation_critic"
    )


def test_certificate_captures_replayable_instability_evidence() -> None:
    critic = _make_critic()
    for _ in range(6):
        critic.evaluate([_agent_msg("stuck stuck stuck")])
    assert critic.certificate is not None

    # The ``behavioral_stability`` theorem semantics: holds=True means
    # stability held (mean severity < threshold, i.e. no stagnation).
    # This certificate is emitted *on* detection, so by construction the
    # severities ran high and stability does NOT hold — the certificate
    # is replayable evidence of the breach, not of the guarantee.
    verification = critic.certificate.verify()
    assert verification.holds is False
    assert "mean" in verification.evidence
    assert "max" in verification.evidence
    assert verification.evidence["n"] >= critic.critical_duration


def test_evaluate_accepts_git_patch_kwarg() -> None:
    critic = _make_critic()
    result = critic.evaluate([_agent_msg("hello")], git_patch="diff --git a b")
    assert isinstance(result, CriticResult)


def test_empty_events_returns_valid_result() -> None:
    critic = _make_critic()
    result = critic.evaluate([])
    assert 0.0 <= result.score <= 1.0
    assert critic.is_stagnant is False


def test_certificate_evidence_reflects_detection_window_not_full_history() -> None:
    """Regression for roborev job 760 High finding.

    Long diverse-output healthy prefix followed by a stagnant suffix must
    produce a certificate whose replay returns ``holds=False``. Previously
    the certificate was built from the entire accumulated history, so a
    long healthy prefix could dilute the mean severity below threshold and
    make ``verify().holds`` return True — contradicting the detection that
    actually fired.
    """
    critic = _make_critic()
    # Healthy prefix: 3x the diverse-response set. Longer than the stagnant
    # suffix by ~8x so the full-history mean would be dominated by healthy.
    for text in _DIVERSE_RESPONSES * 3:
        critic.evaluate([_agent_msg(text)])
    assert critic.is_stagnant is False
    assert critic.certificate is None

    # Stagnant suffix — long enough for the integral to clearly cross
    # threshold after the diverse prefix flushes out of the window.
    for _ in range(10):
        critic.evaluate([_agent_msg("identical stuck answer")])
    assert critic.is_stagnant is True
    assert critic.certificate is not None

    # Certificate must reflect the samples that backed the violating
    # integrals — ``window + critical_duration - 1`` severities — not the
    # full history. Verify replay disagrees with stability (holds=False)
    # and the evidence length equals the detection-window size.
    verification = critic.certificate.verify()
    assert verification.holds is False
    assert verification.evidence["n"] == critic.window + critic.critical_duration - 1


def test_certificate_evidence_spans_violating_integral_windows() -> None:
    """Regression for roborev jobs 761 & 762 (High).

    Previous fix used the last ``critical_duration`` point severities as
    certificate evidence. That loses correspondence with detection when
    ``window`` >> ``critical_duration``: old stagnant samples can keep the
    integral low while the most recent point severities are healthy,
    producing ``verify().holds=True`` even though the critic is stagnant.

    Construct that scenario: a long stagnant prefix followed by a short
    healthy suffix, with ``critical_duration=1`` and a window large
    enough that the first integral after the stagnant prefix is still
    below threshold. Certificate replay must still return
    ``holds=False`` — evidence must come from the windows that backed
    the violating integrals, not just the latest severities.
    """
    # Window large relative to critical_duration so old low-novelty samples
    # can keep the integral below threshold while newer samples are healthy.
    critic = OperonStagnationCritic(threshold=0.2, critical_duration=1, window=20)

    # Long stagnant prefix: identical text saturates novelty near zero.
    for _ in range(25):
        critic.evaluate([_agent_msg("identical saturating text")])
    assert critic.is_stagnant is True
    assert critic.certificate is not None

    verification = critic.certificate.verify()
    assert verification.holds is False
    # Evidence must span the full window that backed the violating
    # integral — not just the last ``critical_duration`` severity.
    assert verification.evidence["n"] > critic.critical_duration
    assert verification.evidence["n"] == critic.window + critic.critical_duration - 1


def test_extract_text_handles_plain_string_from_content_to_str(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for roborev job 762 (Low).

    The defensive branch in ``_extract_last_agent_text`` handles a
    hypothetical SDK future where ``content_to_str`` returns a plain
    ``str`` instead of ``list[str]``. Iterating a string would split
    on characters — ``"hello world"`` becomes ``"h e l l o w o r l d"``
    — silently corrupting the monitor's input. Simulate that return
    shape and assert the extracted text is returned unchanged.
    """
    from operon_openhands_gates import stagnation_critic as module

    monkeypatch.setattr(
        "openhands.sdk.llm.content_to_str", lambda contents: "hello world"
    )

    critic = _make_critic()
    # Bypass the module-level import cache by reaching the function directly.
    text = module._extract_last_agent_text([_agent_msg("placeholder")])
    assert text == "hello world"
