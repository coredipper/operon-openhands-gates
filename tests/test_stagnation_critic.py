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
    assert critic.certificate.theorem == "behavioral_stability_windowed"


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
    assert last.metadata.get("certificate_theorem") == "behavioral_stability_windowed"
    assert last.metadata.get("certificate_source") == ("operon_openhands_gates.stagnation_critic")


def test_certificate_captures_replayable_instability_evidence() -> None:
    critic = _make_critic()
    for _ in range(6):
        critic.evaluate([_agent_msg("stuck stuck stuck")])
    assert critic.certificate is not None

    # Replay semantic: ``max(window_severity_means) < stability_threshold``
    # means stability held. The cert fires on detection, which means
    # every violating window's mean severity was above threshold, so
    # ``max`` is above threshold and ``holds`` is False.
    verification = critic.certificate.verify()
    assert verification.holds is False
    assert "mean" in verification.evidence
    assert "max" in verification.evidence
    # One entry per violating rolling window.
    assert verification.evidence["n"] == critic.critical_duration


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

    # Certificate stores one mean severity per violating rolling window,
    # so ``evidence["n"] == critical_duration`` (not the full history).
    # Replay: ``max < stability_threshold`` is False since every window
    # exceeded threshold.
    verification = critic.certificate.verify()
    assert verification.holds is False
    assert verification.evidence["n"] == critic.critical_duration


def test_certificate_evidence_is_per_window_severity_means() -> None:
    """Regression for roborev jobs 761, 762, 764 (High).

    Detection fires when each of ``critical_duration`` rolling-window
    integrals is below the detection threshold. Certificate evidence
    must be one mean per violating window — not a flattened raw-severity
    sequence. The replay check is ``max(window_means) < stability_threshold``,
    which exactly mirrors detection's "every window violates" predicate.
    """
    critic = OperonStagnationCritic(threshold=0.2, critical_duration=1, window=20)
    for _ in range(25):
        critic.evaluate([_agent_msg("identical saturating text")])
    assert critic.is_stagnant is True
    assert critic.certificate is not None

    # One signal value per violating rolling window.
    signal_values = critic.certificate.parameters["signal_values"]
    assert len(signal_values) == critic.critical_duration

    # Replay: max of per-window means exceeds stability threshold.
    verification = critic.certificate.verify()
    assert verification.holds is False
    assert verification.evidence["n"] == critic.critical_duration
    assert verification.evidence["max"] >= 1.0 - critic.threshold


def test_empty_evidence_is_rejected_not_vacuously_stable() -> None:
    """Regression for roborev job 767 (Low).

    The critic's Pydantic validator enforces ``critical_duration >= 1``,
    so every legitimately-emitted certificate carries at least one
    violating window. Empty ``signal_values`` can therefore only come
    from a malformed or externally-constructed certificate — treating
    that as vacuous stability would turn a broken cert into a silent
    "stability held" attestation. Two layers of defense: the emitter
    raises, and the verifier rejects.
    """
    from operon_ai.core.certificate import resolve_verify_fn

    from operon_openhands_gates.stagnation_critic import _emit_certificate

    # Emission-time guard: can't produce an empty cert in the first place.
    with pytest.raises(ValueError, match="non-empty"):
        _emit_certificate(
            window_severity_means=(),
            threshold=0.8,
            detection_index=42,
        )

    # Verifier-time guard: an externally-constructed empty cert fails
    # replay rather than silently attesting stability.
    verify_fn = resolve_verify_fn("behavioral_stability_windowed")
    assert verify_fn is not None
    holds, evidence = verify_fn({"signal_values": (), "threshold": 0.8})
    assert holds is False
    assert evidence["reason"] == "empty_evidence"
    assert evidence["n"] == 0


def test_certificate_verify_treats_threshold_equality_as_stable() -> None:
    """Regression for roborev job 766 (Low).

    Detection uses a strict ``integral < threshold`` predicate, so a
    window with ``integral == threshold`` is NOT stagnant (stable). The
    severity-domain complement is ``mean(severity) <= 1 - threshold``,
    which is inclusive at the boundary. A strict ``<`` in the verifier
    misclassifies the boundary as unstable.
    """
    from operon_openhands_gates.stagnation_critic import _emit_certificate

    # Window mean exactly at the stability threshold: detection would say
    # stable (integral >= detection threshold), so verify must agree.
    cert_boundary = _emit_certificate(
        window_severity_means=(0.8,),
        threshold=0.8,
        detection_index=1,
    )
    assert cert_boundary.verify().holds is True

    # Just below threshold: clearly stable.
    cert_stable = _emit_certificate(
        window_severity_means=(0.799,),
        threshold=0.8,
        detection_index=1,
    )
    assert cert_stable.verify().holds is True

    # Just above threshold: clearly unstable (detection fires).
    cert_unstable = _emit_certificate(
        window_severity_means=(0.801,),
        threshold=0.8,
        detection_index=1,
    )
    assert cert_unstable.verify().holds is False


def test_certificate_handles_overlapping_windows_counterexample() -> None:
    """Regression for roborev job 764 High (reviewer's counterexample).

    Overlapping rolling windows weight interior samples more heavily
    than a flattened mean. Example: ``window=2, cd=2`` with severities
    ``[0.61, 1.0, 0.61]``. Both window means are ``0.805`` (detection
    fires against stability threshold ``0.8``). The flattened mean over
    the union is only ``0.74``, which would incorrectly say stability
    held.

    Exercises ``_emit_certificate`` directly with constructed inputs so
    the discriminator is input-space-precise, not embedder-dependent.
    """
    from operon_openhands_gates.stagnation_critic import _emit_certificate

    # Reviewer's scenario: two overlapping windows each with mean 0.805.
    cert = _emit_certificate(
        window_severity_means=(0.805, 0.805),
        threshold=0.8,  # stability threshold = 1 - detection threshold
        detection_index=3,
    )
    verification = cert.verify()
    assert verification.holds is False
    assert verification.evidence["max"] == 0.805
    assert verification.evidence["n"] == 2

    # Control: under the old flattened-mean verify, these severities
    # would have given mean=0.74 and holds=True. Confirm that using
    # the correct per-window-means input with max-based verify we get
    # the correct verdict.

    # Positive control: every window mean below threshold → stable.
    stable_cert = _emit_certificate(
        window_severity_means=(0.5, 0.5),
        threshold=0.8,
        detection_index=3,
    )
    assert stable_cert.verify().holds is True


def test_certificate_conclusion_uses_exact_detection_index() -> None:
    """Regression for roborev jobs 765 Low, 773 Low, and 778 Low.

    The conclusion text must quote the exact evaluation count at the
    moment the cert was emitted — and must *keep* reporting that value
    after subsequent ``evaluate()`` calls. Breaking out on first
    emission would let a future regression that rewrites the conclusion
    on later calls pass undetected. Continue evaluating for several
    more turns after emission and assert the conclusion still points at
    the original emission turn.
    """
    import re

    critic = OperonStagnationCritic(threshold=0.2, critical_duration=1, window=20)

    emission_turn: int | None = None
    for turn in range(1, 41):
        critic.evaluate([_agent_msg("identical saturating text")])
        if critic.certificate is not None and emission_turn is None:
            emission_turn = turn
            # Don't break — continue evaluating to verify the conclusion
            # stays pinned to the original emission turn.

    assert emission_turn is not None, "expected a certificate within 40 turns"
    assert critic.certificate is not None

    # By now we've done 40 - emission_turn additional evaluate() calls
    # after the cert appeared. A regression that rewrites the conclusion
    # on later calls (e.g. reporting the final loop count) would show up
    # as a number greater than emission_turn.
    assert 40 - emission_turn >= 5, (
        f"fixture too tight: only {40 - emission_turn} post-emission turns"
    )

    conclusion = critic.certificate.conclusion
    match = re.search(r"after (\d+) measurements", conclusion)
    assert match is not None, f"conclusion lacks measurement count: {conclusion!r}"
    reported_n = int(match.group(1))

    # One evaluate() call produces one monitor measurement, so the
    # reported N equals the turn count at which the cert first fired —
    # not the total number of evaluations performed since.
    assert reported_n == emission_turn, (
        f"conclusion reports N={reported_n} but emission turn was {emission_turn}; "
        f"cert conclusion must stay pinned to first-emission index"
    )


def test_certificate_replay_agrees_with_detection_for_threshold_above_half() -> None:
    """Regression for roborev job 763 (Medium).

    Detection compares epiplexic integrals to threshold directly
    (integral < threshold), while the ``behavioral_stability`` replay
    checks mean(severity) < threshold. These are equivalent only when
    threshold <= 0.5 — for threshold > 0.5, a violating integral only
    guarantees mean(severity) > 1 - threshold, not > threshold, so a
    literal-threshold cert can say ``holds=True`` while the critic has
    flipped stagnant.

    The fix is to store ``1 - threshold`` (the stability threshold) in
    the certificate parameters, so verify's ``< threshold`` semantic
    matches detection at every threshold value.
    """
    # Threshold clearly above 0.5: previously a gap configuration.
    critic = OperonStagnationCritic(threshold=0.7, critical_duration=1, window=5)
    for _ in range(10):
        critic.evaluate([_agent_msg("identical saturating text")])
    assert critic.is_stagnant is True
    assert critic.certificate is not None

    # The cert must store the translated (stability) threshold, not the
    # detection threshold.
    assert critic.certificate.parameters["threshold"] == 1.0 - critic.threshold

    verification = critic.certificate.verify()
    assert verification.holds is False


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

    monkeypatch.setattr("openhands.sdk.llm.content_to_str", lambda contents: "hello world")

    # Bypass the module-level import cache by reaching the function directly.
    text = module._extract_last_agent_text([_agent_msg("placeholder")])
    assert text == "hello world"


def test_emission_failure_leaves_state_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression for roborev jobs 788 & 790 Medium.

    If ``_emit_certificate`` raises (e.g. the theorem isn't registered),
    the critic must NOT be left in a permanent ``is_stagnant=True /
    certificate=None`` state — the ``was_stagnant`` guard in the next
    evaluate() call would otherwise suppress cert emission forever.
    Build the cert before flipping ``_is_stagnant`` so a failure keeps
    the state retryable.
    """
    critic = _make_critic()

    # Force _emit_certificate to raise on first stagnation transition.
    raise_count = [0]
    from operon_openhands_gates import stagnation_critic as module

    original_emit = module._emit_certificate

    def flaky_emit(*args: object, **kwargs: object) -> object:
        if raise_count[0] == 0:
            raise_count[0] += 1
            raise RuntimeError("simulated resolver failure")
        return original_emit(*args, **kwargs)

    monkeypatch.setattr(module, "_emit_certificate", flaky_emit)

    # Drive the critic to the stagnant transition; the first emission fails.
    with pytest.raises(RuntimeError, match="simulated resolver failure"):
        for _ in range(6):
            critic.evaluate([_agent_msg("stuck response")])
            if critic._low_integral_streak >= critic.critical_duration:
                # Next call would transition — force it here to confirm.
                break
        # If the loop didn't trigger the exception path, call once more.
        critic.evaluate([_agent_msg("stuck response")])

    # State after failure: critic is NOT stagnant, no certificate.
    # (The transition was attempted and the state flip was rolled back.)
    assert critic.is_stagnant is False
    assert critic.certificate is None

    # Later call retries emission (now with the real _emit_certificate)
    # and successfully produces the certificate.
    for _ in range(6):
        critic.evaluate([_agent_msg("stuck response")])
    assert critic.is_stagnant is True
    assert critic.certificate is not None


def test_windowed_theorem_resolves_through_upstream_registry() -> None:
    """Same-process contract: windowed theorem resolves to a callable,
    distinct from the legacy theorem's callable. Uses the public
    ``resolve_verify_fn`` API (operon-ai>=0.36.1); no coupling to any
    underscore-prefixed upstream symbol.
    """
    from operon_ai.core.certificate import resolve_verify_fn

    windowed = resolve_verify_fn("behavioral_stability_windowed")
    legacy = resolve_verify_fn("behavioral_stability")

    assert windowed is not None and callable(windowed)
    assert legacy is not None and callable(legacy)
    # Distinct theorems — we added a new entry, didn't alias the old.
    assert windowed is not legacy


def test_windowed_theorem_resolves_without_this_package_imported(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Regression for roborev jobs 780/781/786 Low.

    The user-facing guarantee this package advertises (post-0.36.0) is
    that any process with ``operon-ai>=0.36.0`` resolves
    ``behavioral_stability_windowed`` through the canonical
    ``_THEOREM_FN_PATHS`` — no import of this sibling package required.

    Same-process tests cannot prove that claim: by the time the test
    runs, prior imports have populated module state, so a regression
    that accidentally re-introduced an import-time side effect would
    still pass. Spawn a subprocess that imports *only* ``operon_ai``
    and asserts the resolver returns a callable. Importantly: do NOT
    import ``operon_openhands_gates`` at all, so a future regression
    that moves registration back to a sibling side-effect fails here.

    Writes the probe with explicit ``encoding="utf-8"`` and ASCII-only
    content to avoid 780's em-dash / non-UTF-8-locale hazard.
    """
    import subprocess
    import sys

    probe = tmp_path / "probe.py"
    probe.write_text(
        "from operon_ai.core.certificate import Certificate, resolve_verify_fn\n"
        "\n"
        "# Exercise both public surfaces the package relies on. (1)\n"
        "# resolve_verify_fn must be importable and return a callable\n"
        "# for the windowed theorem — this catches a packaging/export\n"
        "# regression specific to that alias. (2) Certificate.from_theorem\n"
        "# must produce a cert that verifies as 'breach' on per-window\n"
        "# means above the stability threshold — this catches the\n"
        "# factory's resolution path.\n"
        "assert callable(resolve_verify_fn('behavioral_stability_windowed')), (\n"
        "    'public resolve_verify_fn did not return a callable'\n"
        ")\n"
        "cert = Certificate.from_theorem(\n"
        "    theorem='behavioral_stability_windowed',\n"
        "    parameters={'signal_values': (0.9,), 'threshold': 0.8},\n"
        "    conclusion='cold-process probe',\n"
        "    source='test',\n"
        ")\n"
        "verification = cert.verify()\n"
        "assert verification.holds is False, verification\n"
        "\n"
        "import sys as _sys\n"
        "assert 'operon_openhands_gates' not in _sys.modules, (\n"
        "    'sibling package was imported as a side effect; resolution "
        "must not depend on it'\n"
        ")\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.stdout.strip().endswith("ok")
