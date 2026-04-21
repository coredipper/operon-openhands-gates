"""Structural reliability critics for the OpenHands Agent SDK.

Sibling of ``operon-langgraph-gates``: same substrate, same certificate
vocabulary, different host. The load-bearing claim is that harness-level
structural certificates are model-independent and framework-portable.
Two packages, one ``operon_ai`` core — this is that claim in code.

- ``OperonStagnationCritic``: Bayesian stagnation detection via
  ``EpiplexityMonitor``. Plugs into OpenHands' iterative-refinement loop
  through the ``CriticBase`` seam. Emits ``behavioral_stability``
  certificates on the transition to stagnant.

Backed by Paper 4 §4.3 (0.960 real-embed convergence accuracy).
"""

from .stagnation_critic import OperonStagnationCritic

__version__ = "0.1.0a3"

__all__ = ["OperonStagnationCritic", "__version__"]
