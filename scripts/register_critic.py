"""Register OperonStagnationCritic with OpenHands' benchmarks CLI.

Import-time side effect: mutate
``benchmarks.utils.critics.CRITIC_NAME_TO_CLASS`` so ``--critic
operon_stagnation`` resolves to :class:`OperonStagnationCritic`.

The benchmarks runner builds its critic via ``create_critic(args)`` which
looks up the critic name in this dict (see
``benchmarks/utils/critics.py:create_critic``). No PR to the benchmarks
repo is required — registration happens in this process only, no
external state is affected.

Usage::

    import register_critic  # noqa: F401  — load for side effect
    from benchmarks.swebench.run_infer import main as swebench_main
    swebench_main()
"""

from __future__ import annotations

from benchmarks.utils import critics

from operon_openhands_gates import OperonStagnationCritic

_CRITIC_NAME = "operon_stagnation"

if _CRITIC_NAME in critics.CRITIC_NAME_TO_CLASS:
    # Idempotent: re-importing this module in the same process is a no-op.
    if critics.CRITIC_NAME_TO_CLASS[_CRITIC_NAME] is not OperonStagnationCritic:
        raise RuntimeError(
            f"critic name {_CRITIC_NAME!r} already registered to a different class: "
            f"{critics.CRITIC_NAME_TO_CLASS[_CRITIC_NAME]!r}"
        )
else:
    critics.CRITIC_NAME_TO_CLASS[_CRITIC_NAME] = OperonStagnationCritic
