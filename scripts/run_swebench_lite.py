"""SWE-bench-lite delta harness: baseline vs OperonStagnationCritic.

Thin wrapper around ``benchmarks.swebench.run_infer.main``. Picks the
right critic name based on ``--condition``, registers
:class:`OperonStagnationCritic` with the benchmarks CLI before the
runner resolves critics, then delegates to the upstream runner.

Usage:

    # Smoke test: 1 known-pass astropy instance, treatment condition.
    python scripts/run_swebench_lite.py \\
        --condition operon_stagnation \\
        --llm-config scripts/llm.json \\
        --output-dir eval/runs/smoke \\
        --select scripts/smoke_1.txt \\
        --n-limit 1

    # Full baseline (30 instances).
    python scripts/run_swebench_lite.py \\
        --condition baseline \\
        --llm-config scripts/llm.json \\
        --output-dir eval/runs/baseline \\
        --select scripts/instances.txt \\
        --n-limit 30

See ``scripts/README.md`` for end-to-end reproduction commands.
"""

from __future__ import annotations

import argparse
import sys

# IMPORTANT: register_critic mutates benchmarks.utils.critics.CRITIC_NAME_TO_CLASS
# at import time. It must be imported *before* benchmarks.swebench.run_infer
# resolves its critic, which happens inside main(). Keep this first.
import register_critic  # noqa: F401 — load for side effect


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--condition",
        choices=["baseline", "operon_stagnation"],
        required=True,
        help=(
            "Which critic to use. 'baseline' = AgentFinishedCritic "
            "(benchmarks default), 'operon_stagnation' = OperonStagnationCritic."
        ),
    )
    parser.add_argument(
        "--llm-config",
        required=True,
        help=(
            "Path to the OpenHands LLM config JSON "
            "(see https://docs.openhands.dev/sdk/llm). "
            "Must at minimum set ``model`` and ``api_key``."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base directory for benchmark artifacts.",
    )
    parser.add_argument(
        "--select",
        default="scripts/instances.txt",
        help="Path to newline-separated instance-ID selection file.",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=None,
        help="Optional hard cap on instances processed (useful for smoke tests).",
    )
    args, passthrough = parser.parse_known_args()

    critic_name = "finish_with_patch" if args.condition == "baseline" else "operon_stagnation"

    # Reconstruct sys.argv for the benchmarks CLI. The benchmarks runner
    # uses argparse against sys.argv directly, so we rewrite it here.
    sys.argv = [
        "swebench-infer",
        "--llm-config-path",
        args.llm_config,
        "--dataset",
        "princeton-nlp/SWE-bench_Lite",
        "--split",
        "test",
        "--select",
        args.select,
        "--critic",
        critic_name,
        "--output-dir",
        args.output_dir,
        "--workspace",
        "docker",
    ]
    if args.n_limit is not None:
        sys.argv.extend(["--n-limit", str(args.n_limit)])
    # Forward any extra benchmark-runner flags the caller passed through.
    sys.argv.extend(passthrough)

    # Deferred import: benchmarks may pull in heavy deps (modal, swebench,
    # docker). Keep them out of module-load time so ``--help`` is fast.
    from benchmarks.swebench.run_infer import main as swebench_main

    swebench_main()


if __name__ == "__main__":
    main()
