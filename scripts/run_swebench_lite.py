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
import os
import sys
from pathlib import Path

# IMPORTANT: register_critic mutates benchmarks.utils.critics.CRITIC_NAME_TO_CLASS
# at import time. It must be imported *before* benchmarks.swebench.run_infer
# resolves its critic, which happens inside main(). Keep this first.
import register_critic  # noqa: F401 — load for side effect

# Location of the benchmarks repo clone. ``benchmarks.utils.version``
# computes the SDK SHA at import time by running ``git submodule status``
# in whatever cwd Python started in — if that's not the benchmarks git
# repo, the subprocess fails and the whole import aborts. We chdir into
# the clone before the deferred import, then restore. See
# ``scripts/README.md`` for the editable-install rationale.
_VENDOR_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / ".vendor" / "benchmarks"


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

    # Resolve caller-supplied paths relative to the caller's cwd NOW —
    # before we chdir into the vendor benchmarks clone. Otherwise the
    # benchmark runner's argparse would try to resolve them relative to
    # the clone and fail.
    llm_config_abs = str(Path(args.llm_config).resolve())
    select_abs = str(Path(args.select).resolve())
    output_dir_abs = str(Path(args.output_dir).resolve())

    # Reconstruct sys.argv for the benchmarks CLI. The benchmarks runner
    # uses argparse against sys.argv directly, so we rewrite it here.
    sys.argv = [
        "swebench-infer",
        llm_config_abs,  # positional in benchmarks' parser
        "--dataset",
        "princeton-nlp/SWE-bench_Lite",
        "--split",
        "test",
        "--select",
        select_abs,
        "--critic",
        critic_name,
        "--output-dir",
        output_dir_abs,
        "--workspace",
        "docker",
    ]
    if args.n_limit is not None:
        sys.argv.extend(["--n-limit", str(args.n_limit)])
    # Forward any extra benchmark-runner flags the caller passed through.
    sys.argv.extend(passthrough)

    # chdir so ``benchmarks.utils.version.get_sdk_sha()``'s
    # ``git submodule status`` call resolves. Deferred import: benchmarks
    # may pull in heavy deps (modal, swebench, docker). Keep them out of
    # module-load time so ``--help`` is fast.
    original_cwd = os.getcwd()
    os.chdir(_VENDOR_BENCHMARKS_DIR)
    try:
        from benchmarks.swebench.run_infer import main as swebench_main

        swebench_main()
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
