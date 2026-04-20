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
import json
import os
import sys
import tempfile
from pathlib import Path

# IMPORTANT: register_critic mutates benchmarks.utils.critics.CRITIC_NAME_TO_CLASS
# at import time. It must be imported *before* benchmarks.swebench.run_infer
# resolves its critic, which happens inside main(). Keep this first.
import register_critic  # noqa: F401 — load for side effect

# Location of the benchmarks repo clone. chdir into it for the full
# duration of the run for two reasons:
#
#   1. ``benchmarks.utils.version`` computes ``SDK_SHA`` at import time
#      via ``git submodule status <path>`` — resolves only when cwd is
#      inside the benchmarks git worktree (the parent of the submodule).
#   2. ``benchmarks.swebench.build_base_images._get_repo_root`` runs
#      ``git rev-parse --show-toplevel`` to locate
#      ``vendor/software-agent-sdk/.../Dockerfile``. chdir into the
#      submodule (tempting for workspace-root resolution) makes this
#      call return the SDK tree, doubling the path prefix.
#
# The UV workspace-root resolution inside
# ``openhands.agent_server.docker.build._default_sdk_project_root`` is
# cwd-independent once ``openhands-sdk`` / ``-tools`` / ``-workspace`` /
# ``-agent-server`` are editable-installed from the vendored SDK —
# ``__file__``-climb finds the workspace root without relying on cwd.
_VENDOR_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / ".vendor" / "benchmarks"


# Mapping from model-name prefix to the env var LiteLLM uses for that
# provider. Used by ``_inject_api_key`` when the user's LLM config JSON
# omits ``api_key`` — the wrapper reads the provider's env var and
# writes a temp config with ``api_key`` populated so the value travels
# into the remote agent-server conversation (the conversation runs
# inside the Docker workspace; env vars there don't include host keys
# unless explicitly forwarded, which the benchmarks CLI doesn't expose).
_PROVIDER_ENV_VARS: dict[str, str] = {
    "openai/": "OPENAI_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "gemini/": "GEMINI_API_KEY",
    "mistral/": "MISTRAL_API_KEY",
    "deepseek/": "DEEPSEEK_API_KEY",
    "groq/": "GROQ_API_KEY",
    "cohere/": "COHERE_API_KEY",
    "together_ai/": "TOGETHERAI_API_KEY",
    "openrouter/": "OPENROUTER_API_KEY",
    "xai/": "XAI_API_KEY",
}


def _load_dotenv(path: Path) -> None:
    """Read a simple ``KEY=VALUE`` ``.env`` file into ``os.environ``.

    Does not overwrite existing env vars — caller's shell-exported
    values win over the ``.env`` file, matching dotenv convention.
    Minimal parser (no quoting, no ``export`` prefix, no variable
    substitution) so the wrapper doesn't add a third-party dep for
    what's a one-file lookup. Lines starting with ``#`` are comments.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _inject_api_key(llm_config_path: Path) -> Path:
    """Return a JSON path with ``api_key`` populated from the provider's env var.

    If the user's config already sets ``api_key``, returns the path
    unchanged. Otherwise detects the provider from the ``model`` prefix
    (see :data:`_PROVIDER_ENV_VARS`), reads that env var, and writes a
    temp file with the key merged in. The temp file lives in the
    system temp dir with restrictive permissions (``0o600``) so the
    secret isn't world-readable on shared machines.

    Caller is responsible for unlinking the returned temp path after
    the runner exits (compare returned path to the input to decide).
    """
    data = json.loads(llm_config_path.read_text(encoding="utf-8"))
    if data.get("api_key"):
        return llm_config_path
    model = data.get("model", "")
    env_var = next(
        (env for prefix, env in _PROVIDER_ENV_VARS.items() if model.startswith(prefix)),
        None,
    )
    if env_var is None or not os.environ.get(env_var):
        # No known mapping or no env var set — let downstream LLM
        # validation handle the missing-credential case explicitly.
        return llm_config_path
    data["api_key"] = os.environ[env_var]
    fd, tmp_name = tempfile.mkstemp(prefix="llm-config-", suffix=".json")
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.write_text(json.dumps(data), encoding="utf-8")
    tmp_path.chmod(0o600)
    return tmp_path


# Flag-name suffixes that indicate a path-bearing argument. A forwarded
# passthrough flag whose name (before any ``=``) ends with one of these
# suffixes is treated as carrying a filesystem path whose value must be
# resolved against the caller's cwd before the wrapper chdirs into the
# vendor benchmarks clone.
_PATH_SUFFIXES = ("-path", "-file", "-dir", "-config")


def _normalize_path_passthrough(passthrough: list[str], original_cwd: Path) -> list[str]:
    """Normalize path-bearing passthrough args against ``original_cwd``.

    Looks for flags whose name ends with one of :data:`_PATH_SUFFIXES`
    (``--prompt-path``, ``--some-config``, etc.) in either
    ``--flag value`` or ``--flag=value`` form. Resolves the value
    against ``original_cwd`` so a user passing ``--prompt-path ../x.j2``
    ends up pointing at ``original_cwd/../x.j2``, not
    ``.vendor/benchmarks/../x.j2`` after the wrapper's chdir.

    Unknown flags pass through untouched. Matching uses exact suffix
    anchors, not substring — ``--pathwise`` is not treated as a path
    flag.

    "Is the next token another flag?" heuristic: only tokens beginning
    with ``--`` (long-form option, per argparse convention) are treated
    as flags. Single-dash tokens like ``-weird.txt`` or ``-5.0`` are
    treated as legitimate path values — rare but valid filenames
    shouldn't lose cwd normalization. If the user really passes a
    short flag (``-v``) as the value, downstream argparse will raise
    an "unrecognized argument" error on the mangled path, which is a
    clearer failure than silently swallowing a long flag.

    Trailing path-flags with no value — or path-flags immediately
    followed by another ``--``-prefixed flag — pass through as-is so
    the downstream argparse produces its canonical "expected one
    argument" error rather than silently consuming the next flag.
    """
    out: list[str] = []
    i = 0
    while i < len(passthrough):
        tok = passthrough[i]
        if tok.startswith("--") and any(tok.split("=", 1)[0].endswith(s) for s in _PATH_SUFFIXES):
            if "=" in tok:
                flag, value = tok.split("=", 1)
                out.append(f"{flag}={(original_cwd / value).resolve()}")
            elif i + 1 < len(passthrough) and not passthrough[i + 1].startswith("--"):
                # Consume the next token as the path value unless it's
                # itself a long-form flag (``--foo``). Single-dash
                # tokens are treated as values so that legitimate
                # dash-prefixed filenames (``-weird.txt``, ``-3``) keep
                # their cwd normalization. See the guard rationale in
                # the docstring.
                flag = tok
                value = passthrough[i + 1]
                out.extend([flag, str((original_cwd / value).resolve())])
                i += 1  # skip the value token on next iteration
            else:
                out.append(tok)
        else:
            out.append(tok)
        i += 1
    return out


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
            "Must at minimum set ``model``; ``api_key`` is optional and "
            "typically omitted — LiteLLM reads the provider's API key "
            "from the environment (e.g. ``ANTHROPIC_API_KEY``, "
            "``OPENAI_API_KEY``). Export the relevant key before running."
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
    llm_config_path = Path(args.llm_config).resolve()
    select_abs = str(Path(args.select).resolve())
    output_dir_abs = str(Path(args.output_dir).resolve())

    # Pull OPENAI_API_KEY / ANTHROPIC_API_KEY / etc. from a ``.env`` in
    # the caller's cwd before merging into the LLM config. Shell-
    # exported values win over file values, dotenv convention.
    _load_dotenv(Path.cwd() / ".env")

    # If the user's LLM config omits ``api_key``, synthesize a merged
    # temp config with the provider's env-var value. The agent-server
    # running inside the Docker workspace needs an explicit key on the
    # serialized LLM object — host env vars don't propagate into the
    # container unless explicitly forwarded, and the benchmarks CLI
    # doesn't expose a ``--forward-env`` flag.
    effective_llm_config = _inject_api_key(llm_config_path)
    llm_config_abs = str(effective_llm_config)
    temp_llm_config = effective_llm_config if effective_llm_config != llm_config_path else None

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

    # Normalize path-bearing passthrough flags (``--prompt-path``, etc.)
    # against the caller's original cwd *before* chdir — see
    # :func:`_normalize_path_passthrough` for the matching contract.
    original_cwd = Path.cwd()
    sys.argv.extend(_normalize_path_passthrough(passthrough, original_cwd))

    # Stay in the benchmarks repo cwd throughout — see module-level
    # comment on ``_VENDOR_BENCHMARKS_DIR``. Restore the original cwd
    # after the runner completes, and unlink the temp LLM config if we
    # synthesized one (secret cleanup).
    os.chdir(_VENDOR_BENCHMARKS_DIR)
    try:
        from benchmarks.swebench.run_infer import main as swebench_main

        swebench_main()
    finally:
        os.chdir(str(original_cwd))
        if temp_llm_config is not None:
            temp_llm_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
