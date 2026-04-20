"""Regression tests for ``scripts/run_swebench_lite.py`` passthrough
path normalization.

Covers roborev #817 / #818 / #820: path-bearing benchmark CLI flags
forwarded through ``argparse.parse_known_args`` must be resolved
against the caller's original cwd *before* the wrapper chdirs into the
vendor benchmarks clone. Otherwise a user passing
``--prompt-path ../custom.j2`` ends up pointing at a file under the
vendor clone, not their working directory.

Tests import only the pure ``_normalize_path_passthrough`` helper —
not the wrapper's ``main()`` — so the benchmarks package doesn't have
to be importable for the dev test suite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_swebench_lite.py"


def _load_wrapper_module_without_side_effects():
    """Load ``scripts/run_swebench_lite.py`` with ``register_critic`` /
    benchmarks imports stubbed out.

    The wrapper's top-level ``import register_critic`` pulls in
    ``benchmarks.utils.critics``, which in turn loads the heavy
    benchmarks tree (modal, docker, swebench...). For testing the pure
    ``_normalize_path_passthrough`` helper we don't need any of that,
    so we install a placeholder ``register_critic`` module in
    ``sys.modules`` before executing the wrapper's source.
    """
    import types

    stub = types.ModuleType("register_critic")
    sys.modules.setdefault("register_critic", stub)

    spec = importlib.util.spec_from_file_location("_run_swebench_lite", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_run_swebench_lite"] = module
    spec.loader.exec_module(module)
    return module


wrapper = _load_wrapper_module_without_side_effects()
normalize = wrapper._normalize_path_passthrough


def test_path_flag_with_separate_value_resolves_against_original_cwd(
    tmp_path: Path,
) -> None:
    (tmp_path / "custom.j2").write_text("prompt")
    out = normalize(["--prompt-path", "custom.j2"], tmp_path)
    assert out[0] == "--prompt-path"
    assert out[1] == str((tmp_path / "custom.j2").resolve())


def test_path_flag_with_equals_syntax_is_normalized(tmp_path: Path) -> None:
    (tmp_path / "my.json").write_text("{}")
    out = normalize(["--some-config=my.json"], tmp_path)
    assert out == [f"--some-config={(tmp_path / 'my.json').resolve()}"]


def test_non_path_flag_passes_through_untouched(tmp_path: Path) -> None:
    out = normalize(["--num-workers", "4", "--temperature", "0.7"], tmp_path)
    assert out == ["--num-workers", "4", "--temperature", "0.7"]


def test_mixed_flags_preserve_order_and_values(tmp_path: Path) -> None:
    (tmp_path / "x.j2").write_text("")
    out = normalize(
        ["--num-workers", "4", "--prompt-path", "x.j2", "--extra"],
        tmp_path,
    )
    assert out[0] == "--num-workers"
    assert out[1] == "4"
    assert out[2] == "--prompt-path"
    assert out[3] == str((tmp_path / "x.j2").resolve())
    assert out[4] == "--extra"


def test_absolute_path_preserved(tmp_path: Path) -> None:
    abs_target = str((tmp_path / "abs.j2").resolve())
    out = normalize(["--prompt-path", abs_target], tmp_path)
    # ``(tmp_path / abs_path).resolve()`` equals ``abs_path`` when
    # already absolute — preserves the caller's intent.
    assert out[1] == abs_target


def test_trailing_path_flag_with_no_value_passes_through_as_is(
    tmp_path: Path,
) -> None:
    """Dangling ``--prompt-path`` at end of argv: let downstream argparse
    produce the canonical error rather than silently dropping the flag.
    """
    out = normalize(["--some-setting", "x", "--prompt-path"], tmp_path)
    assert out == ["--some-setting", "x", "--prompt-path"]


def test_path_flag_followed_by_long_flag_is_not_swallowed(
    tmp_path: Path,
) -> None:
    """Roborev #821 Medium: ``--prompt-path --num-workers 4`` must NOT
    consume ``--num-workers`` as the path value. The normalizer
    detects ``--``-prefixed tokens as long-form flags and leaves the
    dangling path flag alone so downstream argparse produces its
    canonical "expected one argument" error.
    """
    out = normalize(["--prompt-path", "--num-workers", "4"], tmp_path)
    assert out == ["--prompt-path", "--num-workers", "4"]


def test_path_flag_followed_by_dash_leading_filename_is_still_normalized(
    tmp_path: Path,
) -> None:
    """Roborev #822 Medium: rare but legitimate dash-prefixed filenames
    (``-weird.txt``, negative-ish names from tooling output) must keep
    cwd normalization. The guard treats only ``--``-prefixed tokens as
    flags; a single-dash token is a value.
    """
    (tmp_path / "-weird.txt").write_text("")
    out = normalize(["--prompt-path", "-weird.txt"], tmp_path)
    assert out[0] == "--prompt-path"
    assert out[1] == str((tmp_path / "-weird.txt").resolve())


def test_path_flag_followed_by_short_flag_treats_short_flag_as_value(
    tmp_path: Path,
) -> None:
    """Documented behavior boundary: a short flag like ``-v`` after a
    path flag will be normalized as a path (``/abs/-v``). Downstream
    argparse then raises ``unrecognized argument: /abs/-v`` — a
    clearer failure than silently consuming the short flag. The
    benchmarks SWE-bench CLI uses only long-form flags, so this case
    is a no-op in practice; the test pins the documented boundary.
    """
    out = normalize(["--prompt-path", "-v"], tmp_path)
    assert out[0] == "--prompt-path"
    assert out[1] == str((tmp_path / "-v").resolve())


def test_suffix_match_is_exact_not_substring(tmp_path: Path) -> None:
    """A flag like ``--pathwise`` contains ``path`` but isn't a path
    flag; the matcher uses suffix anchors (``-path`` etc.), so this
    passes through untouched.
    """
    out = normalize(["--pathwise", "something"], tmp_path)
    assert out == ["--pathwise", "something"]


def test_file_and_dir_and_config_suffixes_all_covered(tmp_path: Path) -> None:
    """All four documented suffixes (-path, -file, -dir, -config) get
    normalized. Easy to regress one while editing the others.
    """
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "b").mkdir()
    out = normalize(
        [
            "--prompt-path",
            "a.txt",
            "--custom-file",
            "a.txt",
            "--some-dir",
            "b",
            "--llm-config",
            "a.txt",
        ],
        tmp_path,
    )
    assert out[1] == str((tmp_path / "a.txt").resolve())
    assert out[3] == str((tmp_path / "a.txt").resolve())
    assert out[5] == str((tmp_path / "b").resolve())
    assert out[7] == str((tmp_path / "a.txt").resolve())
