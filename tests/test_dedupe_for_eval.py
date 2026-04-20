"""Regression tests for ``scripts/dedupe_for_eval.py``.

Runs the module directly via ``importlib`` so the tests don't depend on
``scripts/`` being on ``sys.path``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dedupe_for_eval.py"
_spec = importlib.util.spec_from_file_location("_dedupe_for_eval", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
dedupe_mod = importlib.util.module_from_spec(_spec)
sys.modules["_dedupe_for_eval"] = dedupe_mod
_spec.loader.exec_module(dedupe_mod)


def test_dedupe_keeps_max_attempt_per_instance() -> None:
    rows = [
        {"instance_id": "a", "attempt": 1, "test_result": {"git_patch": "A1"}},
        {"instance_id": "b", "attempt": 1, "test_result": {"git_patch": "B1"}},
        {"instance_id": "b", "attempt": 2, "test_result": {"git_patch": "B2"}},
        {"instance_id": "c", "attempt": 1, "test_result": {"git_patch": "C1"}},
        {"instance_id": "c", "attempt": 2, "test_result": {"git_patch": "C2"}},
        {"instance_id": "c", "attempt": 3, "test_result": {"git_patch": "C3"}},
    ]
    deduped = dedupe_mod.dedupe(rows)
    assert len(deduped) == 3
    by_iid = {r["instance_id"]: r for r in deduped}
    assert by_iid["a"]["test_result"]["git_patch"] == "A1"
    assert by_iid["b"]["test_result"]["git_patch"] == "B2"  # attempt=2 wins
    assert by_iid["c"]["test_result"]["git_patch"] == "C3"  # attempt=3 wins


def test_dedupe_preserves_first_seen_order() -> None:
    """Downstream processing order should match input order of the
    first-seen row per instance.
    """
    rows = [
        {"instance_id": "b", "attempt": 1, "test_result": {"git_patch": ""}},
        {"instance_id": "a", "attempt": 1, "test_result": {"git_patch": ""}},
        {"instance_id": "b", "attempt": 2, "test_result": {"git_patch": ""}},
    ]
    deduped = dedupe_mod.dedupe(rows)
    assert [r["instance_id"] for r in deduped] == ["b", "a"]


def test_dedupe_is_idempotent() -> None:
    """Deduping an already-deduped file changes nothing."""
    rows = [
        {"instance_id": "a", "attempt": 1, "test_result": {"git_patch": ""}},
        {"instance_id": "b", "attempt": 2, "test_result": {"git_patch": ""}},
    ]
    once = dedupe_mod.dedupe(rows)
    twice = dedupe_mod.dedupe(once)
    assert once == twice


def test_dedupe_rejects_row_missing_instance_id() -> None:
    import pytest

    rows = [{"attempt": 1, "test_result": {"git_patch": ""}}]
    with pytest.raises(ValueError, match="missing instance_id"):
        dedupe_mod.dedupe(rows)


def test_main_cli_writes_expected_file(tmp_path: Path) -> None:
    """End-to-end CLI smoke: round-trip through main()."""
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    rows = [
        {"instance_id": "a", "attempt": 1, "test_result": {"git_patch": "p1"}},
        {"instance_id": "a", "attempt": 2, "test_result": {"git_patch": "p2"}},
        {"instance_id": "b", "attempt": 1, "test_result": {"git_patch": "p3"}},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rc = dedupe_mod.main([str(src), str(dst)])
    assert rc == 0
    out_rows = [json.loads(line) for line in dst.open()]
    assert len(out_rows) == 2
    assert {r["instance_id"]: r["test_result"]["git_patch"] for r in out_rows} == {
        "a": "p2",
        "b": "p3",
    }
