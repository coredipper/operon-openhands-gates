"""Dedupe an ``output.jsonl`` to one row per ``instance_id`` (max attempt).

The ``benchmarks.swebench.eval_infer`` script does not deduplicate rows
with the same ``instance_id`` before feeding them to
``swebench.harness.run_evaluation``. Under iterative refinement
(``OperonStagnationCritic`` + critic retries), the treatment
``output.jsonl`` carries one row per ``(instance_id, attempt)`` pair —
passing the raw file to eval would score each attempt separately.

The "final" patch per instance is the one the critic accepted on the
highest attempt (or the last attempt if the runner was killed), so we
keep the max-``attempt`` row and drop the rest.

Usage::

    python scripts/dedupe_for_eval.py \\
        eval/runs/treatment/.../output.jsonl \\
        eval/runs/treatment/.../output.dedup.jsonl

Idempotent: deduping an already-deduped file is a no-op.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def dedupe(rows: list[dict]) -> list[dict]:
    """Keep the max-attempt row per ``instance_id``.

    Preserves first-seen order for the kept rows, so downstream
    processing order matches the input's processing order.
    """
    by_iid: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        iid = r.get("instance_id")
        if not iid:
            raise ValueError(f"row missing instance_id: {r!r}")
        attempt = r.get("attempt", 1)
        if iid not in by_iid:
            order.append(iid)
            by_iid[iid] = r
        elif attempt > by_iid[iid].get("attempt", 1):
            by_iid[iid] = r
    return [by_iid[iid] for iid in order]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path, help="Path to output.jsonl")
    parser.add_argument("output_jsonl", type=Path, help="Path to write deduped jsonl")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.input_jsonl.open()]
    deduped = dedupe(rows)
    with args.output_jsonl.open("w") as f:
        for r in deduped:
            f.write(json.dumps(r) + "\n")

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "unique_instances": len(deduped),
                "output": str(args.output_jsonl),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
