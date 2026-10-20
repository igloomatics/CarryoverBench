#!/usr/bin/env python3
"""Check task/verifier consistency for CarryoverBench.

This script performs static checks only:
- variant count / source-mode-payload matrix per benign task
- evaluator coverage for all base task ids
- skill payload mapping coverage hints
- success_pattern sanity snapshots
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from carryover.benchmark import BENCHMARK_TASKS
from carryover.verifier import load_evaluator


def _group_by_base():
    grouped: dict[str, list] = defaultdict(list)
    for task in BENCHMARK_TASKS:
        grouped[task.task_id.split("__", 1)[0]].append(task)
    return dict(sorted(grouped.items()))


def _skill_payloads(tasks: list) -> list[str]:
    return sorted([t.task_id.split("__", 2)[2] for t in tasks if t.attack_source.value == "skill"])


def _summary_rows() -> list[dict]:
    rows = []
    for base_id, tasks in _group_by_base().items():
        evaluator = load_evaluator(base_id)
        rows.append({
            "task_id": base_id,
            "variant_count": len(tasks),
            "modes": sorted({t.attack_mode.value for t in tasks if t.attack_mode}),
            "sources": sorted({t.attack_source.value for t in tasks}),
            "skill_payloads": _skill_payloads(tasks),
            "success_patterns": sorted({t.success_pattern for t in tasks if t.success_pattern}),
            "evaluator_registered": evaluator is not None,
            "variant_task_ids": [t.task_id for t in tasks],
        })
    return rows


def main() -> None:
    rows = _summary_rows()
    print("=== CarryoverBench Task / Verifier Consistency ===")
    for row in rows:
        print(f"- {row['task_id']}: variants={row['variant_count']}, modes={row['modes']}, sources={row['sources']}, evaluator={row['evaluator_registered']}")
        if row["skill_payloads"]:
            print(f"  skill payloads: {row['skill_payloads']}")
    anomalies = [r for r in rows if not r["evaluator_registered"] or r["variant_count"] not in (7, 10)]
    print("\n=== Anomalies ===")
    if anomalies:
        for row in anomalies:
            print(json.dumps(row, ensure_ascii=False, indent=2))
    else:
        print("No anomalies detected")


if __name__ == "__main__":
    main()
