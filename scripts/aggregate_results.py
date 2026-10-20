#!/usr/bin/env python3
"""Aggregate per-model run results into a side-by-side wide-table CSV.

Reads each run directory's ``summary_by_task.csv`` (written by
``run_benchmark.py``) and produces one CSV where rows are base tasks and
columns are grouped by model: ``<model>_ASR / <model>_Plant / <model>_Trig
/ <model>_Utility``. The top of the file has a two-line header so the
result lines up like the spreadsheet template (model name as the merged
header, indicator as the sub-header).

Example:
    python scripts/aggregate_results.py \\
        --run results/benchmark_20260513_120000:GPT5.4 \\
        --run results/benchmark_20260513_134500:claude-sonnet-4-6 \\
        --out results/cross_model_summary.csv

If ``:LABEL`` is omitted the script falls back to the ``model`` column
inside ``summary_by_task.csv``, then to the directory name.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

INDICATORS = ["ASR", "Plant", "Trig", "Utility"]
SUMMARY_FIELDS = {
    "ASR": "asr",
    "Plant": "asr_plant",
    "Trig": "asr_trigger",
    "Utility": "utility",
}


def _parse_run_arg(spec: str) -> tuple[Path, str | None]:
    """Split ``DIR[:LABEL]`` — works even when DIR contains ``:`` on Windows."""
    if ":" in spec:
        dir_part, _, label_part = spec.rpartition(":")
        if dir_part and Path(dir_part).exists():
            return Path(dir_part), label_part or None
    return Path(spec), None


def _load_summary(run_dir: Path) -> tuple[str, list[dict]]:
    """Return (model_label, rows) parsed from ``run_dir/summary_by_task.csv``."""
    csv_path = run_dir / "summary_by_task.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing summary_by_task.csv in {run_dir}. "
            "Re-run run_benchmark.py against this directory to regenerate it."
        )
    rows: list[dict] = []
    model = ""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if not model and row.get("model"):
                model = row["model"]
    return model, rows


def _fmt(value: str) -> str:
    """Pass-through for floats; empty strings stay empty."""
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return value


def aggregate(
    run_specs: list[tuple[Path, str | None]],
    out_path: Path,
) -> Path:
    """Build the wide table and write it to ``out_path``."""
    models: list[str] = []
    per_model_rows: dict[str, dict[str, dict]] = {}  # label -> {base_task_id: row}
    task_order: list[str] = []  # preserve first-seen order, but keep TOTAL last
    task_names: dict[str, str] = {}

    for run_dir, override_label in run_specs:
        file_model, rows = _load_summary(run_dir)
        label = override_label or file_model or run_dir.name
        if label in per_model_rows:
            print(
                f"warning: duplicate model label {label!r}; later run overrides earlier",
                file=sys.stderr,
            )
        else:
            models.append(label)
        per_model_rows[label] = {}
        for row in rows:
            tid = row["base_task_id"]
            per_model_rows[label][tid] = row
            if tid not in task_names:
                task_names[tid] = row.get("task_name") or tid
                if tid != "TOTAL":
                    task_order.append(tid)

    if "TOTAL" in task_names:
        task_order.append("TOTAL")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # Two-line header: model names across the top, indicators below.
        header_top: list[str] = ["", ""]
        header_sub: list[str] = ["base_task_id", "task_name"]
        for m in models:
            header_top.extend([m] + [""] * (len(INDICATORS) - 1))
            header_sub.extend(INDICATORS)
        writer.writerow(header_top)
        writer.writerow(header_sub)

        for tid in task_order:
            row_out: list[str] = [tid, task_names.get(tid, tid)]
            for m in models:
                src = per_model_rows.get(m, {}).get(tid)
                if not src:
                    row_out.extend([""] * len(INDICATORS))
                    continue
                for ind in INDICATORS:
                    row_out.append(_fmt(src.get(SUMMARY_FIELDS[ind], "")))
            writer.writerow(row_out)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multiple run_benchmark.py result dirs into a single "
            "wide CSV (rows=base tasks, columns=<model>×{ASR,Plant,Trig,Utility})."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/aggregate_results.py \\\n"
            "      --run results/benchmark_AAA:GPT5.4 \\\n"
            "      --run results/benchmark_BBB:claude-sonnet-4-6 \\\n"
            "      --out results/cross_model_summary.csv\n"
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="DIR[:LABEL]",
        help=(
            "Path to a run directory containing summary_by_task.csv. "
            "Append ':LABEL' to override the model label used as the "
            "column-group header. Repeat for each model."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/cross_model_summary.csv"),
        help="Output CSV path (default: results/cross_model_summary.csv)",
    )
    args = parser.parse_args()

    run_specs: list[tuple[Path, str | None]] = []
    for spec in args.run:
        run_dir, label = _parse_run_arg(spec)
        if not run_dir.is_dir():
            print(f"error: not a directory: {run_dir}", file=sys.stderr)
            sys.exit(1)
        run_specs.append((run_dir, label))

    out = aggregate(run_specs, args.out)
    print(f"Cross-model summary written: {out}")


if __name__ == "__main__":
    main()
