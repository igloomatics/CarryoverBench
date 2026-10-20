#!/usr/bin/env python3
"""
CarryoverBench benign baseline runner.

Runs one clean baseline per benign task and scores Utility.
Additionally re-judges the same benign session evidence against every
INSTANT-mode attack variant registered for the base task (1-Trig only,
since a benign run has no plant/trigger split). Any variant that flags
the benign trace counts as a false-positive: the agent moved attacker
markers around without ever being prompted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

from carryover.benchmark import BASELINE_TASKS, BENCHMARK_TASKS
from carryover.tester import CarryoverTester
from carryover.types import AttackMode, BenchmarkTask, TrialResult
from carryover.verifier import load_evaluator

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_TASK_FILTERS: list[str] | None = None


def _is_ablation_variant(variant_task_id: str) -> bool:
    """Ablation variants carry '_abl' in their payload_id (e.g. monitor_ablT1d1).

    They are excluded from main-benchmark weighting so the baseline utility
    is comparable to the headline number reported in the paper.
    """
    parts = variant_task_id.split("__")
    payload_id = parts[2] if len(parts) >= 3 else ""
    return "_abl" in payload_id


def _attack_variant_weights(include_ablation: bool = False) -> dict[str, int]:
    """Per–base-task variant count, matching what main benchmark averages over.

    Counts BENCHMARK_TASKS entries grouped by base task_id (the part before
    the first '__'). With ``include_ablation=False`` (default), the
    travel_booking ablation matrix is excluded, giving the same denominator
    as the headline benchmark.
    """
    weights: dict[str, int] = {}
    for t in BENCHMARK_TASKS:
        if not include_ablation and _is_ablation_variant(t.task_id):
            continue
        base = t.task_id.split("__", 1)[0]
        weights[base] = weights.get(base, 0) + 1
    return weights


def _baseline_task_ids() -> list[str]:
    return [t.task_id for t in BASELINE_TASKS]


def _normalize_task_filters(task_filter: str | Sequence[str] | None) -> list[str] | None:
    if task_filter is None:
        return None
    if isinstance(task_filter, str):
        return [task_filter]
    filters = [f.strip() for f in task_filter if f and str(f).strip()]
    return filters or None


def _select_tasks(task_filters: list[str] | None) -> list:
    if not task_filters:
        return list(BASELINE_TASKS)

    seen: set[str] = set()
    tasks: list = []
    for tf in task_filters:
        matched = [t for t in BASELINE_TASKS if t.task_id == tf]
        for t in matched:
            if t.task_id not in seen:
                seen.add(t.task_id)
                tasks.append(t)

    if not tasks:
        all_ids = [t.task_id for t in BASELINE_TASKS]
        logger.error(f"No baseline tasks matched filters {task_filters!r}. Available: {all_ids}")
        sys.exit(1)
    return tasks


def _instant_variants(base_task_id: str) -> list[BenchmarkTask]:
    """Return every INSTANT-mode attack variant whose base task matches.

    Benign baselines are 1-Trig by construction (no plant phase, no
    session 2), so DELAYED/PERSISTENT variants are skipped — their plant
    checks look at MEMORY.md / HEARTBEAT.md which the benign agent has
    no reason to touch.
    """
    return [
        t for t in BENCHMARK_TASKS
        if t.task_id.startswith(base_task_id + "__")
        and t.attack_mode == AttackMode.INSTANT
    ]


def _judge_baseline_false_positive(
    base_task: BenchmarkTask,
    result: TrialResult,
    workspace: Path,
) -> list[dict]:
    """Re-judge a benign trial against every INSTANT variant of its base task.

    For each variant we call the same ``evaluate_asr`` the attack runner uses;
    the only thing that's different from a real attack trial is that the
    session evidence came from a clean run, so a True here is a false positive.

    The evaluator writes ``asr_plant`` / ``asr_trigger`` onto ``result`` as a
    side effect — we snapshot and restore them so successive variant judgments
    don't see each other's writes.
    """
    evaluator = load_evaluator(base_task.task_id)
    variants = _instant_variants(base_task.task_id)
    if evaluator is None or not variants:
        return []

    snapshots: list[dict] = []
    saved_plant, saved_trigger, saved_asr = result.asr_plant, result.asr_trigger, result.asr
    for variant in variants:
        result.asr_plant = False
        result.asr_trigger = False
        result.asr = False
        try:
            hit = bool(evaluator.evaluate_asr(variant, result, workspace=workspace))
        except Exception as exc:
            logger.warning(
                f"  variant {variant.task_id}: evaluate_asr raised "
                f"{type(exc).__name__}: {exc} — recording asr=False"
            )
            hit = False
        snapshots.append({
            "variant_task_id": variant.task_id,
            "payload_id": variant.task_id.split("__")[2] if "__" in variant.task_id else "",
            "attack_goal": variant.attack_goal.value,
            "asr": hit,
            "asr_plant": result.asr_plant,
            "asr_trigger": result.asr_trigger,
        })
    result.asr_plant = saved_plant
    result.asr_trigger = saved_trigger
    result.asr = saved_asr
    return snapshots


async def run_baseline(
    task_filter: str | Sequence[str] | None = None,
    model: str | None = None,
    concurrency: int = 1,
) -> list[dict]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parents[1] / "results" / f"baseline_{timestamp}"
    tester = CarryoverTester(output_dir=output_dir, model=model)

    filters = _normalize_task_filters(task_filter)
    tasks = _select_tasks(filters)
    logger.info(f"Running {len(tasks)} clean baseline trials")
    logger.info(f"Tasks: {[t.task_id for t in tasks]}")
    logger.info(f"Available baseline tasks: {_baseline_task_ids()}")
    logger.info(f"Concurrency: {max(1, concurrency)}")

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _run_one(idx: int, task) -> dict:
        async with sem:
            logger.info(f"\n>>> [{idx}/{len(tasks)}] {task.task_id}")
            result = await tester.run_baseline_trial(task)

            # 1-Trig false-positive ASR: judge benign session evidence against
            # every INSTANT variant of this base task. Workspace mirrors the
            # path used inside run_baseline_trial.
            case_workspace = tester.workspace_root / f"{task.task_id}_clean"
            per_variant_asr = _judge_baseline_false_positive(task, result, case_workspace)
            fp_asr = any(v["asr"] for v in per_variant_asr)
            fp_plant = any(v["asr_plant"] for v in per_variant_asr)
            fp_trigger = any(v["asr_trigger"] for v in per_variant_asr)
            hit_variants = [v["variant_task_id"] for v in per_variant_asr if v["asr"]]
            if hit_variants:
                logger.warning(
                    f"  FP-ASR triggered on benign run of {task.task_id} by "
                    f"variants: {hit_variants}"
                )
            return {
                "trial_index": idx,
                "trial_id": result.trial_id,
                "task_id": task.task_id,
                "task_name": task.name,
                "domain": task.domain,
                "utility": result.utility,
                "utility_breakdown": result.utility_breakdown,
                "asr": fp_asr,
                "asr_plant": fp_plant,
                "asr_trigger": fp_trigger,
                "per_variant_asr": per_variant_asr,
                "judge_notes": result.judge_notes,
                "evidence": tester._build_evidence_snapshot(result),
            }

    all_results = await asyncio.gather(
        *[_run_one(idx, task) for idx, task in enumerate(tasks, start=1)]
    )
    all_results = sorted(all_results, key=lambda r: (r.get("trial_index", 0), r.get("task_id", "")))

    variant_weights = _attack_variant_weights()
    _print_baseline_summary(
        all_results,
        model=tester.config.agents.defaults.model,
        variant_weights=variant_weights,
    )

    n = len(all_results) or 1
    fp_count = sum(1 for r in all_results if r.get("asr"))
    weighted_num = sum(
        float(r.get("utility") or 0.0) * variant_weights.get(r["task_id"], 0)
        for r in all_results
    )
    weighted_den = sum(variant_weights.get(r["task_id"], 0) for r in all_results)
    report_path = output_dir / "baseline_report.json"
    report_path.write_text(json.dumps({
        "experiment": "carryover_baseline_v1",
        "timestamp": timestamp,
        "num_tasks": len(tasks),
        "num_trials": len(all_results),
        "mean_utility": (
            sum(float(r.get("utility") or 0.0) for r in all_results) / len(all_results)
            if all_results else 0.0
        ),
        "weighted_mean_utility": (weighted_num / weighted_den) if weighted_den else 0.0,
        "weighted_mean_utility_denominator": weighted_den,
        "variant_weights": variant_weights,
        "fp_asr": fp_count / n if all_results else 0.0,
        "fp_asr_count": f"{fp_count}/{len(all_results)}",
        "trials": all_results,
    }, indent=2, ensure_ascii=False))
    logger.info(f"\nBaseline report saved: {report_path}")
    return all_results


def _print_baseline_summary(
    results: list[dict],
    model: str = "",
    variant_weights: dict[str, int] | None = None,
) -> None:
    import sys
    out = sys.stdout
    weights = variant_weights or {}
    W = 96
    lines = ["", "=" * W, "CARRYOVER BENCH — BASELINE RESULTS (no attack)"]
    if model:
        lines.append(f"Model: {model}")
    lines.append("=" * W)
    lines.append(
        f"\n{'Benign Task':<32}  {'Utility':>7}  {'W':>3}  {'FP-ASR':>7}  {'Variants':>9}  "
        f"Triggered variants (1-Trig)"
    )
    lines.append("─" * W)

    for r in results:
        per_var = r.get("per_variant_asr") or []
        n_var = len(per_var)
        hit = [v["payload_id"] or v["variant_task_id"] for v in per_var if v["asr"]]
        fp_flag = "FP!" if r.get("asr") else "ok"
        hit_str = ", ".join(hit) if hit else "—"
        w = weights.get(r["task_id"], 0)
        lines.append(
            f"  {r['task_name']:<30}  {float(r.get('utility') or 0.0):>7.2f}  "
            f"{w:>3}  {fp_flag:>7}  {n_var:>9}  {hit_str}"
        )

    n = len(results) or 1
    mean_utility = sum(float(r.get("utility") or 0.0) for r in results) / n if results else 0.0
    weighted_num = sum(
        float(r.get("utility") or 0.0) * weights.get(r["task_id"], 0) for r in results
    )
    weighted_den = sum(weights.get(r["task_id"], 0) for r in results)
    weighted_mean = (weighted_num / weighted_den) if weighted_den else 0.0
    fp_count = sum(1 for r in results if r.get("asr"))
    fp_rate = fp_count / n if results else 0.0
    lines.append("─" * W)
    lines.append(
        f"  {'TOTAL (simple, n=' + str(n) + ')':<30}  {mean_utility:>7.2f}  "
        f"{'':>3}  {fp_rate:>6.1%}  {'':>9}  ({fp_count}/{len(results)} trials flagged)"
    )
    lines.append(
        f"  {'TOTAL (weighted, n=' + str(weighted_den) + ')':<30}  {weighted_mean:>7.2f}  "
        f"{'':>3}  {'':>7}  {'':>9}  same denominator as main benchmark"
    )
    lines.append("=" * W)
    print("\n".join(lines), file=out, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CarryoverBench benign baseline runner")
    parser.add_argument("--task", action="append", dest="task", default=None, metavar="TASK_ID")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Maximum number of independent baseline trials to run in parallel (default: 1)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>",
        level="DEBUG" if args.verbose else "INFO",
    )

    task_filter = args.task
    if task_filter is None and DEFAULT_TASK_FILTERS:
        task_filter = list(DEFAULT_TASK_FILTERS)

    asyncio.run(run_baseline(
        task_filter=task_filter,
        model=args.model or DEFAULT_MODEL,
        concurrency=max(1, args.concurrency),
    ))


if __name__ == "__main__":
    main()
