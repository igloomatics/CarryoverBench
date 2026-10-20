"""
Task-level cluster bootstrap 95% CI for Table 1 (main benchmark results).

Why
---
Table 1 reports aggregate ASR per channel + Utility per model. These
metrics are computed by combining many correlated observations:
  * Autonomous / Latent ASR aggregates 5 rounds of the same planted state
    per variant -- highly correlated within variant.
  * Overall ASR aggregates across 35 + 30 + 36 = 101 tasks spanning 12
    task families -- correlated within family.

A naive Wilson binomial CI assumes i.i.d. Bernoulli draws and will
underestimate uncertainty by ~2x. The right tool is a cluster bootstrap
that resamples at the attack-task level, preserving the within-task
correlation structure across rounds.

What it does
------------
1. Ingest:    one row per (model, task_id) with asr / utility / any-round-fire
              fields pulled from each trial JSON.
2. Bootstrap: for each model, resample 101 task_ids with replacement, B times
              (default B=10,000), recompute all 5 Table-1 metrics each draw.
3. CI:        percentile 95% CI = (2.5%, 97.5%) of the bootstrap distribution.
4. Output:    CSV + LaTeX table ready to drop into the paper.

Usage
-----
    python table1_bootstrap.py \
        --results-dir ../results \
        --out-dir     ../results/_table1_boot \
        --n-boot      10000
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Baseline (No-Atk) dir -> model mapping
# ---------------------------------------------------------------------------
# Matched by weighted_mean_utility against Table 1's No-Atk column:
#   Gemini = 100.0%, Qwen = 92.4%, Claude = 89.3%, GPT = GLM = 97.2%.
# GPT and GLM are interchangeable (identical per-task utility distributions),
# so the assignment of the two 97.2% dirs is arbitrary.
DEFAULT_BASELINE_MAPPING: dict[str, str] = {
    "GPT-5.4":           "baseline_20260516_195440",
    "Claude-Sonnet-4.6": "baseline_20260516_215005",
    "Gemini-3.1-Pro":    "baseline_20260516_195927",
    "GLM-5":             "baseline_20260516_195747",
    "Qwen3-235B-A22B":   "baseline_20260516_195633",
}

# Reuse ingestion helpers from the sibling script
sys.path.insert(0, str(Path(__file__).parent))
from statistical_analysis import (  # noqa: E402
    _canonical_model,
    _channel_from_task_id,
    _task_family_from_task_id,
    _iter_trial_jsons,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Ingestion: one row per (model, task_id)
# ---------------------------------------------------------------------------
def ingest_task_level(results_dir: Path,
                      include: list[str] | None = None,
                      exclude: list[str] | None = None) -> pd.DataFrame:
    """Aggregate each trial JSON into ONE row per (model, task_id).

    Unlike statistical_analysis.ingest_results which produces per-round rows
    and skips Instant, this keeps Instant and collapses rounds to a single
    'any_round_fire' indicator -- matching the granularity at which Table 1
    aggregates its metrics.
    """
    rows: list[dict] = []
    skipped_model: set[str] = set()
    skipped_channel: set[str] = set()

    for parent_dir, trial in _iter_trial_jsons(results_dir, include, exclude):
        model = (_canonical_model(parent_dir)
                 or _canonical_model(trial.get("model", "")))
        if model is None:
            skipped_model.add(parent_dir)
            continue

        task_id = trial.get("task_id") or trial.get("attack_id") or ""
        channel = _channel_from_task_id(task_id)
        if channel is None:
            skipped_channel.add(task_id)
            continue

        family = _task_family_from_task_id(task_id)
        per_round = trial.get("trig_per_round") or []
        any_round_fire = bool(any(per_round)) if per_round else False

        rows.append({
            "model": model,
            "task_id": task_id,
            "task_family": family,
            "channel": channel,
            "asr": int(bool(trial.get("asr", False))),
            "asr_plant": int(bool(trial.get("asr_plant", False))),
            "asr_trigger": int(bool(trial.get("asr_trigger", False))),
            "any_round_fire": int(any_round_fire),
            "utility": float(trial.get("utility", np.nan)),
            "n_rounds": len(per_round),
        })

    if skipped_model:
        print(f"[ingest] skipped {len(skipped_model)} dirs (unknown model)",
              file=sys.stderr)
    if skipped_channel:
        print(f"[ingest] skipped {len(skipped_channel)} task_ids "
              f"(unknown channel)", file=sys.stderr)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No trials ingested. Check --results-dir / --include.")
    # If the same (model, task_id) appears in multiple result dirs (e.g. two
    # runs of the same model), keep the first occurrence.
    df = df.drop_duplicates(subset=["model", "task_id"], keep="first")
    return df


# ---------------------------------------------------------------------------
# Ingestion: baseline (No-Atk) trials
# ---------------------------------------------------------------------------
def ingest_baseline(results_dir: Path,
                    mapping: dict[str, str]) -> pd.DataFrame:
    """Load per-task utility from each model's baseline dir.

    Each baseline dir contains 12 benign-task JSONs (one per task family) plus
    a baseline_report.json with weighted_mean_utility. We extract the
    per-trial utility values to enable bootstrap resampling.
    """
    rows: list[dict] = []
    for model, dirname in mapping.items():
        bdir = results_dir / dirname
        if not bdir.exists():
            print(f"[baseline] WARN: {bdir} missing -- skipping {model}",
                  file=sys.stderr)
            continue
        # Prefer baseline_report.json trials (canonical) but fall back to
        # the per-task *_clean.json files if needed.
        report = bdir / "baseline_report.json"
        if report.exists():
            data = json.loads(report.read_text())
            # variant_weights: family -> number of attack variants in that family.
            # Used to compute a *weighted* No-Atk mean that is apples-to-apples
            # with Atk (which is naturally weighted by per-family variant count).
            weights = data.get("variant_weights", {})
            for t in data.get("trials", []):
                task_id = t.get("task_id") or t.get("domain") or ""
                rows.append({
                    "model": model,
                    "task_id": task_id,
                    "domain": t.get("domain", ""),
                    "utility": float(t.get("utility", np.nan)),
                    "weight": float(weights.get(task_id, 1)),
                    "source_dir": dirname,
                })
        else:
            for jf in bdir.glob("*_clean.json"):
                if jf.name.endswith(".case.json"):
                    continue
                try:
                    t = json.loads(jf.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                rows.append({
                    "model": model,
                    "task_id": t.get("task_id", jf.stem),
                    "domain": "",
                    "utility": float(t.get("utility", np.nan)),
                    "weight": 1.0,
                    "source_dir": dirname,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        print("[baseline] WARN: no baseline trials loaded",
              file=sys.stderr)
    return df


# ---------------------------------------------------------------------------
# Metric computation (matches Table 1 semantics)
# ---------------------------------------------------------------------------
def _attack_success(row: pd.Series) -> int:
    """Per-task success indicator. Channel-specific definition.

    * Instant: the single trial's `asr` field.
    * Autonomous / Latent: any of the 5 follow-up rounds fired.
    """
    if row.channel == "Instant":
        return int(row.asr)
    return int(row.any_round_fire)


def compute_table1_row(df_model: pd.DataFrame,
                       baseline_df: pd.DataFrame | None = None) -> dict:
    """Compute Table 1 metrics for one model.

    df_model:      attack-trial task-level dataframe for this model.
    baseline_df:   benign-trial dataframe for this model (optional). If given,
                   also reports Utility_NoAtk and Delta_Utility.

    Returns dict mapping metric_name -> value (in percentage points).
    """
    df = df_model.copy()
    df["success"] = df.apply(_attack_success, axis=1)

    metrics: dict[str, float] = {}
    for ch in ("Instant", "Autonomous", "Latent"):
        sub = df[df.channel == ch]
        metrics[f"ASR_{ch}"] = (sub.success.mean() * 100
                                if len(sub) else np.nan)
    metrics["ASR_Overall"] = df.success.mean() * 100 if len(df) else np.nan

    util_atk = df.utility.dropna()
    util_atk_pct = util_atk.mean() * 100 if len(util_atk) else np.nan
    metrics["Utility_Atk"] = util_atk_pct

    if baseline_df is not None and len(baseline_df):
        bsub = baseline_df.dropna(subset=["utility"])
        if len(bsub) == 0:
            util_noatk_pct = np.nan
        elif "weight" in bsub.columns and bsub.weight.sum() > 0:
            # Variant-count weighted mean -- matches paper's
            # weighted_mean_utility for apples-to-apples comparison with Atk.
            util_noatk_pct = ((bsub.utility * bsub.weight).sum()
                              / bsub.weight.sum() * 100)
        else:
            util_noatk_pct = bsub.utility.mean() * 100
        metrics["Utility_NoAtk"] = util_noatk_pct
        if not (np.isnan(util_atk_pct) or np.isnan(util_noatk_pct)):
            metrics["Delta_Utility"] = util_atk_pct - util_noatk_pct
        else:
            metrics["Delta_Utility"] = np.nan

    return metrics


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------
def cluster_bootstrap_one_model(df_model: pd.DataFrame,
                                baseline_df: pd.DataFrame | None = None,
                                n_boot: int = 10_000,
                                seed: int = 42) -> dict[str, tuple[float, float, float]]:
    """Task-level bootstrap for one model.

    Resamples *attack* tasks and *baseline* tasks independently per iteration
    (Δ Utility's CI thus correctly reflects both sources of variability).

    Returns metric -> (point, lo, hi).
    """
    rng = np.random.default_rng(seed)

    # Pre-index attack tasks
    atk_tasks = df_model.task_id.unique()
    n_atk = len(atk_tasks)
    atk_indexed = {t: df_model[df_model.task_id == t] for t in atk_tasks}

    # Pre-index baseline tasks (if any)
    use_baseline = baseline_df is not None and len(baseline_df) > 0
    if use_baseline:
        # Pull only this model's baseline rows (caller should pre-filter, but defensive)
        bdf = baseline_df.reset_index(drop=True)
        bln_idx = bdf.index.to_numpy()
        n_bln = len(bln_idx)
    else:
        bdf = None
        n_bln = 0

    # Point estimates
    point = compute_table1_row(df_model, baseline_df=bdf)
    metric_names = list(point.keys())

    boot: dict[str, list[float]] = {m: [] for m in metric_names}
    for _ in range(n_boot):
        sampled_atk = rng.choice(atk_tasks, size=n_atk, replace=True)
        boot_atk = pd.concat([atk_indexed[t] for t in sampled_atk],
                             ignore_index=True)

        if use_baseline:
            sampled_bln = rng.choice(bln_idx, size=n_bln, replace=True)
            boot_bln = bdf.iloc[sampled_bln]
        else:
            boot_bln = None

        m_vals = compute_table1_row(boot_atk, baseline_df=boot_bln)
        for m in metric_names:
            boot[m].append(m_vals.get(m, np.nan))

    out: dict[str, tuple[float, float, float]] = {}
    for m in metric_names:
        arr = np.asarray(boot[m], dtype=float)
        clean = arr[~np.isnan(arr)]
        if len(clean) == 0:
            out[m] = (point[m], np.nan, np.nan)
        else:
            lo, hi = np.percentile(clean, [2.5, 97.5])
            out[m] = (point[m], lo, hi)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _format_cell(point: float, lo: float, hi: float) -> str:
    if np.isnan(point):
        return "—"
    if np.isnan(lo) or np.isnan(hi):
        return f"{point:.1f}"
    return f"{point:.1f} [{lo:.1f}, {hi:.1f}]"


def to_long_csv(results: dict, n_tasks_per_model: dict) -> pd.DataFrame:
    rows = []
    for model, metrics in results.items():
        for metric, (point, lo, hi) in metrics.items():
            rows.append({
                "model": model,
                "metric": metric,
                "point": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_width": hi - lo if not np.isnan(hi) else np.nan,
                "n_tasks": n_tasks_per_model[model],
                "display": _format_cell(point, lo, hi),
            })
    return pd.DataFrame(rows)


def to_pivot(long_df: pd.DataFrame) -> pd.DataFrame:
    order = ["ASR_Instant", "ASR_Autonomous", "ASR_Latent",
             "ASR_Overall", "Utility_NoAtk", "Utility_Atk", "Delta_Utility"]
    pivot = long_df.pivot(index="model", columns="metric", values="display")
    return pivot[[c for c in order if c in pivot.columns]]


def to_latex(long_df: pd.DataFrame, label: str = "tab:table1_bootstrap") -> str:
    pivot = to_pivot(long_df)
    header = (r"\begin{table*}[t]" "\n" r"\centering" "\n" r"\small" "\n"
              r"\setlength{\tabcolsep}{4pt}" "\n"
              r"\begin{tabular}{l" + "c" * len(pivot.columns) + "}\n"
              r"\toprule" "\n")
    col_label = {
        "ASR_Instant":    r"\textbf{Instant}",
        "ASR_Autonomous": r"\textbf{Autonomous}",
        "ASR_Latent":     r"\textbf{Latent}",
        "ASR_Overall":    r"\textbf{Overall ASR}",
        "Utility_NoAtk":  r"\textbf{No-Atk}",
        "Utility_Atk":    r"\textbf{Atk}",
        "Delta_Utility":  r"$\boldsymbol{\Delta}$",
    }
    headers = " & ".join([r"\textbf{Model}"] +
                         [col_label.get(c, c) for c in pivot.columns])
    body_rows = []
    for model in pivot.index:
        cells = [model] + [pivot.loc[model, c] for c in pivot.columns]
        body_rows.append(" & ".join(cells) + r" \\")
    table = (header + headers + r" \\" + "\n" +
             r"\midrule" + "\n" +
             "\n".join(body_rows) + "\n" +
             r"\bottomrule" + "\n" +
             r"\end{tabular}" + "\n" +
             r"\caption{Table 1 metrics with task-level cluster bootstrap 95\% CI "
             r"in brackets (B=10{,}000 iterations resampling 101 attack tasks "
             r"with replacement). ASR for Autonomous and Latent uses "
             r"any-round-fire per variant; Instant ASR uses single-shot success. "
             r"Overall ASR weights tasks equally across all three channels.}" + "\n" +
             r"\label{" + label + r"}" + "\n" +
             r"\end{table*}" + "\n")
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Root containing benchmark_v8full_* result dirs.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--include", nargs="*",
                    default=["benchmark_v8full"],
                    help="Only scan sub-dirs whose name contains these.")
    ap.add_argument("--exclude", nargs="*",
                    default=["benchmark_v8full_gpt-5.4_20260516_120921",
                             "benchmark_v8full_gpt-5.4_20260516_123849"],
                    help="Skip these dirs (e.g. older gpt-5.4 reruns).")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip baseline (No-Atk) data; only bootstrap ASR + Utility(Atk).")
    ap.add_argument("--baseline-map", nargs="*", default=None,
                    metavar="MODEL=DIR",
                    help="Override baseline dir per model, e.g. "
                         "'GPT-5.4=baseline_20260516_195440'. "
                         "Unspecified models fall back to DEFAULT_BASELINE_MAPPING.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1a. Ingest attack-trial data
    df = ingest_task_level(args.results_dir, args.include, args.exclude)
    print(f"[ingest] {len(df)} task-level rows  | "
          f"models={sorted(df.model.unique())}  | "
          f"channels={sorted(df.channel.unique())}")
    df.to_csv(args.out_dir / "task_level.csv", index=False)

    # Sanity-print per-model task counts
    print("\nPer-model attack-task counts:")
    print(df.groupby(["model", "channel"]).size().unstack(fill_value=0))

    # 1b. Ingest baseline (No-Atk) data
    if args.no_baseline:
        baseline_df = pd.DataFrame()
        print("\n[baseline] skipped via --no-baseline")
    else:
        mapping = dict(DEFAULT_BASELINE_MAPPING)
        for pair in (args.baseline_map or []):
            if "=" not in pair:
                raise SystemExit(f"Bad --baseline-map entry: {pair!r}, "
                                 "expected MODEL=DIR")
            k, v = pair.split("=", 1)
            mapping[k.strip()] = v.strip()
        baseline_df = ingest_baseline(args.results_dir, mapping)
        print(f"\n[baseline] {len(baseline_df)} trials loaded across "
              f"{baseline_df.model.nunique() if len(baseline_df) else 0} models")
        if len(baseline_df):
            baseline_df.to_csv(args.out_dir / "baseline_level.csv", index=False)
            print("Per-model baseline weighted-mean utility "
                  "(should match paper Table 1 No-Atk column):")
            for m, sub in baseline_df.groupby("model"):
                wmean = (sub.utility * sub.weight).sum() / sub.weight.sum() * 100
                print(f"  {m:25s} {wmean:5.1f}%")

    # 2. Bootstrap each model
    results: dict[str, dict] = {}
    n_tasks_per_model: dict[str, int] = {}
    for model in sorted(df.model.unique()):
        df_m = df[df.model == model]
        bln_m = (baseline_df[baseline_df.model == model]
                 if len(baseline_df) else None)
        n_tasks_per_model[model] = len(df_m)
        n_bln = len(bln_m) if bln_m is not None else 0
        print(f"\n[bootstrap] {model}  "
              f"(n_atk={len(df_m)}, n_bln={n_bln}, B={args.n_boot}) ...",
              end=" ", flush=True)
        results[model] = cluster_bootstrap_one_model(
            df_m, baseline_df=bln_m,
            n_boot=args.n_boot, seed=args.seed,
        )
        print("done")

    # 3. Format and save
    long_df = to_long_csv(results, n_tasks_per_model)
    long_df.to_csv(args.out_dir / "table1_bootstrap_long.csv", index=False)

    pivot = to_pivot(long_df)
    pivot.to_csv(args.out_dir / "table1_bootstrap_pivot.csv")

    latex = to_latex(long_df)
    (args.out_dir / "table1_bootstrap.tex").write_text(latex)

    # 4. Print human-readable summary
    print("\n" + "=" * 88)
    print("Table 1 metrics with task-level bootstrap 95% CI "
          f"(B={args.n_boot:,})")
    print("=" * 88)
    print(pivot.to_string())
    print("\n[done] artifacts written to:", args.out_dir)
    print("  - task_level.csv             (raw aggregated data)")
    print("  - table1_bootstrap_long.csv  (one row per (model, metric))")
    print("  - table1_bootstrap_pivot.csv (Table 1 layout)")
    print("  - table1_bootstrap.tex       (drop-in LaTeX)")


if __name__ == "__main__":
    main()
