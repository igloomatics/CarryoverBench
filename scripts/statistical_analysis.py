"""
Statistical analysis for CarryoverBench: formally test the "no decay" claim.

Pipeline
--------
1. Ingest:  Walk a results directory, load every trial JSON, and flatten
            `trig_per_round` into a long-format dataframe with columns:
            (model, variant_id, channel, task_family, round, trigger).

2. GEE:     Per (model, channel), fit a Generalized Estimating Equations
            logistic model `trigger ~ round` with variant_id as the
            correlation cluster. Output: β, 95% CI, p-value.
            CI including 0 supports "no monotonic decay".

3. M-K:     Per (model, channel), run Mann-Kendall trend test on the
            per-round mean trigger rate. Easier to communicate in prose.

4. Wilson:  Per (model, channel, round) cell, compute the Wilson 95%
            binomial CI -- drop-in replacement for the bare percentages
            in Table 2 / Figure 3.

Usage
-----
    # Ingest from a results dir tree and run all three analyses:
    python statistical_analysis.py --results-dir ../results \
                                   --out-dir ../results/_stats

    # If you already have a long-format CSV:
    python statistical_analysis.py --long-csv my_long.csv \
                                   --out-dir ../results/_stats

Output
------
    <out-dir>/long_data.csv           (the rectangular dataset)
    <out-dir>/gee_results.csv         (β, CI, p per model × channel)
    <out-dir>/mannkendall.csv         (tau, p per model × channel)
    <out-dir>/wilson_ci_per_cell.csv  (CI per model × channel × round)
    <out-dir>/summary.txt             (human-readable report)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CHANNEL_MAP = {
    "instant": "Instant",
    "delayed": "Autonomous",
    "persistent": "Latent",
}

MODEL_PATTERNS = [
    (re.compile(r"gpt-?5\.?4", re.I), "GPT-5.4"),
    (re.compile(r"claude.*sonnet.*4[-_.]?6", re.I), "Claude-Sonnet-4.6"),
    (re.compile(r"gemini.*3\.?1.*pro", re.I), "Gemini-3.1-Pro"),
    (re.compile(r"glm[-_]?5", re.I), "GLM-5"),
    (re.compile(r"qwen3?[-_]?235b", re.I), "Qwen3-235B-A22B"),
]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def _canonical_model(name: str) -> str | None:
    for pattern, canonical in MODEL_PATTERNS:
        if pattern.search(name):
            return canonical
    return None


def _channel_from_task_id(task_id: str) -> str | None:
    for key, val in CHANNEL_MAP.items():
        if f"__{key}__" in task_id or task_id.endswith(f"__{key}"):
            return val
    return None


def _task_family_from_task_id(task_id: str) -> str:
    return task_id.split("__")[0]


def _iter_trial_jsons(root: Path, include: list[str] | None = None,
                      exclude: list[str] | None = None) -> Iterable[tuple[str, dict]]:
    """Yield (parent_dir_name, parsed_json) for every trial-level JSON.

    If `include` is given, only directories whose name matches at least one
    of the substrings are scanned. `exclude` is the inverse filter.
    """
    inc = [s.lower() for s in (include or [])]
    exc = [s.lower() for s in (exclude or [])]
    for json_path in root.rglob("*.json"):
        name = json_path.name
        if name in {"full_report.json"} or name.endswith(".case.json"):
            continue
        parent_lc = json_path.parent.name.lower()
        if inc and not any(s in parent_lc for s in inc):
            continue
        if exc and any(s in parent_lc for s in exc):
            continue
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "trig_per_round" not in data:
            continue
        yield json_path.parent.name, data


def ingest_results(results_dir: Path,
                   include: list[str] | None = None,
                   exclude: list[str] | None = None) -> pd.DataFrame:
    """Flatten every trial JSON into a long-format DataFrame."""
    rows: list[dict] = []
    skipped_model: set[str] = set()
    skipped_channel: set[str] = set()

    for parent_dir, trial in _iter_trial_jsons(results_dir, include, exclude):
        model = _canonical_model(parent_dir) or _canonical_model(trial.get("model", ""))
        if model is None:
            skipped_model.add(parent_dir)
            continue

        task_id = trial.get("task_id") or trial.get("attack_id") or ""
        channel = _channel_from_task_id(task_id)
        if channel is None:
            skipped_channel.add(task_id)
            continue
        if channel == "Instant":
            # Instant has no temporal axis; not relevant to the no-decay claim.
            continue

        family = _task_family_from_task_id(task_id)
        # Use task_id (stable across re-runs), not trial_id (random per run),
        # so the same variant tested twice is de-duplicated downstream.
        variant_id = task_id or trial.get("trial_id")
        per_round = trial.get("trig_per_round", [])
        if not isinstance(per_round, list) or len(per_round) == 0:
            continue

        for k, fired in enumerate(per_round, start=1):
            rows.append({
                "model": model,
                "variant_id": variant_id,
                "task_id": task_id,
                "task_family": family,
                "channel": channel,
                "round": k,
                "trigger": int(bool(fired)),
            })

    if skipped_model:
        print(f"[ingest] skipped {len(skipped_model)} dirs (model unrecognized): "
              f"{sorted(skipped_model)[:3]}{'...' if len(skipped_model) > 3 else ''}",
              file=sys.stderr)
    if skipped_channel:
        print(f"[ingest] skipped {len(skipped_channel)} task_ids (channel unrecognized)",
              file=sys.stderr)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No trial data ingested. Check --results-dir.")
    df = df.drop_duplicates(subset=["model", "variant_id", "channel", "round"])
    return df


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def run_gee(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, channel): fit GEE logistic, return β & 95% CI for round."""
    import statsmodels.api as sm

    rows = []
    for (model, channel), sub in df.groupby(["model", "channel"]):
        n_variants = sub.variant_id.nunique()
        n_obs = len(sub)
        note = "ok"
        beta = ci_lo = ci_hi = p = np.nan

        if sub.trigger.nunique() < 2:
            note = "no variance (all 0 or all 1)"
        else:
            try:
                fit = sm.GEE.from_formula(
                    "trigger ~ round",
                    groups="variant_id",
                    data=sub,
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable(),
                ).fit()
                beta = fit.params["round"]
                ci_lo, ci_hi = fit.conf_int().loc["round"]
                p = fit.pvalues["round"]
            except Exception as exc:
                note = f"fit error: {exc}"

        verdict = ""
        if note == "ok":
            if pd.isna(ci_lo) or pd.isna(ci_hi):
                verdict = "no decay (CI degenerate; β≈0 across rounds)"
                note = "near-flat across rounds (CI undefined)"
            elif ci_lo <= 0 <= ci_hi:
                verdict = "no decay (CI crosses 0)"
            elif ci_hi < 0:
                verdict = "decay (β<0)"
            else:
                verdict = "growth (β>0)"

        rows.append({
            "model": model,
            "channel": channel,
            "n_variants": n_variants,
            "n_obs": n_obs,
            "beta": beta,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_value": p,
            "verdict": verdict,
            "note": note,
        })
    return pd.DataFrame(rows).sort_values(["channel", "model"]).reset_index(drop=True)


def run_mann_kendall(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, channel): Mann-Kendall trend test on per-round mean rate."""
    from scipy.stats import kendalltau

    rows = []
    for (model, channel), sub in df.groupby(["model", "channel"]):
        rates = sub.groupby("round").trigger.mean().sort_index()
        x = rates.index.to_numpy(dtype=float)
        y = rates.to_numpy(dtype=float)
        if len(y) < 3 or np.allclose(y, y[0]):
            tau, p = np.nan, np.nan
            note = "flat or insufficient rounds"
        else:
            tau, p = kendalltau(x, y)
            note = ""
        rows.append({
            "model": model,
            "channel": channel,
            "tau": tau,
            "p_value": p,
            "n_rounds": len(y),
            "rates": ",".join(f"{v:.3f}" for v in y),
            "note": note,
        })
    return pd.DataFrame(rows).sort_values(["channel", "model"]).reset_index(drop=True)


def run_wilson_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Wilson 95% binomial CI per (model, channel, round) cell."""
    from statsmodels.stats.proportion import proportion_confint

    rows = []
    for (model, channel, rnd), sub in df.groupby(["model", "channel", "round"]):
        k = int(sub.trigger.sum())
        n = int(len(sub))
        rate = k / n if n else np.nan
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson") if n else (np.nan, np.nan)
        rows.append({
            "model": model,
            "channel": channel,
            "round": rnd,
            "k": k,
            "n": n,
            "rate": rate,
            "ci_lo": lo,
            "ci_hi": hi,
            "display": f"{rate*100:5.1f} [{lo*100:4.1f}, {hi*100:4.1f}]" if n else "—",
        })
    return pd.DataFrame(rows).sort_values(["channel", "model", "round"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_gee(g: pd.DataFrame) -> str:
    lines = ["GEE: trigger ~ round, clustered by variant_id",
             "-" * 88,
             f"{'Model':<22}{'Channel':<12}{'n_var':>7}{'β':>8}{'95% CI':>22}{'p':>8}  verdict"]
    for _, r in g.iterrows():
        if r["note"] != "ok":
            lines.append(f"{r['model']:<22}{r['channel']:<12}{r['n_variants']:>7}    —    "
                         f"  ({r['note']})")
            continue
        lines.append(
            f"{r['model']:<22}{r['channel']:<12}{r['n_variants']:>7}"
            f"{r['beta']:>+8.3f}  [{r['ci_lo']:>+6.3f}, {r['ci_hi']:>+6.3f}]"
            f"{r['p_value']:>8.3f}  {r['verdict']}"
        )
    return "\n".join(lines)


def _fmt_mk(m: pd.DataFrame) -> str:
    lines = ["Mann-Kendall trend test on per-round trigger rate",
             "-" * 88,
             f"{'Model':<22}{'Channel':<12}{'τ':>10}{'p':>10}   per-round rates"]
    for _, r in m.iterrows():
        if pd.isna(r["tau"]):
            lines.append(f"{r['model']:<22}{r['channel']:<12}{'—':>10}{'—':>10}   ({r['note']})")
        else:
            lines.append(
                f"{r['model']:<22}{r['channel']:<12}{r['tau']:>+10.3f}{r['p_value']:>10.3f}   "
                f"[{r['rates']}]"
            )
    return "\n".join(lines)


def _fmt_wilson(w: pd.DataFrame) -> str:
    pivoted = w.pivot_table(index=["model", "channel"], columns="round",
                            values="display", aggfunc="first")
    lines = ["Wilson 95% CI per (model, channel, round)  -- rate [lo, hi] in %",
             "-" * 88,
             pivoted.to_string()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--results-dir", type=Path,
                     help="Root directory containing trial JSONs (recursively).")
    src.add_argument("--long-csv", type=Path,
                     help="Pre-built long-format CSV with columns: "
                          "model, variant_id, channel, round, trigger "
                          "[, task_family].")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory to write CSV + summary outputs.")
    ap.add_argument("--include", nargs="*", default=None,
                    help="(with --results-dir) Only scan sub-dirs whose name "
                         "contains any of these substrings, e.g. "
                         "'--include main_benchmark' to skip ablation runs.")
    ap.add_argument("--exclude", nargs="*",
                    default=["backup", "failed", "ablation", "supplement",
                             "probe", "temporal", "smoketest", "_stats"],
                    help="Sub-dirs containing any of these substrings are skipped "
                         "(default excludes ablation / supplement / backup dirs).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.long_csv:
        df = pd.read_csv(args.long_csv)
        print(f"[ingest] loaded {len(df)} rows from {args.long_csv}")
    else:
        print(f"[ingest] scanning {args.results_dir} ...")
        df = ingest_results(args.results_dir,
                            include=args.include, exclude=args.exclude)
        print(f"[ingest] {len(df)} rows  | "
              f"{df.model.nunique()} models  | "
              f"{df.variant_id.nunique()} variants  | "
              f"channels: {sorted(df.channel.unique())}")
        df.to_csv(args.out_dir / "long_data.csv", index=False)

    gee_df = run_gee(df)
    mk_df = run_mann_kendall(df)
    wilson_df = run_wilson_ci(df)

    gee_df.to_csv(args.out_dir / "gee_results.csv", index=False)
    mk_df.to_csv(args.out_dir / "mannkendall.csv", index=False)
    wilson_df.to_csv(args.out_dir / "wilson_ci_per_cell.csv", index=False)

    summary = "\n\n".join([
        "=" * 88,
        f"CarryoverBench statistical analysis  |  n_rows={len(df)}",
        "=" * 88,
        _fmt_gee(gee_df),
        _fmt_mk(mk_df),
        _fmt_wilson(wilson_df),
    ])
    print("\n" + summary)
    (args.out_dir / "summary.txt").write_text(summary)
    print(f"\n[done] artifacts written to {args.out_dir}/")


if __name__ == "__main__":
    main()
