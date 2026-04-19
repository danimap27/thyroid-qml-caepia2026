"""
analyze_ablation.py — Ablation study analysis for thyroid-qml CAEPIA experiments.

Loads all results CSVs, aggregates across seeds, and generates:
  - Summary table (mean ± std per model × subset × backend)
  - Boxplots: accuracy/F1/MCC by model, by subset, by backend
  - Heatmaps: metric × model × subset
  - Noise impact plot: ideal vs noisy per model
  - Backend comparison: qiskit vs cudaq per model
  - Seed stability plot: std of accuracy across seeds

Usage:
    python core/analyze_ablation.py                   # uses config.yaml in cwd
    python core/analyze_ablation.py --config my.yaml
    python core/analyze_ablation.py --metric roc_auc
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))  # core/
from plot_utils import plot_ablation_boxplot, plot_heatmap


METRICS = ["accuracy", "f1_macro", "mcc", "balanced_accuracy", "roc_auc",
           "sensitivity", "specificity", "kappa", "training_time"]


def load_results(results_dir: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, "*", "results.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results CSVs found in {results_dir}")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["status"] == "ok"].copy()
    print(f"  Loaded {len(df)} successful runs from {len(files)} files.")
    return df


def summary_table(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["n_seeds"] = len(grp)
        for m in metrics:
            if m not in grp.columns:
                continue
            vals = grp[m].dropna()
            if len(vals) == 0:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
            else:
                row[f"{m}_mean"] = float(vals.mean())
                row[f"{m}_std"] = float(vals.std())
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df_summary: pd.DataFrame, primary_metric: str):
    mean_col = f"{primary_metric}_mean"
    std_col = f"{primary_metric}_std"
    if mean_col not in df_summary.columns:
        print(f"  [WARN] {primary_metric} not in results.")
        return
    df_s = df_summary.sort_values(mean_col, ascending=False)
    id_cols = [c for c in df_summary.columns if not c.endswith(("_mean", "_std")) and c != "n_seeds"]
    print(f"\n  {'  |  '.join(id_cols):<50}  {primary_metric} (mean ± std)   n")
    print("  " + "-" * 75)
    for _, row in df_s.iterrows():
        key = "  |  ".join(str(row[c]) for c in id_cols)
        m = row.get(mean_col, float("nan"))
        s = row.get(std_col, float("nan"))
        n = int(row.get("n_seeds", 0))
        print(f"  {key:<50}  {m:.4f} ± {s:.4f}   ({n} seeds)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--metric", default="accuracy",
                    help="Primary metric for summary table")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for figures (default: paper/figures/)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg.get("output_dir", "./results")
    out_dir = Path(args.out_dir or os.path.join(
        os.path.dirname(args.config) or ".", "paper", "figures"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[ABLATION] Loading results from {results_dir}")
    print(f"[ABLATION] Saving figures to {out_dir}")

    df = load_results(results_dir)

    available_metrics = [m for m in METRICS if m in df.columns]
    primary = args.metric if args.metric in df.columns else "accuracy"

    # ── 1. Full summary table ─────────────────────────────────────────────────
    group_cols = ["backend", "model", "subset"]
    group_cols = [c for c in group_cols if c in df.columns]
    summary = summary_table(df, group_cols, available_metrics)
    csv_out = out_dir / "ablation_summary.csv"
    summary.to_csv(csv_out, index=False)
    print(f"\n[OK] Summary CSV → {csv_out}")
    print_summary(summary, primary)

    # ── 2. Boxplots by model ──────────────────────────────────────────────────
    for metric in [m for m in ["accuracy", "f1_macro", "mcc", "roc_auc"] if m in df.columns]:
        path = out_dir / f"boxplot_by_model_{metric}.png"
        hue = "backend" if "backend" in df.columns and df["backend"].nunique() > 1 else None
        plot_ablation_boxplot(
            df, metric=metric, group_col="model", path=path,
            title=f"{metric} by model (all subsets, all seeds)",
            hue_col=hue,
        )
        print(f"[OK] {path.name}")

    # ── 3. Boxplots by subset ─────────────────────────────────────────────────
    if "subset" in df.columns:
        for metric in [m for m in ["accuracy", "f1_macro"] if m in df.columns]:
            path = out_dir / f"boxplot_by_subset_{metric}.png"
            hue = "backend" if "backend" in df.columns and df["backend"].nunique() > 1 else None
            plot_ablation_boxplot(
                df, metric=metric, group_col="subset", path=path,
                title=f"{metric} by feature subset",
                hue_col=hue,
            )
            print(f"[OK] {path.name}")

    # ── 4. Backend comparison ─────────────────────────────────────────────────
    if "backend" in df.columns and df["backend"].nunique() > 1:
        for metric in [m for m in ["accuracy", "f1_macro", "training_time"] if m in df.columns]:
            path = out_dir / f"boxplot_by_backend_{metric}.png"
            plot_ablation_boxplot(
                df, metric=metric, group_col="backend", path=path,
                title=f"{metric}: qiskit vs cudaq",
                hue_col="model",
            )
            print(f"[OK] {path.name}")

    # ── 5. Noise ablation: ideal vs noisy ─────────────────────────────────────
    if "noise" in df.columns and df["noise"].nunique() > 1:
        for metric in [m for m in ["accuracy", "f1_macro", "mcc"] if m in df.columns]:
            path = out_dir / f"boxplot_noise_ablation_{metric}.png"
            plot_ablation_boxplot(
                df, metric=metric, group_col="noise", path=path,
                title=f"Noise ablation — {metric}",
                hue_col="subset",
            )
            print(f"[OK] {path.name}")

    # ── 6. Heatmaps: model × subset ──────────────────────────────────────────
    if "model" in df.columns and "subset" in df.columns:
        for backend in (df["backend"].unique() if "backend" in df.columns else [None]):
            sub = df[df["backend"] == backend] if backend is not None else df
            for metric in [m for m in ["accuracy", "mcc", "roc_auc"] if m in sub.columns]:
                tag = f"_{backend}" if backend else ""
                path = out_dir / f"heatmap_model_subset_{metric}{tag}.png"
                try:
                    plot_heatmap(sub, row_col="model", col_col="subset",
                                 metric=metric, path=path,
                                 title=f"{metric} mean — {backend or 'all'}")
                    print(f"[OK] {path.name}")
                except Exception as e:
                    print(f"  [WARN] heatmap failed: {e}")

    # ── 7. Seed stability: std across seeds per model ─────────────────────────
    for metric in [m for m in ["accuracy", "f1_macro"] if m in df.columns]:
        stab_cols = [c for c in ["model", "backend"] if c in df.columns]
        if not stab_cols:
            continue
        stab = (df.groupby(stab_cols)[metric]
                  .std()
                  .reset_index()
                  .rename(columns={metric: f"{metric}_std"}))
        path = out_dir / f"seed_stability_{metric}.png"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(6, len(stab) * 0.8), 4))
        x = range(len(stab))
        ax.bar(x, stab[f"{metric}_std"], color="steelblue", alpha=0.75)
        labels = [" / ".join(str(stab.iloc[i][c]) for c in stab_cols) for i in range(len(stab))]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"Std of {metric} across seeds", fontsize=9)
        ax.set_title(f"Seed stability — {metric}", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {path.name}")

    print(f"\n[DONE] All figures saved to {out_dir}\n")


if __name__ == "__main__":
    main()
