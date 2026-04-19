"""
generate_tables.py — Generic LaTeX table generator.

Reads results from CSV files in output_dir and generates tables
according to the structure defined in config.yaml.

Labels, metrics, and grouping are fully driven by configuration.
No paper-specific code lives here.

Usage:
    python core/generate_tables.py
    python core/generate_tables.py --config config.yaml --out-dir paper/tables
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_results(results_dir: str, file_pattern: str = "*.csv") -> pd.DataFrame:
    pattern = os.path.join(results_dir, "*", file_pattern)
    dfs = []
    for f in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"  [WARN] {f}: {e}")
    if not dfs:
        print(f"[ERROR] No CSV files in {results_dir}/*/{file_pattern}")
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    id_col = next((c for c in ["run_id", "id"] if c in all_df.columns), None)
    if id_col:
        all_df = all_df.sort_values("timestamp", na_position="last").drop_duplicates(
            subset=[id_col], keep="last"
        )
    print(f"[OK] Loaded {len(all_df)} runs from {len(dfs)} files.")
    return all_df


def fmt(series: pd.Series, pct: bool = False, decimals: int = 2) -> str:
    if series.empty or series.isna().all():
        return "---"
    m = series.mean()
    s = series.std(ddof=1) if len(series) > 1 else 0.0
    scale = 100.0 if pct else 1.0
    d = decimals if not pct else 1
    return f"${m*scale:.{d}f} \\pm {s*scale:.{d}f}$"


def write_tex(path: str, content: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [OK] {path}")


def make_table(df: pd.DataFrame, table_cfg: dict, cfg: dict, out_dir: str):
    """
    Generate one LaTeX table from a table definition block in config.yaml.

    table_cfg keys:
      name        : output filename (without .tex)
      caption     : LaTeX caption string
      label       : LaTeX label
      rows        : column name to iterate over as row groups
      cols        : column name to iterate over as columns (optional)
      metrics     : list of {column, label, pct} dicts
      filter      : dict of {column: value} to pre-filter the dataframe
    """
    labels = cfg.get("labels", {})

    # Apply filter
    sub = df.copy()
    for col, val in (table_cfg.get("filter") or {}).items():
        if col in sub.columns:
            sub = sub[sub[col] == val]

    if sub.empty:
        print(f"  [WARN] No data for table '{table_cfg['name']}' after filtering.")
        return

    row_key = table_cfg.get("rows")
    col_key = table_cfg.get("cols")
    metrics = table_cfg.get("metrics", [])

    row_values = list(dict.fromkeys(sub[row_key].dropna())) if row_key else [None]
    col_values = list(dict.fromkeys(sub[col_key].dropna())) if col_key else [None]

    row_labels_map = labels.get(row_key, {}) if row_key else {}
    col_labels_map = labels.get(col_key, {}) if col_key else {}

    # Build header
    n_metric_cols = len(metrics) * (len(col_values) if col_key else 1)
    col_spec = ("l" if row_key else "") + "c" * n_metric_cols

    header_parts = []
    if row_key:
        header_parts.append(f"\\textbf{{{row_key.replace('_',' ').title()}}}")
    if col_key and len(col_values) > 1:
        for cv in col_values:
            cv_label = col_labels_map.get(str(cv), str(cv))
            span = len(metrics)
            header_parts.append(f"\\multicolumn{{{span}}}{{c}}{{{cv_label}}}")
    else:
        for m in metrics:
            header_parts.append(m.get("label", m["column"]))

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{table_cfg.get('caption', '')}}}",
        f"\\label{{{table_cfg.get('label', 'tab:' + table_cfg['name'])}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_parts) + r" \\",
    ]

    # Sub-header for metric names when there are multiple col groups
    if col_key and len(col_values) > 1:
        sub_parts = [""] if row_key else []
        for _ in col_values:
            for m in metrics:
                sub_parts.append(m.get("label", m["column"]))
        lines.append(" & ".join(sub_parts) + r" \\")

    lines.append(r"\midrule")

    # Body
    for rv in row_values:
        rv_df = sub[sub[row_key] == rv] if row_key else sub
        rv_label = row_labels_map.get(str(rv), str(rv)) if rv is not None else ""
        row_parts = [rv_label] if row_key else []

        for cv in col_values:
            cell_df = rv_df[rv_df[col_key] == cv] if col_key else rv_df
            for m in metrics:
                col_name = m["column"]
                if col_name not in cell_df.columns:
                    row_parts.append("---")
                    continue
                row_parts.append(fmt(cell_df[col_name], pct=m.get("pct", False)))

        lines.append(" & ".join(row_parts) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex(
        os.path.join(out_dir, table_cfg["name"] + ".tex"),
        "\n".join(lines) + "\n",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg.get("output_dir", "./results")
    out_dir = args.out_dir or cfg.get("tables_dir", "./paper/tables")
    file_pattern = cfg.get("results", {}).get("file_pattern", "*.csv")

    print(f"\n{'='*60}")
    print(f"  Table Generator  |  Results: {results_dir}  |  Out: {out_dir}")
    print(f"{'='*60}\n")

    df = load_results(results_dir, file_pattern)

    table_defs = cfg.get("tables", [])
    if not table_defs:
        print("[INFO] No 'tables:' section found in config.yaml. Nothing to generate.")
        return

    for t in table_defs:
        print(f"Generating: {t['name']}...")
        if df.empty and t.get("static"):
            # Static tables (no results needed) can define their own content
            write_tex(os.path.join(out_dir, t["name"] + ".tex"), t.get("content", ""))
        elif not df.empty:
            make_table(df, t, cfg, out_dir)
        else:
            print(f"  [SKIP] No results loaded yet.")

    # Master include file
    files = glob.glob(os.path.join(out_dir, "*.tex"))
    master = "\n".join(
        r"\input{" + os.path.relpath(f, start=os.path.dirname(out_dir)).replace("\\", "/") + "}"
        for f in sorted(files)
        if "tables.tex" not in f
    )
    write_tex(os.path.join(out_dir, "tables.tex"), "% AUTO-GENERATED\n" + master + "\n")
    print(f"\n[SUCCESS] Tables in {out_dir}/")


if __name__ == "__main__":
    main()
