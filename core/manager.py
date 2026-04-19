"""
manager.py — Generic Experiment HUB for Hercules HPC.

Reads all experiment configuration from config.yaml.
Compatible with any project that uses runner.py + slurm_generic.sh.

Features:
  [R]  Refresh command files from config.yaml
  [1-N] Submit individual phases as SLURM array jobs
  [F]  Submit all phases with sequential dependencies
  [M]  Live progress monitor — refreshes every 2s until keypress
  [C]  Check completed/pending runs before submitting
  [T]  Generate LaTeX tables
  [X]  Exit

Usage:
    python core/manager.py                      # uses config.yaml in cwd
    python core/manager.py --config my_cfg.yaml
"""

import glob
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import termios
    import tty
    import select
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def header(name: str):
    w = 70
    print("=" * w)
    print(f"{'HERCULES HUB — ' + name.upper():^{w}}")
    print("=" * w)


def run(cmd: str, capture: bool = False):
    try:
        if capture:
            r = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return r.stdout.strip()
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {e}")
        return None


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    for enc in ["utf-8-sig", "utf-16", "latin1"]:
        try:
            with open(path, encoding=enc) as f:
                return sum(1 for l in f if l.strip())
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 0


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _progress_bar(done: int, total: int, width: int = 36) -> str:
    if total == 0:
        return f"[{'?' * width}] ?/?  ?%"
    pct = done / total
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done}/{total}  {pct*100:.1f}%"


def _scan_progress(cfg: dict):
    """Scan results directory. Returns (completed, metric_summary, dataframe|None)."""
    results_dir = cfg.get("output_dir", "./results")
    pattern = os.path.join(results_dir, "*", "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        return 0, {}, None

    if HAS_PANDAS:
        try:
            dfs = [pd.read_csv(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
            id_col = next((c for c in ["run_id", "id"] if c in df.columns), None)
            if id_col:
                df = df.drop_duplicates(subset=[id_col])
            completed = len(df)

            # Per-group metric summary from config
            summary = {}
            group_cols = cfg.get("results", {}).get("group_by", [])
            metrics = cfg.get("results", {}).get("metrics", [])
            if group_cols and metrics:
                cols = [c for c in group_cols + metrics if c in df.columns]
                if len(cols) > len(group_cols):
                    summary = (
                        df[cols].groupby(group_cols)
                        .mean()
                        .reset_index()
                        .to_dict(orient="records")
                    )
            return completed, summary, df
        except Exception:
            pass

    return len(files), {}, None


def _kbhit() -> bool:
    """Non-blocking keypress check (Unix only)."""
    if not HAS_TERMIOS:
        return False
    return select.select([sys.stdin], [], [], 0)[0] != []


# ---------------------------------------------------------------------------
# Live monitor
# ---------------------------------------------------------------------------

def do_monitor(cfg: dict):
    """
    Live progress monitor. Refreshes every 2 seconds.
    Press any key to return to the main menu.
    """
    expected = cfg.get("expected_runs", 0)
    results_dir = cfg.get("output_dir", "./results")
    group_cols = cfg.get("results", {}).get("group_by", [])
    metrics = cfg.get("results", {}).get("metrics", [])

    old_settings = None
    if HAS_TERMIOS:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        while True:
            clear()
            header(cfg.get("experiment_name", "experiment"))
            print()
            print(f"  Results dir : {results_dir}")
            print(f"  Expected    : {expected} runs")
            print()

            completed, summary, df = _scan_progress(cfg)
            print(f"  Progress  {_progress_bar(completed, expected)}")
            print()

            # Per-phase breakdown
            print("  Phase breakdown:")
            for phase in cfg.get("phases", []):
                phase_total = count_lines(phase.get("file", ""))
                phase_done = 0
                if df is not None and HAS_PANDAS:
                    try:
                        mask = pd.Series([True] * len(df), index=df.index)
                        for key, val in phase.get("filters", {}).items():
                            col_map = {"noise": "noise_model", "source": "source", "ansatz": "ansatz"}
                            col = col_map.get(key, key)
                            if col in df.columns:
                                mask &= df[col] == val
                        phase_done = int(mask.sum())
                    except Exception:
                        pass
                pbar = _progress_bar(phase_done, phase_total, width=18)
                print(f"    [{phase['id']}] {phase['description']:<42} {pbar}")
            print()

            # Metric summary
            if summary:
                metric_labels = cfg.get("labels", {}).get("metrics", {})
                print(f"  Mean metrics by {', '.join(group_cols)}:")
                for row in summary:
                    group_str = " / ".join(str(row[g]) for g in group_cols if g in row)
                    metric_str = "  ".join(
                        f"{metric_labels.get(m, m)}: {row[m]*100:.1f}%"
                        for m in metrics if m in row
                    )
                    print(f"    {group_str:<30} {metric_str}")
                print()

            # SLURM queue
            squeue = run(
                "squeue -u $USER --format='%.10i %.9P %.30j %.8T %.10M' 2>/dev/null",
                capture=True,
            )
            if squeue:
                lines = squeue.splitlines()
                active = max(len(lines) - 1, 0)
                print(f"  Active SLURM jobs: {active}")
                for line in lines[:6]:
                    print(f"    {line}")
            else:
                print("  SLURM queue: not available")

            print()
            print("  " + "─" * 60)
            print("  [Press any key to return to the main menu]")

            if HAS_TERMIOS:
                if _kbhit():
                    sys.stdin.read(1)
                    break
            else:
                time.sleep(2)
                continue

            time.sleep(2)

    finally:
        if old_settings is not None and HAS_TERMIOS:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    print()


# ---------------------------------------------------------------------------
# Check completed / pending runs
# ---------------------------------------------------------------------------

def _get_completed_run_ids(cfg: dict) -> set:
    """Return set of run_ids that have a CSV result file."""
    results_dir = cfg.get("output_dir", "./results")
    completed = set()
    for csv_path in glob.glob(os.path.join(results_dir, "*", "*.csv")):
        run_id = Path(csv_path).parent.name
        completed.add(run_id)
    return completed


def do_check(cfg: dict, phase: Optional[dict] = None):
    """
    Display completed / pending runs for a phase (or all phases).
    Offers skip-all / overwrite-all before submitting.
    Returns "skip_all", "overwrite_all", or None (user cancelled).
    """
    completed_ids = _get_completed_run_ids(cfg)

    # Build run list for the phase
    phases = [phase] if phase else cfg.get("phases", [])
    all_run_ids: list[str] = []
    for ph in phases:
        cmd_file = ph.get("file", "")
        if not os.path.exists(cmd_file):
            continue
        with open(cmd_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Extract run-id from the command line
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "--run-id" and i + 1 < len(parts):
                        all_run_ids.append(parts[i + 1])
                        break

    if not all_run_ids:
        print("\n  No command files found. Run [R] first.")
        input("\nEnter to return...")
        return None

    done = [r for r in all_run_ids if r in completed_ids]
    pending = [r for r in all_run_ids if r not in completed_ids]

    print(f"\n  Total: {len(all_run_ids)}  |  Done: {len(done)}  |  Pending: {len(pending)}")

    if done:
        print(f"\n  Completed runs ({len(done)}):")
        for r in done[:20]:
            print(f"    [x] {r}")
        if len(done) > 20:
            print(f"    ... and {len(done) - 20} more")

    if pending:
        print(f"\n  Pending runs ({len(pending)}):")
        for r in pending[:20]:
            print(f"    [ ] {r}")
        if len(pending) > 20:
            print(f"    ... and {len(pending) - 20} more")

    if not done:
        input("\nAll runs are pending. Enter to return...")
        return None

    print("\n  Completed runs exist. What to do when submitting?")
    print("  [S] Skip completed (only run pending)")
    print("  [O] Overwrite all (re-run everything)")
    print("  [C] Cancel — return to menu")

    while True:
        choice = input("  Choice: ").strip().upper()
        if choice == "S":
            print(f"  [OK] Will skip {len(done)} completed run(s).")
            input("\nEnter to return...")
            return "skip_all"
        if choice == "O":
            results_dir = cfg.get("output_dir", "./results")
            print(f"\n  Deleting results for {len(done)} run(s)...")
            import shutil
            for run_id in done:
                run_dir = Path(results_dir) / run_id
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                    print(f"    Removed: {run_dir}")
            print(f"  Done. {len(done)} folder(s) deleted.")
            input("\nEnter to continue...")
            return "overwrite_all"
        if choice == "C":
            return None
        print("  Enter S, O, or C.")


# ---------------------------------------------------------------------------
# Phase submission
# ---------------------------------------------------------------------------

def do_refresh(config_path: str, cfg: dict):
    print(f"\n[INFO] Generating command files via runner.py...")
    ok = run(f"python runner.py --config {config_path} --export-commands")
    if ok:
        print("[OK] Done.")
        for p in cfg.get("phases", []):
            n = count_lines(p.get("file", ""))
            print(f"  [{p['id']}] {p['description']}: {n} tasks")
    else:
        print("[FAIL] Check runner.py for errors.")
    input("\nEnter to return...")


def do_submit(
    phase: dict,
    dependency_id: Optional[str] = None,
    slurm_sh: str = "core/slurm_generic.sh",
    conda_env: str = "experiment",
    overwrite: bool = False,
) -> Optional[str]:
    n = count_lines(phase.get("file", ""))
    if n == 0:
        print(f"\n[WARN] {phase['file']} is empty. Run [R] first.")
        return None
    dep = f"--dependency=afterok:{dependency_id}" if dependency_id else ""
    name = f"{phase['id']}_{phase['name']}"
    overwrite_flag = "--overwrite" if overwrite else ""
    cmd = (
        f"sbatch --parsable --job-name='{name}' "
        f"--array=1-{n}%30 {dep} "
        f"--export=CMD_FILE={phase['file']},CONDA_ENV={conda_env},"
        f"EXTRA_ARGS={overwrite_flag} "
        f"{slurm_sh}"
    )
    print(f"\n[SUBMIT] {phase['description']} ({n} tasks)...")
    job_id = run(cmd, capture=True)
    if job_id:
        print(f"[OK] Job ID: {job_id}")
    return job_id


def do_full_pipeline(cfg: dict, slurm_sh: str, conda_env: str, overwrite: bool = False):
    phases = cfg.get("phases", [])
    print(f"\n[PIPELINE] Submitting {len(phases)} phases sequentially...")
    prev, ids = None, []
    for p in phases:
        jid = do_submit(p, prev, slurm_sh, conda_env, overwrite)
        ids.append(jid or "?")
        if jid:
            prev = jid
    print(f"\n[OK] Chain: {' -> '.join(ids)}")
    input("\nEnter to return...")


def do_tables(config_path: str):
    print("\n[TABLES] Running generate_tables.py...")
    run(f"python core/generate_tables.py --config {config_path}")
    input("\nEnter to return...")


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg.get("experiment_name", "experiment")
    phases = cfg.get("phases", [])
    slurm_sh = cfg.get("slurm_script", "core/slurm_generic.sh")
    conda_env = cfg.get("conda_env", "experiment")

    os.makedirs("logs", exist_ok=True)
    os.makedirs(cfg.get("output_dir", "./results"), exist_ok=True)

    while True:
        clear()
        header(name)
        print()
        print("  [R]  Refresh command files from config.yaml")
        print()
        for p in phases:
            n = count_lines(p.get("file", ""))
            status = f"{n} tasks" if n > 0 else "empty — run [R]"
            print(f"  [{p['id']}]  {p['description']}  ({status})")
        print()
        print("  [F]  Launch FULL PIPELINE (all phases, sequential deps)")
        print("  [M]  Monitor progress  (live, refreshes every 2s)")
        print("  [C]  Check completed / pending runs")
        print("  [T]  Generate LaTeX tables")
        print("  [X]  Exit")
        print("-" * 70)

        c = input("Option: ").strip().upper()

        if c == "R":
            do_refresh(args.config, cfg)

        elif c == "F":
            overwrite_mode = do_check(cfg)
            overwrite = overwrite_mode == "overwrite_all"
            do_full_pipeline(cfg, slurm_sh, conda_env, overwrite)

        elif c == "M":
            do_monitor(cfg)

        elif c == "C":
            do_check(cfg)

        elif c == "T":
            do_tables(args.config)

        elif c == "X":
            print("\nDone.\n")
            break

        elif c in {p["id"] for p in phases}:
            p = next(ph for ph in phases if ph["id"] == c)
            overwrite_mode = do_check(cfg, phase=p)
            overwrite = overwrite_mode == "overwrite_all"
            do_submit(p, slurm_sh=slurm_sh, conda_env=conda_env, overwrite=overwrite)
            input("\nEnter to return...")

        else:
            print("Unknown option.")
            time.sleep(1)


if __name__ == "__main__":
    main()
