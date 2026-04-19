"""
runner.py — Thyroid cancer recurrence QML experiment runner.

Executes one (model, subset, seed) combination per invocation, saves results
to CSV in results/<run_id>/results.csv.

Models:
  - svm_classical   : RBF-SVM baseline
  - pegasos_ideal   : PegasosQSVC with exact statevector kernel (noiseless)
  - pegasos_noise   : PegasosQSVC with AerSimulator + IBM backend noise model
  - qnn_ideal       : SamplerQNN + TwoLocal, statevector sampler (noiseless)
  - qnn_noise       : SamplerQNN + TwoLocal, AerSimulator + IBM noise model

Feature subsets:
  - all   : full 13-feature matrix
  - top9  : 9 RULEx-ranked features
  - top4  : 4 RULEx-ranked features

Feature map and ansatz use reps=1, linear entanglement: the canonical
ZZFeatureMap baseline plus a TwoLocal(ry, rz; CX) ansatz. This matches the
configuration documented in the companion paper.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Plot utilities — optional, fail silently if matplotlib not available
try:
    sys.path.insert(0, "core")
    from plot_utils import plot_confusion_matrix, plot_roc_curve, plot_loss_curve
    _HAS_PLOTS = True
except Exception:
    _HAS_PLOTS = False


def _save_plots(run_dir: Path, y_true, y_pred, y_scores, loss_history, title: str):
    if not _HAS_PLOTS:
        return
    try:
        plot_confusion_matrix(y_true, y_pred, run_dir / "confusion_matrix.png", title=title)
    except Exception as e:
        logger.warning(f"confusion_matrix plot failed: {e}")
    if y_scores is not None:
        try:
            from tools import compute_extended_metrics
            auc = compute_extended_metrics(y_true, y_pred, y_scores).get("roc_auc")
            plot_roc_curve(y_true, y_scores, run_dir / "roc_curve.png",
                           title=title, auc_val=auc)
        except Exception as e:
            logger.warning(f"roc_curve plot failed: {e}")
    if loss_history:
        try:
            plot_loss_curve(loss_history, run_dir / "loss_curve.png", title=f"{title} | Loss")
        except Exception as e:
            logger.warning(f"loss_curve plot failed: {e}")


# =============================================================================
# Data loading
# =============================================================================

DATA_FILE = "thyroid_clean.xlsx"

COLS_DROP = [
    "Types of Thyroid Cancer (Pathology)",
    "Thyroid Function",
    "Treatment Response",
    "Adenopathy",
]

# RULEx-ranked feature indices (positions in the cleaned feature matrix,
# after dropping COLS_DROP and any leading id column). Order reflects
# descending importance from the association-rule analysis.
SUBSET_INDICES = {
    "all":  None,
    "top9": [7, 0, 1, 2, 5, 6, 9, 10, 11],
    "top4": [7, 0, 1, 2],
}


def load_dataset(data_path: str):
    import pandas as pd
    df = pd.read_excel(data_path)
    if df.iloc[:, 0].is_monotonic_increasing and df.iloc[:, 0].nunique() == len(df):
        df = df.iloc[:, 1:]
    df = df.drop(columns=COLS_DROP, errors="ignore").copy()
    if df.columns[-1] != "target":
        df = df.rename(columns={df.columns[-1]: "target"})
    X = df.drop(columns=["target"]).astype(int)
    y = df["target"].astype(int)
    return X, y


def prepare_split(X, y, subset: str, test_size: float, seed: int):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    indices = SUBSET_INDICES[subset]
    if indices is None:
        indices = list(range(X_tr_raw.shape[1]))
    X_tr = X_tr_raw.iloc[:, indices].values
    X_te = X_te_raw.iloc[:, indices].values
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, np.array(y_tr), X_te, np.array(y_te), indices


# =============================================================================
# Model dispatch — returns (metrics, classifier)
# =============================================================================

def _heron_r2_noise_model(cfg: dict):
    """Depolarizing NoiseModel approximating IBM Heron R2 (ibm_fez) without API."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    p1 = float(cfg.get("heron_r2_noise_p1", 0.0003))  # median 1Q ~0.03%
    p2 = float(cfg.get("heron_r2_noise_p2", 0.004))   # median 2Q ECR ~0.4%
    nm = NoiseModel()
    err_1q = depolarizing_error(p1, 1)
    err_2q = depolarizing_error(p2, 2)
    for gate in ["u1", "u2", "u3", "h", "rz", "sx", "x"]:
        nm.add_all_qubit_quantum_error(err_1q, gate)
    for gate in ["cx", "ecr"]:
        nm.add_all_qubit_quantum_error(err_2q, gate)
    return nm


def _make_noisy_sampler(backend_name: str, use_gpu: bool, cfg: dict = None):
    from qiskit.primitives import BackendSampler
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    cfg = cfg or {}
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        real_backend = QiskitRuntimeService().backend(backend_name)
        noise_model = NoiseModel.from_backend(real_backend)
        logger.info(f"[NOISE] Loaded model from IBM API ({backend_name})")
    except Exception as e:
        logger.warning(f"[NOISE] IBM API failed ({e}); using Heron R2 fallback params")
        noise_model = _heron_r2_noise_model(cfg)

    method = "tensor_network" if use_gpu else "matrix_product_state"
    sim_kw = dict(method=method, noise_model=noise_model)
    if use_gpu:
        sim_kw.update(
            device="GPU",
            batched_shots_gpu=True,
            blocking_enable=True,
            blocking_qubits=20,
        )
    return BackendSampler(backend=AerSimulator(**sim_kw))


def _run_model_cudaq(model: str, X_tr, y_tr, X_te, y_te, cfg: dict):
    import tools_cudaq as tc
    from tools import evaluate_classical_svm

    n_q = X_tr.shape[1]
    reps_fm = int(cfg.get("reps_fm", 1))
    reps_ansatz = int(cfg.get("reps_ansatz", 1))
    p1 = float(cfg.get("cudaq_noise_p1", 0.001))
    p2 = float(cfg.get("cudaq_noise_p2", 0.005))

    if model == "svm_classical":
        return evaluate_classical_svm(X_tr, y_tr, X_te, y_te)
    if model == "pegasos_ideal":
        return tc.evaluate_qsvm_statevector(n_q, reps_fm, X_tr, y_tr, X_te, y_te)
    if model == "qnn_ideal":
        return tc.evaluate_qnn_statevector(n_q, reps_fm, reps_ansatz, X_tr, y_tr, X_te, y_te)
    if model == "pegasos_noise":
        return tc.evaluate_qsvm_noise_sim(n_q, reps_fm, X_tr, y_tr, X_te, y_te, p1=p1, p2=p2)
    if model == "qnn_noise":
        return tc.evaluate_qnn_noise_sim(n_q, reps_fm, reps_ansatz, X_tr, y_tr, X_te, y_te, p1=p1, p2=p2)
    raise ValueError(f"Unknown model: {model}")


def run_model(model: str, X_tr, y_tr, X_te, y_te, cfg: dict):
    if cfg.get("quantum_backend", "qiskit") == "cudaq":
        return _run_model_cudaq(model, X_tr, y_tr, X_te, y_te, cfg)

    import tools
    from tools import (
        createFeatureMapLinear,
        evaluate_qsvm_statevector,
        evaluate_qnn_statevector,
        evaluate_classical_svm,
    )

    n_qubits = X_tr.shape[1]
    backend = cfg.get("backend_name", "ibm_brisbane")
    use_gpu = bool(cfg.get("use_gpu", False))
    reps_fm = int(cfg.get("reps_fm", 1))
    reps_ansatz = int(cfg.get("reps_ansatz", 1))

    if model == "svm_classical":
        return evaluate_classical_svm(X_tr, y_tr, X_te, y_te)

    fm = createFeatureMapLinear(n_qubits, reps=reps_fm)

    if model == "pegasos_ideal":
        return evaluate_qsvm_statevector(fm, X_tr, y_tr, X_te, y_te)

    if model == "qnn_ideal":
        return evaluate_qnn_statevector(
            fm, X_tr, y_tr, X_te, y_te, reps_ansatz=reps_ansatz
        )

    if model == "pegasos_noise":
        from qiskit_algorithms.state_fidelities import ComputeUncompute
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.algorithms import PegasosQSVC

        sampler = _make_noisy_sampler(backend, use_gpu, cfg)
        fidelity = ComputeUncompute(sampler=sampler)
        qkernel = FidelityQuantumKernel(feature_map=fm, fidelity=fidelity)
        clf = PegasosQSVC(
            quantum_kernel=qkernel,
            C=tools._QSVM_C,
            num_steps=tools._QSVM_NUM_STEPS,
            seed=tools._QSVM_SEED,
        )
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        m = tools.compute_metrics(clf, X_te, y_te)
        m["training_time"] = float(time.time() - t0)
        return m, clf

    if model == "qnn_noise":
        from qiskit_machine_learning.neural_networks import SamplerQNN

        sampler = _make_noisy_sampler(backend, use_gpu, cfg)
        ansatz, _, wt_params = tools.build_ansatz(n_qubits, reps=reps_ansatz)
        qc, in_params, wt_params = tools.create_qnn_circuit(n_qubits, fm, ansatz)
        qnn = SamplerQNN(
            circuit=qc,
            input_params=in_params,
            weight_params=wt_params,
            sampler=sampler,
            interpret=tools.parity,
            output_shape=2,
        )
        loss_history = []
        clf = tools._make_qnn_classifier(qnn, wt_params, loss_history=loss_history)
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        m = tools.compute_metrics(clf, X_te, y_te)
        m["training_time"] = float(time.time() - t0)
        m["loss_history"] = loss_history
        return m, clf

    raise ValueError(f"Unknown model: {model}")


def sens_spec_from_clf(clf, X_te, y_te) -> tuple[float, float]:
    from sklearn.metrics import confusion_matrix
    y_pred = np.asarray(clf.predict(X_te))
    y_true = np.asarray(y_te)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return float(sens), float(spec)


# =============================================================================
# Framework hooks
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_run_id(model: str, subset: str, seed, backend: str = "qiskit") -> str:
    return f"{model}__{subset}__s{seed}__{backend}"


def iter_runs(cfg: dict):
    backend = cfg.get("quantum_backend", "qiskit")
    models = [m["name"] for m in cfg.get("models", [])]
    subsets = [s["name"] for s in cfg.get("subsets", [])]
    seeds = cfg.get("seeds", [12345])
    for model, subset, seed in product(models, subsets, seeds):
        yield {
            "run_id": make_run_id(model, subset, seed, backend),
            "model": model,
            "subset": subset,
            "seed": int(seed),
            "noise": model.endswith("_noise"),
            "backend": backend,
        }


def apply_filter(runs, phase):
    filters = phase.get("filters", {})
    out = []
    for r in runs:
        if all(r.get(k) == v for k, v in filters.items()):
            out.append(r)
    return out


def execute_run(run_spec: dict, cfg: dict, machine_id: str = "local"):
    run_id = run_spec["run_id"]
    model = run_spec["model"]
    subset = run_spec["subset"]
    seed = int(run_spec.get("seed", 12345))

    results_dir = cfg.get("output_dir", "./results")
    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "results.csv"
    json_path = run_dir / "metrics.json"

    if csv_path.exists():
        logger.info(f"[SKIP] {run_id} already completed.")
        return

    logger.info(f"[START] {run_id} | model={model} subset={subset} seed={seed}")

    np.random.seed(seed)
    if cfg.get("quantum_backend", "qiskit") == "qiskit":
        from qiskit_algorithms.utils import algorithm_globals
        algorithm_globals.random_seed = seed

    X, y = load_dataset(cfg.get("data_file", DATA_FILE))
    X_tr, y_tr, X_te, y_te, indices = prepare_split(
        X, y, subset=subset, test_size=float(cfg.get("test_size", 0.3)), seed=seed
    )

    sensitivity = float("nan")
    specificity = float("nan")
    ext_metrics: dict = {}
    loss_history: list = []
    t0 = time.time()
    try:
        metrics, clf = run_model(model, X_tr, y_tr, X_te, y_te, cfg)
        loss_history = metrics.pop("loss_history", [])
        sensitivity, specificity = sens_spec_from_clf(clf, X_te, y_te)

        from tools import compute_extended_metrics, get_scores
        y_pred = np.asarray(clf.predict(X_te))
        y_true_arr = np.asarray(y_te)
        y_scores = get_scores(clf, X_te)
        ext_metrics = compute_extended_metrics(y_true_arr, y_pred, y_scores)

        _save_plots(run_dir, y_true_arr, y_pred, y_scores, loss_history,
                    title=f"{model} | {subset} | s{seed}")
        status = "ok"
        error_msg = ""
    except Exception as exc:
        logger.exception(f"[ERROR] {run_id}: {exc}")
        metrics = {
            "accuracy": float("nan"),
            "precision_macro": float("nan"),
            "recall_macro": float("nan"),
            "f1_macro": float("nan"),
            "inference_time": float("nan"),
            "training_time": float("nan"),
        }
        status = "error"
        error_msg = str(exc)
    elapsed = time.time() - t0

    row = {
        "run_id": run_id,
        "model": model,
        "subset": subset,
        "seed": seed,
        "backend": run_spec.get("backend", cfg.get("quantum_backend", "qiskit")),
        "n_features": len(indices),
        "noise": bool(run_spec.get("noise", False)),
        "accuracy": metrics.get("accuracy", float("nan")),
        "precision_macro": metrics.get("precision_macro", float("nan")),
        "recall_macro": metrics.get("recall_macro", float("nan")),
        "f1_macro": metrics.get("f1_macro", float("nan")),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mcc": ext_metrics.get("mcc", float("nan")),
        "balanced_accuracy": ext_metrics.get("balanced_accuracy", float("nan")),
        "kappa": ext_metrics.get("kappa", float("nan")),
        "roc_auc": ext_metrics.get("roc_auc", float("nan")),
        "training_time": metrics.get("training_time", float("nan")),
        "inference_time": metrics.get("inference_time", float("nan")),
        "wall_time_total": elapsed,
        "status": status,
        "error": error_msg,
        "machine_id": machine_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    logger.info(
        f"[DONE] {run_id} | acc={row['accuracy']:.4f} auc={row['roc_auc']:.4f} "
        f"mcc={row['mcc']:.4f} sens={sensitivity:.4f} t={elapsed:.1f}s status={status}"
    )


# =============================================================================
# Command export
# =============================================================================

def export_commands(runs, out_path, config_path, backend=None):
    lines = []
    backend_flag = f" --backend {backend}" if backend else ""
    for r in runs:
        params = " ".join(
            f"--{k} {v}"
            for k, v in r.items()
            if k not in ("run_id", "noise", "backend")
        )
        lines.append(
            f"python runner.py --config {config_path} --run-id {r['run_id']} {params}{backend_flag}"
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Exported {len(lines)} commands to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--export-commands", action="store_true")
    ap.add_argument("--machine-id", default="local")
    ap.add_argument("--backend", choices=["qiskit", "cudaq"], default=None,
                    help="Override quantum_backend in config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg["quantum_backend"] = args.backend
    all_runs = list(iter_runs(cfg))

    if args.export_commands:
        for phase in cfg.get("phases", []):
            filtered = apply_filter(all_runs, phase)
            export_commands(filtered, phase["file"], args.config, backend=args.backend)
        return

    if args.run_id:
        run_spec = next(
            (r for r in all_runs if r["run_id"] == args.run_id),
            {
                "run_id": args.run_id,
                "model": args.model,
                "subset": args.subset,
                "seed": args.seed if args.seed is not None else 12345,
                "noise": bool(args.model and args.model.endswith("_noise")),
                "backend": cfg.get("quantum_backend", "qiskit"),
            },
        )
        execute_run(run_spec, cfg, args.machine_id)
        return

    runs = all_runs
    if args.model:
        runs = [r for r in runs if r["model"] == args.model]
    if args.subset:
        runs = [r for r in runs if r["subset"] == args.subset]
    if args.seed is not None:
        runs = [r for r in runs if r["seed"] == args.seed]

    logger.info(f"Planned runs: {len(runs)}")
    if args.dry_run:
        for r in runs:
            print(f"  {r['run_id']}")
        return

    for r in runs:
        execute_run(r, cfg, args.machine_id)


if __name__ == "__main__":
    main()
