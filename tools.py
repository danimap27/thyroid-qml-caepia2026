"""
tools.py — SEQUENT toolkit.

Models  : PegasosQSVC (QSVM) and NeuralNetworkClassifier (QNN).
Modes   : statevector (ideal), noise emulator, real hardware.
Metrics : accuracy, macro precision/recall/F1, per-class breakdown (R2-10).
"""

import os
import time
import random

import numpy as np
import pandas as pd

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.transpiler import generate_preset_pass_manager, PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit.circuit.library import XGate

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.primitives import BackendSampler

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier, PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             roc_auc_score, matthews_corrcoef,
                             balanced_accuracy_score, cohen_kappa_score)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pmlb import fetch_data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

os.environ["QISKIT_AER_CPU_THREADS"] = str(os.cpu_count())
os.environ["QISKIT_AER_CUQUANTUM"] = "1"
os.environ["CUQUANTUM_MGPU"] = "1"

np.random.seed(12345)
random.seed(12345)
algorithm_globals.random_seed = 12345


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════

def _preprocess_heart(df):
    df = df.drop(['RestingECG', 'ST_Slope', 'Age', 'Sex'], axis=1)
    return pd.get_dummies(df, columns=['ChestPainType', 'ExerciseAngina'])

def _preprocess_fitness(df):
    df = df.drop(['booking_id'], axis=1)
    df['day_of_week'] = df['day_of_week'].map(
        {'Mon':1,'Tue':2,'Wed':3,'Thu':4,'Fri':5,'Sat':6,'Sun':7})
    df['time']        = df['time'].map({'AM':0,'PM':1})
    df['category']    = df['category'].map({'Strength':1,'HIIT':2,'Cycling':3,'Aqua':4})
    df['days_before'] = df['days_before'].str.extract(r'(\d+)').astype(int)
    df['day_of_week'].fillna(df['day_of_week'].mode()[0], inplace=True)
    df['category'].fillna(df['category'].mode()[0], inplace=True)
    df['weight'].fillna(df['weight'].mean(), inplace=True)
    df['weight'] = df['weight'].round().astype(int)
    return df

def load_data(path=None, option=0, dataset=None):
    """
    option=0 → local CSV/TSV at `path`.
    option=1 → PMLB dataset by name.
    Returns X (DataFrame), y (Series).
    """
    if option == 1:
        df = fetch_data(dataset)
    else:
        if "breast" in path or "flare" in path:
            df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path)
        if "fitness" in path: df = _preprocess_fitness(df)
        if "heart"   in path: df = _preprocess_heart(df)

    y = df['target']
    X = df.drop(['target'], axis=1).astype(int)
    return X, y


# ═══════════════════════════════════════════════════════════════
#  FEATURE SELECTION  (R2-13: ANOVA vs mutual info vs autoencoder)
# ═══════════════════════════════════════════════════════════════

# ── Autoencoder architecture ─────────────────────────────────────────────────

class _Autoencoder(nn.Module):
    """
    Symmetric autoencoder for unsupervised feature compression.

    Architecture
    ────────────
    Encoder : n_features → hidden_dim → latent_dim (k)
    Decoder : k          → hidden_dim → n_features

    The latent layer uses no activation so the latent space is unbounded
    and can represent both positive and negative feature interactions.
    All other layers use ReLU + BatchNorm for stable training.

    Parameters
    ----------
    n_features  : int — number of input features
    latent_dim  : int — target number of latent dimensions (= k)
    hidden_dim  : int — size of the intermediate hidden layer
    """

    def __init__(self, n_features: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            # No activation: latent space must remain unconstrained
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def _train_autoencoder(
    X_scaled: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: torch.device,
    seed: int,
) -> _Autoencoder:
    """
    Trains the autoencoder with MSE reconstruction loss, Adam optimiser
    and early stopping on a held-out validation split (10% of data).

    Returns the trained model in eval() mode.
    """
    torch.manual_seed(seed)

    n, n_feat = X_scaled.shape
    val_size  = max(1, int(0.1 * n))
    idx       = np.random.permutation(n)
    tr_idx, val_idx = idx[val_size:], idx[:val_size]

    X_tr  = torch.tensor(X_scaled[tr_idx],  dtype=torch.float32, device=device)
    X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(X_tr, X_tr),
                        batch_size=min(batch_size, len(tr_idx)),
                        shuffle=True)

    model = _Autoencoder(n_feat, latent_dim, hidden_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    wait          = 0

    model.train()
    for epoch in range(1, epochs + 1):
        for xb, _ in loader:
            opt.zero_grad()
            loss_fn(model(xb), xb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), X_val).item()
        model.train()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            wait          = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    → early stop at epoch {epoch}  "
                      f"(best val_loss={best_val_loss:.6f})")
                break

        if epoch % 50 == 0:
            print(f"    epoch {epoch:>4}  val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def apply_feature_selection(X, y=None, method="anova", k=5,
                             # Autoencoder-specific hyperparameters
                             ae_epochs=200,
                             ae_batch_size=32,
                             ae_lr=1e-3,
                             ae_patience=20,
                             ae_hidden_factor=2,
                             ae_seed=12345):
    """
    Applies feature selection and returns a reduced DataFrame.

    Parameters
    ----------
    X              : DataFrame — input features (already preprocessed, int-cast)
    y              : Series    — target labels (required for anova/mutual_info)
    method         : str
        'anova'       — ANOVA F-score. Fast, captures only linear feature-target
                        relationships. Best baseline choice.
        'mutual_info' — Mutual information. Model-free, captures non-linear
                        dependencies between features and target.
        'autoencoder' — Unsupervised deep compression. Trains a symmetric
                        autoencoder on X and returns the k-dimensional latent
                        representation. Captures complex non-linear inter-feature
                        correlations without using y (unsupervised).
                        Output columns are named ae_0 … ae_{k-1}.
    k              : int — target number of dimensions / features to keep.
    ae_epochs      : int — maximum training epochs for the autoencoder.
    ae_batch_size  : int — mini-batch size for autoencoder training.
    ae_lr          : float — Adam learning rate.
    ae_patience    : int — early-stopping patience (epochs without improvement).
    ae_hidden_factor: int — hidden_dim = ae_hidden_factor × k
                            (minimum: n_features // 2).
    ae_seed        : int — random seed for reproducibility.

    Returns
    -------
    X_reduced : DataFrame  — shape (n_samples, k)
    cols      : Index      — column names of the reduced DataFrame
    """
    k = min(k, X.shape[1])

    # ── SelectKBest methods ──────────────────────────────────────────────────
    if method in ("anova", "mutual_info"):
        if y is None:
            raise ValueError(f"method='{method}' requires y (target labels).")
        score_func = f_classif if method == "anova" else mutual_info_classif
        sel        = SelectKBest(score_func=score_func, k=k)
        X_arr      = sel.fit_transform(X, y)
        cols       = X.columns[sel.get_support()]
        print(f"    → kept {k} features: {list(cols)}")
        return pd.DataFrame(X_arr, columns=cols), cols

    # ── Autoencoder ──────────────────────────────────────────────────────────
    elif method == "autoencoder":
        n_features  = X.shape[1]
        hidden_dim  = max(ae_hidden_factor * k, n_features // 2, k + 1)

        print(f"    [AE] architecture: {n_features} → {hidden_dim} → {k} "
              f"→ {hidden_dim} → {n_features}")
        print(f"    [AE] epochs={ae_epochs}  batch={ae_batch_size}  "
              f"lr={ae_lr}  patience={ae_patience}")

        # Normalise: autoencoder training is sensitive to feature scale
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X.values.astype(np.float32))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    [AE] device: {device}")

        model = _train_autoencoder(
            X_scaled    = X_scaled,
            latent_dim  = k,
            hidden_dim  = hidden_dim,
            epochs      = ae_epochs,
            batch_size  = ae_batch_size,
            lr          = ae_lr,
            patience    = ae_patience,
            device      = device,
            seed        = ae_seed,
        )

        # Extract latent representations for the full dataset
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            latent   = model.encode(X_tensor).cpu().numpy()

        cols      = pd.Index([f"ae_{i}" for i in range(k)])
        X_reduced = pd.DataFrame(latent, columns=cols, index=X.index)

        print(f"    [AE] latent shape: {X_reduced.shape}  "
              f"(columns: {list(cols)})")
        return X_reduced, cols

    else:
        raise ValueError(
            f"fs_method must be 'anova', 'mutual_info' or 'autoencoder'; "
            f"got '{method}'"
        )


# ═══════════════════════════════════════════════════════════════
#  CORRELATION → ENTANGLEMENT MAP
# ═══════════════════════════════════════════════════════════════

def transformCorrelations(correlation_matrix):
    """Upper-triangular values of the correlation matrix as a flat array."""
    upper = np.triu(correlation_matrix.values, k=1)
    return upper[np.triu_indices_from(upper, k=1)]

def createCouples(triangular_array, columns):
    """Maps flat-array positions back to (col_i, col_j) pairs."""
    couples, n, idx = [], len(columns), 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx < len(triangular_array):
                couples.append((columns[i], columns[j]))
            idx += 1
    return couples


# ═══════════════════════════════════════════════════════════════
#  FEATURE MAPS  (R1-02: linear, ring, full baselines)
# ═══════════════════════════════════════════════════════════════

def createFeatureMapLinear(num_features, reps=1):
    """ZZFeatureMap with linear entanglement — main baseline."""
    return ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement="linear")

def createFeatureMapRing(num_features, reps=1):
    """ZZFeatureMap with circular/ring entanglement — intermediate baseline."""
    return ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement="circular")

def createFeatureMapFull(num_features, reps=1):
    """ZZFeatureMap with full entanglement — upper-bound baseline."""
    return ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement="full")

def createFeatureMap(couples, columns, reps=1):
    """
    SEQUENT selective-entanglement feature map.
    Entangles only the qubit pairs in `couples`; all qubits get a single-qubit
    phase gate regardless to ensure no qubit is left unencoded.
    """
    ent_list = [(columns.get_loc(p[0]), columns.get_loc(p[1])) for p in couples]
    ent_dict = {
        1: [(i,) for i in range(len(columns))],
        2: ent_list,
    }
    fm  = ZZFeatureMap(feature_dimension=len(columns), reps=reps, entanglement=ent_dict)
    dec = fm.decompose()

    num_qubits  = dec.num_qubits
    qubit_gates = {i: [] for i in range(num_qubits)}
    for instr, qargs, _ in dec.data:
        for qa in qargs:
            qubit_gates[qa._index].append(instr)

    x = ParameterVector('x', num_qubits)
    for i in range(num_qubits):
        if [g.name for g in qubit_gates[i]] == ['h']:
            dec.p(2.0 * x[i], i)
    return dec


# ═══════════════════════════════════════════════════════════════
#  SPLIT
# ═══════════════════════════════════════════════════════════════

def splitData(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=12345)


# ═══════════════════════════════════════════════════════════════
#  METRICS  (R2-10: per-class precision/recall/F1)
# ═══════════════════════════════════════════════════════════════

def compute_metrics(model, X, y_true):
    """
    Returns macro-averaged metrics.
    Macro averaging weights all classes equally — correct for imbalanced
    datasets (BreastW 63/37, Fitness 70/30, Heart 44/56).
    """
    t0      = time.time()
    y_pred  = model.predict(X)
    elapsed = time.time() - t0
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(   y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":        float(f1_score(       y_true, y_pred, average="macro", zero_division=0)),
        "inference_time":  float(elapsed),
    }

def compute_extended_metrics(y_true, y_pred, y_scores=None) -> dict:
    """MCC, balanced accuracy, Cohen's kappa, ROC-AUC (if scores provided)."""
    result = {
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa":             float(cohen_kappa_score(y_true, y_pred)),
        "roc_auc":           float("nan"),
    }
    if y_scores is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except Exception:
            pass
    return result


def get_scores(clf, X) -> np.ndarray | None:
    """Return decision scores for ROC-AUC. Tries predict_scores → predict_proba → decision_function."""
    for method in ("predict_scores", "predict_proba", "decision_function"):
        fn = getattr(clf, method, None)
        if fn is None:
            continue
        try:
            out = fn(X)
            if out.ndim == 2:
                return out[:, 1]
            return out
        except Exception:
            pass
    return None


def compute_metrics_per_class(model, X, y_true):
    """
    Returns per-class precision, recall and F1 as a dict of dicts.
    Used for R2-10 (class-imbalance analysis).
    Example output: {"0": {"precision": 0.9, "recall": 0.85, "f1": 0.87}, ...}
    """
    y_pred  = model.predict(X)
    report  = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    classes = sorted([k for k in report if k not in ("accuracy","macro avg","weighted avg")])
    return {cls: {
        "precision": float(report[cls]["precision"]),
        "recall":    float(report[cls]["recall"]),
        "f1":        float(report[cls]["f1-score"]),
        "support":   int(report[cls]["support"]),
    } for cls in classes}


# ═══════════════════════════════════════════════════════════════
#  CIRCUIT COMPLEXITY  (R1-08: scalability analysis)
# ═══════════════════════════════════════════════════════════════

def circuit_complexity(feature_map):
    """
    Returns depth, total gate count and two-qubit gate count.
    Search-space size = 2^(k*(k-1)/2) where k = num_qubits.
    """
    dec   = feature_map.decompose()
    two_q = sum(1 for _, qargs, _ in dec.data if len(qargs) == 2)
    n     = dec.num_qubits
    return {
        "n_qubits":        n,
        "depth":           dec.depth(),
        "total_gates":     dec.size(),
        "two_qubit_gates": two_q,
        "search_space":    2 ** (n * (n - 1) // 2),   # 2^(C(k,2))
    }


# ═══════════════════════════════════════════════════════════════
#  QNN BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════

def parity(x):
    return bin(x).count("1") % 2

def build_ansatz(n_qubits, reps=1):
    ansatz        = TwoLocal(n_qubits, ["ry","rz"], "cx", entanglement="linear", reps=reps)
    ordered_params = sorted(ansatz.parameters, key=lambda p: p.name)
    return ansatz, ordered_params, n_qubits

def create_qnn_circuit(num_qubits, feature_map, ansatz):
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.barrier()
    qc.compose(ansatz, inplace=True)
    in_params = sorted(feature_map.parameters, key=lambda p: p.name)
    wt_params = sorted(ansatz.parameters,      key=lambda p: p.name)
    return qc, in_params, wt_params

def _make_qnn_classifier(qnn, wt_params, loss_history=None):
    callback = None
    if loss_history is not None:
        def callback(weights, obj_val):
            loss_history.append(float(obj_val))
    return NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=COBYLA(maxiter=100, tol=1e-4),
        one_hot=False,
        initial_point=np.random.uniform(0, 2 * np.pi, len(wt_params)),
        callback=callback,
    )


# ═══════════════════════════════════════════════════════════════
#  QSVM — THREE EXECUTION MODES
#
#  Recommended parameters for fast metaheuristic search:
#    C=1000, num_steps=100
#  Reasoning: each SA/GA evaluation trains a full QSVM.
#  With num_steps=100 (vs 500) you get 5× speedup per evaluation.
#  SA with max_iterations=10, num_neighbors=3 → ~30 total evaluations.
#  C=1000 vs 5000 has negligible accuracy impact on small datasets.
# ═══════════════════════════════════════════════════════════════

# ── QSVM config (single place to change) ────────────────────────
_QSVM_C         = 1000   # Regularisation. 1000 balances accuracy vs speed.
_QSVM_NUM_STEPS = 100    # Pegasos iterations. 100 is ~5× faster than 500.
_QSVM_SEED      = 12345

def evaluate_qsvm_statevector(feature_map, train_features, train_labels,
                               val_features, val_labels, **kwargs):
    """
    PegasosQSVC with exact statevector kernel — fastest mode, no noise.
    Ideal for metaheuristic search iterations.
    """
    qkernel = FidelityQuantumKernel(feature_map=feature_map)
    clf     = PegasosQSVC(quantum_kernel=qkernel, C=_QSVM_C,
                          num_steps=_QSVM_NUM_STEPS, seed=_QSVM_SEED)
    t0 = time.time()
    clf.fit(train_features, train_labels)
    train_t = time.time() - t0
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics["training_time"] = float(train_t)
    return metrics, clf

def evaluate_qsvm_noise_sim(feature_map, train_features, train_labels,
                             val_features, val_labels,
                             backend_name="ibm_brisbane", gpu=None, **kwargs):
    """PegasosQSVC with AerSimulator noise model from a real IBM backend."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    real_backend = QiskitRuntimeService().backend(backend_name)
    noise_model  = NoiseModel.from_backend(real_backend)
    sim_kw       = dict(method="tensor_network", noise_model=noise_model)
    if gpu is not None:
        sim_kw.update(device="GPU", batched_shots_gpu=True,
                      blocking_enable=True, blocking_qubits=20)

    sampler  = BackendSampler(backend=AerSimulator(**sim_kw),
                               options={"resilience_level": 2})
    fidelity = ComputeUncompute(sampler=sampler)
    qkernel  = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    clf      = PegasosQSVC(quantum_kernel=qkernel, C=_QSVM_C,
                           num_steps=_QSVM_NUM_STEPS, seed=_QSVM_SEED)
    t0 = time.time()
    clf.fit(train_features, train_labels)
    train_t = time.time() - t0
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics["training_time"] = float(train_t)
    return metrics, clf

def evaluate_qsvm_hardware(feature_map, train_features, train_labels,
                            val_features, val_labels,
                            backend_name="ibm_strasbourg", shots=10, **kwargs):
    """PegasosQSVC on real IBM Quantum hardware with dynamical decoupling."""
    backend = QiskitRuntimeService().backend(backend_name)
    pm      = generate_preset_pass_manager(backend=backend, optimization_level=2)
    pm.scheduling = PassManager([
        ALAPScheduleAnalysis(target=backend.target),
        PadDynamicalDecoupling(target=backend.target, dd_sequence=[XGate(), XGate()]),
    ])
    sampler  = SamplerV2(backend)
    sampler.options.default_shots = shots
    fidelity = ComputeUncompute(sampler, pass_manager=pm)
    qkernel  = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    clf      = PegasosQSVC(quantum_kernel=qkernel, C=_QSVM_C,
                           num_steps=_QSVM_NUM_STEPS, seed=_QSVM_SEED)
    t0 = time.time()
    clf.fit(train_features, train_labels)
    train_t = time.time() - t0
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics["training_time"] = float(train_t)
    return metrics, clf


# ═══════════════════════════════════════════════════════════════
#  QNN — THREE EXECUTION MODES
# ═══════════════════════════════════════════════════════════════

def evaluate_qnn_statevector(feature_map, train_features, train_labels,
                              val_features, val_labels, reps_ansatz=1, **kwargs):
    """QNN with default statevector sampler — no noise, no hardware."""
    num_q   = train_features.shape[1]
    ansatz, _, wt_params = build_ansatz(num_q, reps=reps_ansatz)
    qc, in_params, wt_params = create_qnn_circuit(num_q, feature_map, ansatz)
    qnn = SamplerQNN(circuit=qc, input_params=in_params, weight_params=wt_params,
                     interpret=parity, output_shape=2)
    loss_history = []
    clf = _make_qnn_classifier(qnn, wt_params, loss_history=loss_history)
    t0  = time.time()
    clf.fit(train_features, train_labels)
    train_t = time.time() - t0
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics["training_time"] = float(train_t)
    metrics["loss_history"] = loss_history
    return metrics, clf

def evaluate_qnn_noise_sim(feature_map, train_features, train_labels,
                            val_features, val_labels,
                            backend_name="ibm_brisbane", reps_ansatz=1,
                            gpu=None, **kwargs):
    """QNN on AerSimulator with real-device noise model."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    real_backend = QiskitRuntimeService().backend(backend_name)
    noise_model  = NoiseModel.from_backend(real_backend)
    sim_kw       = dict(method="tensor_network", noise_model=noise_model)
    if gpu is not None:
        sim_kw.update(device="GPU", batched_shots_gpu=True,
                      blocking_enable=True, blocking_qubits=20)
    sampler = BackendSampler(backend=AerSimulator(**sim_kw),
                              options={"resilience_level": 2})
    num_q   = train_features.shape[1]
    ansatz, _, wt_params = build_ansatz(num_q, reps=reps_ansatz)
    qc, in_params, wt_params = create_qnn_circuit(num_q, feature_map, ansatz)
    qnn = SamplerQNN(circuit=qc, input_params=in_params, weight_params=wt_params,
                     sampler=sampler, interpret=parity, output_shape=2)
    loss_history = []
    clf = _make_qnn_classifier(qnn, wt_params, loss_history=loss_history)
    t0  = time.time()
    clf.fit(train_features, train_labels)
    train_t = time.time() - t0
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics["training_time"] = float(train_t)
    metrics["loss_history"] = loss_history
    return metrics, clf

def evaluate_qnn_hardware(feature_map, train_features, train_labels,
                           val_features, val_labels,
                           backend_name="ibm_pittsburgh", shots=10,
                           job_tag="SEQUENT_QNN", reps_ansatz=1, **kwargs):
    """QNN on real IBM Quantum hardware with dynamical decoupling."""
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    pm      = generate_preset_pass_manager(backend=backend, optimization_level=2)
    pm.scheduling = PassManager([
        ALAPScheduleAnalysis(target=backend.target),
        PadDynamicalDecoupling(target=backend.target, dd_sequence=[XGate(), XGate()]),
    ])
    sampler = SamplerV2(backend)
    sampler.options.default_shots = shots
    sampler.options.environment.job_tags = [job_tag]
    num_q   = train_features.shape[1]
    ansatz, _, wt_params = build_ansatz(num_q, reps=reps_ansatz)
    qc, in_params, wt_params = create_qnn_circuit(num_q, feature_map, ansatz)
    qc.measure_all()
    qc_t = pm.run(qc)
    qnn  = SamplerQNN(circuit=qc_t, input_params=in_params, weight_params=wt_params,
                      sampler=sampler, interpret=parity, output_shape=2)
    clf  = _make_qnn_classifier(qnn, wt_params)
    t0   = time.time()
    clf.fit(train_features, train_labels)
    train_t = time.time() - t0
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics["training_time"] = float(train_t)
    return metrics, clf


# ═══════════════════════════════════════════════════════════════
#  CLASSICAL BASELINE
# ═══════════════════════════════════════════════════════════════

def evaluate_classical_svm(train_features, train_labels,
                            test_features, test_labels, kernel="rbf"):
    """
    RBF-SVM baseline.
    Purpose: isolate the quantum circuit's actual contribution to any gain.
    A quantum model that cannot beat a classical SVM needs justification.
    """
    clf = SVC(kernel=kernel, C=1000, random_state=12345)
    t0  = time.time()
    clf.fit(train_features, train_labels)
    train_t = time.time() - t0
    metrics = compute_metrics(clf, test_features, test_labels)
    metrics["training_time"] = float(train_t)
    return metrics, clf