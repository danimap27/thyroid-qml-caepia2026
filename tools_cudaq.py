"""
tools_cudaq.py — CUDA-Q backend for thyroid-qml CAEPIA experiments.

Drop-in replacement for the quantum parts of tools.py.
Mirrors the same public function signatures used by runner.py.

Targets:
  nvidia             — GPU statevector via custatevec (ideal, preferred on Hercules)
  qpp-cpu            — CPU statevector fallback (ideal, no GPU)
  density-matrix-cpu — CPU density matrix for noisy simulation

Noise model: generic depolarizing channels (IBM fez not directly supported).
Configure p1/p2 in config.yaml under cudaq_noise_p1 / cudaq_noise_p2.
"""

import math
import time

import numpy as np
import cudaq
from cudaq import spin
from scipy.optimize import minimize
from sklearn.svm import SVC

from tools import compute_metrics, _QSVM_C, _QSVM_SEED, evaluate_classical_svm

_PI = math.pi


# ── Simulator target management ───────────────────────────────────────────────

def _use_ideal():
    try:
        cudaq.set_target("nvidia")
    except Exception:
        cudaq.set_target("qpp-cpu")


def _use_noisy(p1: float, p2: float):
    cudaq.set_target("density-matrix-cpu")
    noise = cudaq.NoiseModel()
    for gate in ("h", "ry", "rz"):
        noise.add_all_qubit_channel(gate, cudaq.DepolarizationChannel(p1))
    noise.add_all_qubit_channel("cx", cudaq.DepolarizationChannel(p2))
    cudaq.set_noise(noise)


def _clear_noise():
    try:
        cudaq.unset_noise()
    except Exception:
        pass


# ── CUDA-Q kernels ─────────────────────────────────────────────────────────────

@cudaq.kernel
def _zz_fm(x: list[float], n_q: int, reps: int):
    """ZZFeatureMap linear entanglement — mirrors Qiskit ZZFeatureMap."""
    qvec = cudaq.qvector(n_q)
    for _ in range(reps):
        for i in range(n_q):
            h(qvec[i])
            rz(2.0 * x[i], qvec[i])
        for i in range(n_q - 1):
            cx(qvec[i], qvec[i + 1])
            rz(2.0 * (_PI - x[i]) * (_PI - x[i + 1]), qvec[i + 1])
            cx(qvec[i], qvec[i + 1])


@cudaq.kernel
def _vqc(x: list[float],
         ry0: list[float], rz0: list[float],
         ry1: list[float], rz1: list[float],
         n_q: int, reps_fm: int):
    """ZZFeatureMap + TwoLocal(ry,rz,cx-linear,reps=1) — mirrors Qiskit ansatz."""
    qvec = cudaq.qvector(n_q)
    # Feature map
    for _ in range(reps_fm):
        for i in range(n_q):
            h(qvec[i])
            rz(2.0 * x[i], qvec[i])
        for i in range(n_q - 1):
            cx(qvec[i], qvec[i + 1])
            rz(2.0 * (_PI - x[i]) * (_PI - x[i + 1]), qvec[i + 1])
            cx(qvec[i], qvec[i + 1])
    # Rotation layer 0
    for i in range(n_q):
        ry(ry0[i], qvec[i])
    for i in range(n_q):
        rz(rz0[i], qvec[i])
    # Linear entanglement
    for i in range(n_q - 1):
        cx(qvec[i], qvec[i + 1])
    # Rotation layer 1
    for i in range(n_q):
        ry(ry1[i], qvec[i])
    for i in range(n_q):
        rz(rz1[i], qvec[i])


# ── Fidelity kernel matrix ─────────────────────────────────────────────────────

def _fidelity_matrix(X1: np.ndarray, X2: np.ndarray, reps: int = 1) -> np.ndarray:
    n_q = X1.shape[1]
    sv1 = [np.array(cudaq.get_state(_zz_fm, x.tolist(), n_q, reps)) for x in X1]
    sv2 = [np.array(cudaq.get_state(_zz_fm, x.tolist(), n_q, reps)) for x in X2]
    K = np.zeros((len(X1), len(X2)))
    for i, s1 in enumerate(sv1):
        for j, s2 in enumerate(sv2):
            if s1.ndim == 1:
                K[i, j] = abs(np.vdot(s1, s2)) ** 2
            else:
                K[i, j] = abs(np.trace(s1 @ s2))
    return K


# ── Quantum kernel SVM ─────────────────────────────────────────────────────────

class _KernelSVC:
    def __init__(self, C: float, reps: int):
        self._C = C
        self._reps = reps
        self._svc = None
        self._X_train = None

    def fit(self, X, y):
        self._X_train = np.asarray(X)
        K = _fidelity_matrix(self._X_train, self._X_train, self._reps)
        self._svc = SVC(kernel="precomputed", C=self._C, random_state=_QSVM_SEED)
        self._svc.fit(K, y)
        return self

    def predict(self, X):
        K = _fidelity_matrix(np.asarray(X), self._X_train, self._reps)
        return self._svc.predict(K)

    def predict_scores(self, X):
        K = _fidelity_matrix(np.asarray(X), self._X_train, self._reps)
        return self._svc.decision_function(K)


# ── VQC classifier ─────────────────────────────────────────────────────────────

def _parity_H(n_q: int):
    h_op = spin.z(0)
    for i in range(1, n_q):
        h_op = h_op * spin.z(i)
    return h_op


class _VQCC:
    """
    VQC binary classifier.
    Output: sign of <Z_0 Z_1 ... Z_{n-1}> (parity observable, same as Qiskit).
    Optimizer: COBYLA (same as Qiskit NeuralNetworkClassifier).
    """

    def __init__(self, n_q: int, reps_fm: int = 1, max_iter: int = 100, seed: int = 12345):
        self._n_q = n_q
        self._reps_fm = reps_fm
        self._max_iter = max_iter
        self._seed = seed
        self._theta = None
        self._H = _parity_H(n_q)
        self._loss_history: list[float] = []

    def _exp(self, xi: np.ndarray, theta: np.ndarray) -> float:
        ry0 = theta[           : self._n_q].tolist()
        rz0 = theta[  self._n_q: 2*self._n_q].tolist()
        ry1 = theta[2*self._n_q: 3*self._n_q].tolist()
        rz1 = theta[3*self._n_q: 4*self._n_q].tolist()
        return cudaq.observe(
            _vqc, self._H,
            xi.tolist(), ry0, rz0, ry1, rz1,
            self._n_q, self._reps_fm,
        ).expectation()

    def fit(self, X, y):
        np.random.seed(self._seed)
        theta0 = np.random.uniform(0, 2 * _PI, 4 * self._n_q)
        self._loss_history = []

        def loss(theta):
            eps = 1e-7
            total = 0.0
            for xi, yi in zip(X, y):
                p1 = float(np.clip((1.0 - self._exp(xi, theta)) / 2.0, eps, 1.0 - eps))
                total -= float(yi) * math.log(p1) + (1.0 - float(yi)) * math.log(1.0 - p1)
            val = total / len(X)
            self._loss_history.append(val)
            return val

        res = minimize(loss, theta0, method="COBYLA",
                       options={"maxiter": self._max_iter, "rhobeg": 0.1})
        self._theta = res.x
        return self

    def predict(self, X):
        return np.array([1 if self._exp(xi, self._theta) < 0.0 else 0
                         for xi in np.asarray(X)])

    def predict_scores(self, X):
        """Expectation values negated so higher score → class 1."""
        return np.array([-self._exp(xi, self._theta) for xi in np.asarray(X)])


# ── Public API (signatures called by runner.py _run_model_cudaq) ──────────────

def evaluate_qsvm_statevector(n_qubits: int, reps_fm: int,
                               X_tr, y_tr, X_te, y_te, **_):
    _use_ideal()
    clf = _KernelSVC(C=_QSVM_C, reps=reps_fm)
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    m = compute_metrics(clf, X_te, y_te)
    m["training_time"] = float(time.time() - t0)
    return m, clf


def evaluate_qsvm_noise_sim(n_qubits: int, reps_fm: int,
                             X_tr, y_tr, X_te, y_te,
                             p1: float = 0.001, p2: float = 0.005, **_):
    _use_noisy(p1, p2)
    clf = _KernelSVC(C=_QSVM_C, reps=reps_fm)
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    m = compute_metrics(clf, X_te, y_te)
    m["training_time"] = float(time.time() - t0)
    _clear_noise()
    return m, clf


def evaluate_qnn_statevector(n_qubits: int, reps_fm: int, reps_ansatz: int,
                              X_tr, y_tr, X_te, y_te, **_):
    _use_ideal()
    clf = _VQCC(n_q=n_qubits, reps_fm=reps_fm, seed=_QSVM_SEED)
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    m = compute_metrics(clf, X_te, y_te)
    m["training_time"] = float(time.time() - t0)
    m["loss_history"] = clf._loss_history
    return m, clf


def evaluate_qnn_noise_sim(n_qubits: int, reps_fm: int, reps_ansatz: int,
                            X_tr, y_tr, X_te, y_te,
                            p1: float = 0.001, p2: float = 0.005, **_):
    _use_noisy(p1, p2)
    clf = _VQCC(n_q=n_qubits, reps_fm=reps_fm, seed=_QSVM_SEED)
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    m = compute_metrics(clf, X_te, y_te)
    m["training_time"] = float(time.time() - t0)
    m["loss_history"] = clf._loss_history
    _clear_noise()
    return m, clf
