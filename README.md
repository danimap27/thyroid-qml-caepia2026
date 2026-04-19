# Thyroid Cancer Recurrence Prediction — XAI + QML (CAEPIA 2026)

Hybrid classical–quantum experimental suite for binary prediction of thyroid
cancer recurrence. Benchmarks a classical RBF-SVM baseline against two
quantum model families (Pegasos QSVC and a variational QNN), each evaluated
in an ideal (statevector) setting and under a realistic IBM-device noise
model, across three feature subsets derived from RULEx association-rule
importance rankings.

## Models

| id              | Description                                                     |
|-----------------|-----------------------------------------------------------------|
| `svm_classical` | RBF SVM (`sklearn.svm.SVC`, `C=1000`) — classical baseline.      |
| `pegasos_ideal` | Pegasos QSVC with `FidelityQuantumKernel` on a noiseless sampler.|
| `pegasos_noise` | Pegasos QSVC with AerSimulator using an IBM backend noise model. |
| `qnn_ideal`     | `SamplerQNN` + `TwoLocal` ansatz, noiseless statevector sampler. |
| `qnn_noise`     | `SamplerQNN` + `TwoLocal` ansatz, AerSimulator with noise model. |

Feature map: `ZZFeatureMap` with linear entanglement, `reps=1`.
Ansatz: `TwoLocal(["ry","rz"], "cx", entanglement="linear", reps=1)`.

## Feature subsets

- `all`  — full 13-feature matrix (clinical + demographic).
- `top9` — nine features retained by RULEx association-rule analysis.
- `top4` — four most-informative features: *Risk*, *Age*, *Gender*, *Smoking*.

## Dataset

`thyroid_clean.xlsx` — preprocessed dataset; categorical variables already
encoded numerically, no missing values, target column is binary
(`Recurred`, ~30% positive rate).

## Repository layout

```
thyroid-qml-caepia2026/
├── core/                  # Hercules framework (HUB, SLURM template, table generator)
│   ├── manager.py
│   ├── slurm_generic.sh
│   ├── generate_tables.py
│   └── deploy.sh
├── tools.py               # Quantum model evaluators and feature maps
├── runner.py              # Per-run entry point (one model × subset × seed)
├── config.yaml            # Sweep configuration (models, subsets, phases, tables)
├── thyroid_clean.xlsx     # Dataset
├── requirements.txt
└── README.md
```

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Noise-model modes require an IBM Quantum token:
export IBM_QUANTUM_TOKEN="..."

# Inspect the sweep:
python runner.py --dry-run

# Run a single combination:
python runner.py --model pegasos_ideal --subset top4 --seed 12345

# Run the full sweep locally:
python runner.py
```

### GPU requirements

Noisy Aer simulations run on GPU via `qiskit-aer-gpu==0.16.0`, which requires
a CUDA 12 runtime. On a system without an NVIDIA GPU, replace that line in
`requirements.txt` with `qiskit-aer==0.16.0` and set `use_gpu: false` in
`config.yaml` to fall back to CPU execution with `matrix_product_state`.

## Running on Hercules (CICA SLURM cluster)

```bash
# 1. Clone onto Hercules
git clone https://github.com/danimap27/thyroid-qml-caepia2026.git
cd thyroid-qml-caepia2026

# 2. Set up the conda environment (once)
module load Miniconda3
conda create -n thyroid-qml python=3.10 -y
source activate thyroid-qml
pip install -r requirements.txt
# Hercules gpu partition provides CUDA 12, which qiskit-aer-gpu requires.

# 3. Launch the interactive HUB
python core/manager.py
#   [R] → export SLURM command files from config.yaml
#   [F] → submit Phase 1 (ideal) → Phase 2 (noise) as dependent array jobs
#   [T] → generate LaTeX tables from ./results/*/results.csv
```

The `ideal` phase is submitted first; the `noise` phase starts automatically
once `ideal` completes.

## Output layout

```
results/
├── svm_classical__all__s12345/
│   ├── results.csv
│   └── metrics.json
├── pegasos_ideal__top4__s12345/
│   ├── results.csv
│   └── metrics.json
└── ...
```

Each `results.csv` records: accuracy, macro precision/recall/F1, clinical
sensitivity and specificity (class 1 = *Recurred*), training time,
inference time, total wall time, run metadata, and (for failures) the
exception message.

## Reproducibility

- Global seed `12345` applied to `numpy`, `random`, and
  `qiskit_algorithms.utils.algorithm_globals`.
- Fixed 70/30 train–test split (`random_state=12345`).
- Min–max feature scaling to the range $[0, \pi]$ before quantum encoding.
- All dependency versions pinned in `requirements.txt`.

## Metrics of interest

Sensitivity is prioritised for the clinical discussion: false negatives
(missed recurrences) are the dominant cost in this task. Accuracy and
specificity are reported jointly; training time is recorded to quantify the
overhead of the noisy-simulation variants relative to the ideal
(statevector) variants.
