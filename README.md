# 🛡️ AINID — AI Network Intrusion Detector

> A production-style Machine Learning pipeline and web dashboard that classifies
> network traffic as **Normal** or **Attack**, built around the UNSW-NB15
> (and CIC-IDS2017-compatible) intrusion detection datasets.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI/CD](https://img.shields.github/actions/workflow/status/Shapi07/AI-Network-Intrusion-Detector/ci.yml?branch=main&label=CI%2FCD&logo=github)](https://github.com/Shapi07/AI-Network-Intrusion-Detector/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



---

## 📑 Table of Contents

- [🎯 Overview](#-overview)
- [🏗️ Architecture & Project Structure](#️-architecture--project-structure)
- [📊 ML Pipeline & Results](#-ml-pipeline--results)
- [📡 Phase 12 — Real-Time Network Traffic Analysis](#-phase-12--real-time-network-traffic-analysis-experimental-extension)
- [🖥️ Streamlit Dashboard](#️-streamlit-dashboard)
- [⚙️ Installation & Quick Start](#️-installation--quick-start)
- [🧪 Testing & CI/CD](#-testing--cicd)
- [🔮 Future Improvements](#-future-improvements)
- [📜 License](#-license)


---

## 🎯 Overview

**AINID (AI Network Intrusion Detector)** is an end-to-end binary classification
system for **Network Intrusion Detection (NIDS)**. It ingests raw network-flow
records (UNSW-NB15 or CIC-IDS2017 formatted CSVs), runs them through a
reproducible cleaning → feature-engineering → training → evaluation pipeline,
and serves the result through an interactive **Streamlit dashboard** that lets
a user upload traffic, get instant Normal/Attack predictions, and inspect model
performance — no notebook or command line required.

**Problem it solves:** manually triaging network traffic logs for malicious
activity doesn't scale, and signature-based intrusion detection tools miss
novel or subtly disguised attacks. AINID demonstrates how a supervised ML
model — trained on labelled flow-level features (protocol, duration, byte
counts, service, connection state, etc.) — can flag anomalous traffic
automatically, with the full pipeline built to be reproducible, containerized,
and CI-tested rather than a one-off notebook experiment.

**Design goals:**
- **Reproducibility** — every stage (cleaning, encoding, scaling, training) is
  a pure, testable function driven by a single `src/config.py`, not notebook
  cells run out of order.
- **Format flexibility** — auto-detects UNSW-NB15 vs. CIC-IDS2017 column
  signatures so the same pipeline can ingest either dataset family.
- **Inference/training parity** — `predict.py` reuses the exact same
  `preprocessing` and `feature_engineering` functions used at training time,
  so there is no train/serve skew.
- **Usable by non-engineers** — the Streamlit dashboard wraps the pipeline so
  a security analyst can drag-and-drop a CSV and get a verdict, without
  touching Python.
- **Shippable** — Dockerized, with a GitHub Actions workflow that builds the
  image on every push/PR.

<details>
<summary>🇷🇺 Краткое описание на русском</summary>

<br>

**AINID** — это end-to-end ML-система обнаружения сетевых вторжений (NIDS),
классифицирующая сетевой трафик как **Normal** (нормальный) или **Attack**
(атака) на основе датасетов UNSW-NB15 / CIC-IDS2017. Проект решает задачу
автоматической фильтрации подозрительного трафика там, где сигнатурные
системы обнаружения пропускают новые или замаскированные атаки. Пайплайн
полностью воспроизводим: очистка данных → инженерия признаков → обучение
нескольких моделей → выбор лучшей по F1-score → инференс через
Streamlit-дашборд, упакованный в Docker с CI-пайплайном на GitHub Actions.

</details>

---

## 🏗️ Architecture & Project Structure

The pipeline is organized as a linear sequence of independently testable
stages, each owned by one module in `src/`:

```
raw CSV ──▶ preprocessing.py ──▶ feature_engineering.py ──▶ train.py ──▶ models/*.joblib
                  │                       │                     │
                  ▼                       ▼                     ▼
           format detection        one-hot encode +      RandomForest /
           + cleaning report       StandardScaler         LogisticRegression /
                                                            DecisionTree
                                                                  │
                                                                  ▼
                                                           evaluate.py ──▶ reports/*.json
                                                                  │
                                                                  ▼
                                                    predict.py ──▶ src/app.py (Streamlit UI)
```

```
AINID/
│
├── src/                        # Core pipeline package
│   ├── config.py                 # Single source of truth: paths, hyperparameters, logging
│   ├── preprocessing.py          # CSV loading, format auto-detection, cleaning, validation
│   ├── feature_engineering.py    # One-hot encoding, train/test split, StandardScaler, feature-name persistence
│   ├── train.py                  # Trains Random Forest / Logistic Regression / Decision Tree, selects best by F1
│   ├── evaluate.py               # Detailed metrics: accuracy, precision/recall/F1, confusion matrix -> reports/
│   ├── predict.py                # Production inference: reuses training-time preprocessing for train/serve parity
│   ├── utils.py                  # Shared helpers (logging, timing decorator, JSON I/O, dir management)
│   └── app.py                    # Streamlit dashboard — the user-facing entry point
│
├── models/                     # Persisted artifacts (excluded from git, see .gitignore)
│   ├── best_model.joblib          # Winning classifier, selected on weighted F1-score
│   ├── scaler.joblib               # StandardScaler fitted on training data only (no leakage)
│   └── feature_names.joblib       # Exact column order the model expects at inference time
│
├── reports/                    # JSON evaluation reports generated by evaluate.py
│   └── evaluation_metrics.json
│
├── data/
│   ├── raw/                      # Place source dataset CSVs here (UNSW-NB15 / CIC-IDS2017)
│   └── processed/                # Auto-generated cleaned datasets
│
├── scripts/
│   ├── generate_sample_data.py   # Synthetic UNSW-NB15-shaped CSV generator (demo mode / CI, no download needed)
│   └── run_full_test.py          # End-to-end smoke test: preprocessing -> features -> train -> evaluate
│
├── tests/                      # Pytest unit tests
├── notebooks/                  # Exploratory analysis (kept out of the production pipeline)
├── logs/                       # Rotating pipeline logs (ainid.log)
├── predictions/                # Batch prediction CSV outputs (created at runtime)
│
├── .github/workflows/ci.yml    # GitHub Actions: installs deps, builds the Docker image on push/PR
├── Dockerfile                  # Streamlit UI image (python:3.12-slim base)
├── docker-compose.yml          # One-command local deployment with bind-mounted models/reports/data/logs
├── requirements.txt
├── LICENSE                     # MIT
└── README.md
```

**Why this layout:** `src/config.py` centralizes every path and hyperparameter
so no module hard-codes a file location — swapping datasets, tuning
hyperparameters, or relocating the model store is a one-line change. Splitting
`preprocessing` from `feature_engineering` keeps *cleaning* (format-agnostic)
separate from *ML-specific transforms* (encoding/scaling), which is what lets
`predict.py` and `app.py` reuse both stages verbatim at inference time instead
of re-implementing them.

---

## 📊 ML Pipeline & Results

**1. Preprocessing (`preprocessing.py`)**
- Loads CSVs via path or in-memory buffer (chunked reading for large files).
- Auto-detects dataset family by column signature (UNSW-NB15 vs. CIC-IDS2017).
- Strips whitespace, coerces numeric dtypes, handles inf/NaN values, drops
  identifier-only columns (`id`, `attack_cat`), and returns a structured
  cleaning report for logging and the UI.

**2. Feature Engineering (`feature_engineering.py`)**
- One-hot encodes categorical columns (`proto`, `service`, `state`, …) via
  `pd.get_dummies(drop_first=True)`.
- Splits into train/test (80/20, `random_state=42` for reproducibility).
- Fits `StandardScaler` **on the training split only** to prevent data
  leakage, then persists it alongside the exact expected feature-name order
  to `models/` — both are reloaded at inference time so new traffic is
  transformed identically.

**3. Training (`train.py`)**
Three classifiers are trained and compared on the held-out test set, all with
`class_weight="balanced"` to handle the natural class imbalance in intrusion
datasets:

| Model | Key hyperparameters |
|---|---|
| Random Forest | `n_estimators=200`, `min_samples_split=5`, `min_samples_leaf=2` |
| Logistic Regression | `solver="lbfgs"`, `max_iter=1000` |
| Decision Tree | `max_depth=20`, `min_samples_split=5` |

The model with the highest **weighted F1-score** on the test split is selected
automatically and persisted as `models/best_model.joblib`.

**4. Evaluation (`evaluate.py`)**
Computes accuracy, weighted precision/recall/F1, a full confusion matrix, and
the per-class `classification_report`, saved as
`reports/evaluation_metrics.json` — the same file the dashboard reads to
render its metrics tab.

> **On the numbers:** the `evaluation_metrics.json` currently checked into the
> repo was produced by `scripts/run_full_test.py`, an end-to-end **smoke test**
> that runs the full pipeline against a handful of synthetic rows to prove
> every stage works together — it is not a benchmark and its "1.0" scores
> reflect that tiny sample, not real-world performance. For meaningful metrics,
> download the full UNSW-NB15 (or CIC-IDS2017) dataset into `data/raw/` and run
> `python -m src.train` followed by evaluation on the real test split — see
> [Installation & Quick Start](#️-installation--quick-start).

---

## 🖥️ Streamlit Dashboard

`src/app.py` is the user-facing layer on top of the pipeline, served at
**http://localhost:8501**. It provides:

- **Sidebar controls** — upload your own CSV of network traffic, or generate
  and download a synthetic demo CSV on the spot for a no-dataset test drive.
- **🔍 Traffic Analysis tab**
  - Runs the uploaded file through the same `preprocessing` →
    `feature_engineering` → `predict` pipeline used in training (no train/serve
    skew).
  - Shows a traffic summary, a confusion matrix if ground-truth labels are
    present in the upload, and the model's feature-importance ranking.
  - Renders the full prediction table and lets you download it as CSV.
- **📊 Training Report tab**
  - Displays the metrics persisted in `reports/evaluation_metrics.json` from
    the last training run (accuracy, precision/recall/F1, confusion matrix).
  - Re-renders feature importances for the currently loaded model.

Artifacts (`model`, `scaler`, `feature_names`) are loaded once and cached via
`st.cache_resource`, so repeated predictions in the UI don't reload the model
from disk on every interaction.

---

## ⚙️ Installation & Quick Start

### Option A — Docker Compose (recommended)

Requires Docker Engine 24+ with the Compose plugin (`docker compose`, not the
deprecated standalone `docker-compose`).

**⚠️ Train the model before your first build** — `models/*.joblib` files are
intentionally excluded from git (see `.gitignore`) as large binary artifacts.
`docker-compose.yml` bind-mounts `./models` from the host, so if that folder
is empty the app starts but predictions fail until a model exists. Train
locally first:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_full_test.py      # smoke test, or: python -m src.train on a real dataset
```

Then build and run:

```bash
docker compose up --build
```

Open **http://localhost:8501** in your browser.

**Bind-mounted host folders** (results persist across restarts/rebuilds):

| Host folder | Container path | Purpose |
|---|---|---|
| `./models` | `/app/models` | Trained model, scaler, feature names |
| `./reports` | `/app/reports` | Evaluation metrics JSON |
| `./predictions` | `/app/predictions` | Batch prediction CSV outputs |
| `./logs` | `/app/logs` | Pipeline logs |
| `./data` | `/app/data` | Raw / processed datasets |

Retrain inside the running container (new artifacts land on the host
automatically):

```bash
docker compose exec ainid-ui python -m src.train
```

Stop the stack:

```bash
docker compose down
```

### Option B — Local Python

```bash
git clone https://github.com/<your-username>/AINID.git
cd AINID

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place a dataset in data/raw/ (e.g. UNSW_NB15_training-set.csv), or generate a demo one:
python scripts/generate_sample_data.py

# Train and select the best model
python -m src.train

# Launch the dashboard
streamlit run src/app.py
```

Then open **http://localhost:8501**.

---

## 🧪 Testing & CI/CD

- `scripts/run_full_test.py` runs the entire pipeline end-to-end (generate
  data → clean → engineer features → train → evaluate) as a fast smoke test.
- `tests/` holds Pytest unit tests (`pip install -r requirements.txt` includes
  `pytest` and `pytest-cov`).
- `.github/workflows/ci.yml` runs on every push/PR to `main`: installs
  dependencies and does a Docker Buildx dry-run build of the image, catching
  broken dependencies or a broken `Dockerfile` before merge.

```bash
pytest --cov=src tests/
```

---

## 📡 Phase 12 — Real-Time Network Traffic Analysis (Experimental Extension)

> **Notice:** Phase 12 is an **experimental portfolio NIDS extension** that adds live network packet capture and flow aggregation capabilities to the existing AINID system.

### Overview & Architecture
Phase 12 captures real network traffic from local interfaces using **Scapy**, aggregates packets into bidirectional 5-tuple flows, extracts statistical flow features (`duration`, `src_bytes`, `dst_bytes`, `protocol_type`, `service`, `flag`), and routes compatible flow DataFrames directly to the existing AINID inference pipeline (`src/predict.py`).

```
Real Network Traffic ──▶ Scapy Sniffer (src/live_capture.py)
                              │
                              ▼
                 Bidirectional 5-Tuple Aggregation
             (src_ip, dst_ip, src_port, dst_port, proto)
                              │
                              ▼
                   Flow Feature Extraction
             (duration, bytes, pkts, rates, flags)
                              │
                              ▼
                Feature Validation & ML Check
             ├─► [Compatible]   ──▶ src.predict.predict ──▶ Live Monitor Verdict
             └─► [Incompatible] ──► "LIVE MODEL INCOMPATIBLE" Diagnostic Report
```

### Key Components & Features
- **Packet Capture (`Scapy`)**: Configurable capture window (duration timeout and packet limit bounds).
- **Bidirectional 5-Tuple Aggregation**: Sorts IP and port pairs canonically so forward ($A \rightarrow B$) and reverse ($B \rightarrow A$) packets are grouped into a single connection flow.
- **Statistical Feature Derivation**: Computes flow duration (guaranteed $\ge 0$), directional byte/packet totals, rates, average packet sizes, inferred services (`http`, `dns`, `smtp`, `ftp`, `eco_i`), and TCP flag states (`SF`, `S0`, `REJ`).
- **Strict ML Compatibility Verification**: Before running inference, `validate_live_features()` checks whether the model requires lab testbed-specific features (`ct_srv_src`, `ct_dst_ltm`, etc.) that cannot be extracted from live packet flows. If unsupported features are required, the system **rejects fake predictions** and outputs a `LIVE MODEL INCOMPATIBLE` diagnostic report instead of substituting arbitrary zeros.

### Windows Requirements & Permissions
- **Packet Capture Driver**: Requires [Npcap](https://npcap.com/) (or WinPcap) installed in WinPcap-compatible mode.
- **Administrator Privileges**: Raw socket sniffing on Windows requires running the command prompt or terminal with **Administrator privileges**.

### CLI Usage Commands
```bash
# List all available network capture interfaces
python -m src.live_capture --list-interfaces

# Capture live traffic on default interface for 10 seconds (max 1000 packets)
python -m src.live_capture --duration 10 --max-packets 1000

# Capture traffic on a specific interface
python -m src.live_capture --interface "Wi-Fi" --duration 15 --max-packets 500
```

### Security Boundary & Defensive Scope
- **Defensive Monitoring ONLY**: Monitors authorized local interfaces (localhost, user's own machine/Wi-Fi/lab).
- **No Payload Collection**: Extracts transport/network layer metadata and statistical metrics only; never harvests application payloads or credentials.
- **No Offensive Mechanisms**: Contains zero packet injection, modification, or exploit automation capabilities.

### Limitations
- **Distribution Shift**: Live host network traffic patterns differ from offline lab benchmark datasets (UNSW-NB15/CIC-IDS2017).
- **Experimental Classifier**: Live verdicts are provided as experimental alerts for monitoring prototypes, not production SOC firewalls.

---

## 🔮 Future Improvements


- [ ] Train and publish benchmark metrics on the full UNSW-NB15 / CIC-IDS2017
      datasets (current in-repo report is a smoke-test artifact only).
- [ ] Multi-class attack categorization (`attack_cat`) instead of binary
      Normal/Attack classification.
- [ ] Add gradient-boosted models (XGBoost / LightGBM) to the training
      comparison alongside Random Forest / Logistic Regression / Decision Tree.
- [ ] Expose a REST API (FastAPI dependencies are already scaffolded, commented
      out in `requirements.txt`) for programmatic/service-to-service inference
      alongside the Streamlit UI.
- [ ] Hyperparameter tuning via `GridSearchCV` / `Optuna` instead of the fixed
      parameters in `config.py`.
- [ ] Model explainability (SHAP values) in the dashboard, beyond built-in
      feature importances.
- [ ] Push CI further: run `pytest` in the GitHub Actions workflow (currently
      only a Docker build dry-run) and add coverage reporting.
- [ ] Push the built image to a container registry (GHCR/Docker Hub) as a CI
      release step.

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for the full
text.

---

<p align="center">Built as a portfolio-grade demonstration of an end-to-end,
containerized ML pipeline — from raw network flow data to a deployable
detection dashboard.</p>
