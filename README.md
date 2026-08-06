# 🛡️ AI Network Intrusion Detector

> A production-quality Machine Learning–based Network Intrusion Detection
> System (NIDS) that classifies network traffic as **Normal** or **Attack**
> using the UNSW-NB15 / CIC-IDS2017 dataset.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Project Status

> 🚧 **Under active development** — README will be completed in Phase 11.

---

## 📁 Folder Structure

```
AI-Network-Intrusion-Detector/
│
├── data/
│   ├── raw/            ← Place dataset CSV files here
│   └── processed/      ← Auto-generated cleaned datasets
│
├── models/             ← Saved .joblib model artefacts
├── notebooks/          ← Exploratory notebooks
├── tests/              ← Pytest unit tests
│
├── src/
│   ├── config.py       ← Central configuration
│   ├── utils.py        ← Shared helpers
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── visualization.py
│   └── app.py          ← Streamlit dashboard
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🐳 Running with Docker

Requires Docker Engine 24+ with the Compose plugin (`docker compose`, not the
deprecated standalone `docker-compose`).

### ⚠️ First-time setup — train the model before building

Trained model artifacts (`models/*.joblib`) are intentionally excluded from
git — see `.gitignore` — because they're large binary files that don't
belong in version control. `docker-compose.yml` mounts `./models` as a
volume, so the container simply uses whatever is on your **host**. If that
folder is empty, the app will build and start fine, but the first
prediction will fail with:

```
FileNotFoundError: Missing model artifact(s): [...]. Run training (Phase 5) first.
```

So before your very first `docker compose up`, train the model **locally**
so `models/best_model.joblib`, `models/scaler.joblib`, and
`models/feature_names.joblib` exist on the host:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_full_test.py      # or: python -m src.train
```

### Build and run

```bash
docker compose up --build
```

Then open **http://localhost:8501** in your browser.

### What gets mounted

These host folders are bind-mounted into the container so results persist
across restarts and rebuilds:

| Host folder     | Container path      | Purpose                                |
|------------------|----------------------|-----------------------------------------|
| `./models`       | `/app/models`        | Trained model, scaler, feature names    |
| `./reports`      | `/app/reports`       | Evaluation metrics (Phase 6)            |
| `./predictions`  | `/app/predictions`   | Batch prediction CSVs (Phase 7)         |
| `./logs`         | `/app/logs`          | Pipeline logs                           |
| `./data`         | `/app/data`          | Raw / processed datasets                |

### Retraining inside the container (optional)

Instead of training on the host, you can also retrain inside the already
running container — the new artifacts land in the mounted `./models`
folder on the host automatically:

```bash
docker compose exec ainid-ui python -m src.train
```

### Stopping

```bash
docker compose down
```

Everything above is a bind mount rather than a named Docker volume, so your
models, reports, predictions, and logs stay on the host either way.

---

*Full documentation coming in Phase 11.*
