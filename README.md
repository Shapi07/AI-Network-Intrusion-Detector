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

*Full documentation coming in Phase 11.*
