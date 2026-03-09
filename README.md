---
title: fraud-pattern-detector
emoji: 🕵️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

## Fraud Pattern Detector (XGBoost + Network Patterns) — FastAPI + Streamlit

Fraud detection demo trained on Kaggle's **IEEE-CIS Fraud Detection** dataset.

### What you get

- XGBoost baseline with compact feature engineering
- Per-transaction explanations (SHAP-style drivers)
- Lightweight transaction-network view (cluster patterns around shared identifiers)
- FastAPI inference service (`/score`)
- Streamlit UI with 2 tabs:
  - Single transaction scorer
  - Network visualization with flagged clusters
- Docker image compatible with Hugging Face Spaces

### Local setup

Requires Python **3.11+**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download data (Kaggle)

1. Place your Kaggle credentials at `~/.kaggle/kaggle.json` (recommended), or ensure a `kaggle.json` exists in your workspace root.
2. Ensure you've accepted the competition rules on Kaggle.

```bash
python -m scripts.download_data
```

### Train (XGBoost baseline)

```bash
python -m scripts.train_xgb --seed 42 --max_rows 200000
```

### Run locally

Terminal 1:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0
```

