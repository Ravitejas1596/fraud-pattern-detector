# 🕵️ Fraud Pattern Detector

> *Because fraud doesn't look like one bad transaction — it looks like a pattern.*

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)
![HuggingFace](https://img.shields.io/badge/Live%20Demo-HuggingFace-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Open%20on%20HuggingFace-yellow?style=for-the-badge)](https://huggingface.co/spaces/ravitejas1596/fraud-pattern-detector)

---

## 💡 Why I Built This

Most fraud detection projects stop at the model.

They train a classifier, hit 99% accuracy on an imbalanced dataset,
and call it done — never stopping to ask *why* a transaction was
flagged, or *what network of activity* surrounds it.

Real-world fraud doesn't happen in isolation. A fraudster doesn't
make one suspicious transaction. They make dozens — linked by shared
email addresses, device fingerprints, IP clusters, and card patterns.
The signal isn't in the transaction. It's in the **network around it.**

I built this project to go beyond the standard ML pipeline and explore
two things that production fraud systems actually care about:

1. **Explainability** — not just *is this fraud?* but *why did the model think so?*
2. **Network patterns** — visualizing clusters of transactions linked
   by shared identifiers to surface rings of suspicious activity

Trained on Kaggle's IEEE-CIS Fraud Detection dataset — one of the
most realistic public fraud datasets available, with 590k transactions
and 400+ features from a real e-commerce platform.

---

## ✨ What It Does

### 🔍 Tab 1 — Single Transaction Scorer
Submit a transaction and get an instant fraud probability score with
a breakdown of the top features driving the decision.

- Real-time scoring via FastAPI `/score` endpoint
- SHAP-style feature importance — shows exactly which fields
  pushed the score up or down
- Clean risk indicator: Low / Medium / High / Critical
- Works on any transaction format matching the IEEE-CIS schema

### 🕸️ Tab 2 — Network Visualization
See the bigger picture. Visualize clusters of transactions grouped
by shared identifiers — card numbers, email domains, device IDs,
IP addresses — and watch fraud rings emerge.

- Graph-based view of transaction clusters
- Flagged nodes highlighted in red
- Cluster-level risk scoring — if one node is fraud, the whole
  cluster gets elevated risk
- Hover over any node to see transaction details

---

## 🧠 How It Works
```
Raw Transaction Data (IEEE-CIS)
          │
          ▼
┌─────────────────────────┐
│   Feature Engineering   │
│  - Frequency encoding   │
│  - Time-based features  │
│  - Interaction terms    │
│  - Network aggregates   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    XGBoost Classifier   │
│  - Trained on 200k rows │
│  - Handles imbalance    │
│  - AUC-optimized        │
└────────────┬────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌───────────────┐
│  Score   │  │ SHAP Drivers  │
│  0–1     │  │ Top features  │
│  risk    │  │ per tx        │
└──────────┘  └───────────────┘
             │
             ▼
┌─────────────────────────┐
│   Network Graph Build   │
│  - Group by shared IDs  │
│  - Flag connected nodes │
│  - Score clusters       │
└─────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│  FastAPI /score endpoint│
│  + Streamlit UI         │
│  + Docker container     │
│  + HuggingFace Spaces   │
└─────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Model** | XGBoost | Gradient boosted classifier, handles tabular data + imbalance |
| **Explainability** | SHAP | Per-transaction feature importance drivers |
| **API** | FastAPI | High-performance inference endpoint `/score` |
| **UI** | Streamlit | Two-tab interactive interface |
| **Network Graph** | NetworkX + PyVis | Transaction cluster visualization |
| **Data** | IEEE-CIS (Kaggle) | 590k real e-commerce transactions |
| **Containerization** | Docker | Reproducible deployment |
| **Hosting** | Hugging Face Spaces | Free live demo deployment |
| **Process Manager** | Supervisord | Runs FastAPI + Streamlit in one container |

---

## 📊 Dataset

**IEEE-CIS Fraud Detection** — Kaggle Competition Dataset

| Property | Value |
|---|---|
| Total transactions | ~590,000 |
| Fraud rate | ~3.5% (highly imbalanced) |
| Features | 400+ (transaction + identity) |
| Source | Vesta Corporation (real e-commerce) |
| Training rows used | 200,000 |

Key feature groups: transaction amount, product type, card info,
address, distance, email domain, device type, browser, and 300+
anonymized Vesta-engineered features (V1–V339).

See [DATA_CARD.md](DATA_CARD.md) for full data documentation.
See [MODEL_CARD.md](MODEL_CARD.md) for model performance and limitations.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Kaggle account with IEEE-CIS competition rules accepted
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/Ravitejas1596/fraud-pattern-detector.git
cd fraud-pattern-detector

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Download Data

Place your Kaggle credentials at `~/.kaggle/kaggle.json`, then:
```bash
python -m scripts.download_data
```

> Make sure you've accepted the competition rules at
> kaggle.com/competitions/ieee-fraud-detection before downloading.

### Train the Model
```bash
python -m scripts.train_xgb --seed 42 --max_rows 200000
```

This trains the XGBoost baseline and saves artifacts to `./artifacts/`.

### Run Locally

Open two terminals:
```bash
# Terminal 1 — FastAPI inference server
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit UI
streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0
```

Open [http://localhost:7860](http://localhost:7860)

### Run with Docker
```bash
# Build image
docker build -t fraud-detector .

# Run container
docker run -p 7860:7860 fraud-detector
```

---

## 📁 Project Structure
```
fraud-pattern-detector/
├── app/
│   ├── api.py                 # FastAPI /score endpoint
│   └── streamlit_app.py       # Two-tab Streamlit UI
│
├── scripts/
│   ├── download_data.py       # Kaggle data downloader
│   └── train_xgb.py           # XGBoost training pipeline
│
├── artifacts/                 # Saved model + encoders (post-training)
├── reports/                   # EDA + evaluation reports
│
├── DATA_CARD.md               # Dataset documentation
├── MODEL_CARD.md              # Model performance + limitations
├── Dockerfile                 # Docker build (HuggingFace compatible)
├── supervisord.conf           # Runs FastAPI + Streamlit together
└── requirements.txt
```

---

## 🌐 Live Demo

**Try it → [https://huggingface.co/spaces/ravitejas1596/fraud-pattern-detector](https://huggingface.co/spaces/ravitejas1596/fraud-pattern-detector)**

> No setup needed. Score a transaction and explore the network graph live.

---

## 🗺️ Roadmap

- [x] XGBoost baseline with feature engineering
- [x] SHAP-style per-transaction explanations
- [x] Network graph visualization
- [x] FastAPI inference endpoint
- [x] Docker + HuggingFace Spaces deployment
- [ ] LightGBM / CatBoost comparison
- [ ] Real-time streaming simulation
- [ ] Graph Neural Network (GNN) approach
- [ ] Alert system for high-risk clusters
- [ ] REST API authentication + rate limiting

---

## ⚠️ Limitations

- Trained on a single dataset — may not generalize to all fraud types
- Network graph is lightweight — not a full graph database
- Model is a baseline — production systems require continuous retraining
- See [MODEL_CARD.md](MODEL_CARD.md) for full details

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Kaggle + Vesta Corporation](https://www.kaggle.com/competitions/ieee-fraud-detection) — for the dataset
- [XGBoost](https://xgboost.readthedocs.io) — for the model
- [SHAP](https://shap.readthedocs.io) — for explainability
- [Hugging Face](https://huggingface.co) — for free model hosting

---

<p align="center">
  Built for anyone who believes catching fraud 
  means understanding the pattern, not just the transaction.
</p>
