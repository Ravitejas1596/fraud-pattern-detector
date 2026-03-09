## Model Card — Fraud Pattern Detector

### Model details
- **Task**: binary classification (fraud vs non-fraud)
- **Baseline model**: XGBoost with compact feature set + frequency encoding for selected categorical fields
- **Explainability**: SHAP TreeExplainer (top driver contributions)

### Intended use
- Educational/demo fraud scoring and pattern exploration.
- Not for production blocking decisions without proper monitoring, retraining, and policy controls.

### Training data
- Source: Kaggle “IEEE-CIS Fraud Detection” competition dataset (user must accept Kaggle rules).
- Target: `isFraud`

### Metrics (holdout split)
See `artifacts/metrics.json`.

### Limitations
- Uses a simplified feature set to keep the demo light and portable.
- The “network patterns” tab is illustrative of cluster behavior; it is not the full dataset graph.

