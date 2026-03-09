from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scripts.train_xgb import TARGET, _read_joined, _select_columns, _freq_encode


def _group_metrics(y: np.ndarray, p: np.ndarray, group: pd.Series) -> pd.DataFrame:
    rows = []
    group = group.fillna("missing").astype(str)
    group = group.reset_index(drop=True)
    for g in group.unique().tolist():
        mask = (group == g).to_numpy()
        n = int(mask.sum())
        if n < 500:
            continue
        yy = y[mask]
        pp = p[mask]
        auc = float("nan")
        if len(np.unique(yy)) == 2:
            auc = float(roc_auc_score(yy, pp))
        rows.append(
            {
                "group": str(g),
                "n": n,
                "fraud_rate": float(np.mean(yy)),
                "mean_score": float(np.mean(pp)),
                "roc_auc": auc,
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "data" / "raw"
    artifacts_path = repo_root / "artifacts" / "model.joblib"
    out_dir = repo_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_joined(raw_dir, max_rows=200000)
    df, numeric, categorical = _select_columns(df)
    df = df.dropna(subset=[TARGET]).copy()

    y = df[TARGET].astype(int).to_numpy()
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_test = X_test.reset_index(drop=True)

    payload = joblib.load(artifacts_path)
    model = payload["model"]
    imputer = payload["imputer"]
    cat_cols = list(payload.get("categorical_cols") or [])

    X_train_enc, X_test_enc, _ = _freq_encode(X_train, X_test, cat_cols)
    Xte = imputer.transform(X_test_enc)
    p = model.predict_proba(Xte)[:, 1]

    product = X_test.get("ProductCD", pd.Series([None] * len(X_test)))
    device = X_test.get("DeviceType", pd.Series([None] * len(X_test)))

    product_table = _group_metrics(y_test, p, product)
    device_table = _group_metrics(y_test, p, device)

    md = []
    md.append("## Bias audit (limited slices)\n")
    md.append(
        "This audit checks basic performance skews across a couple of available categorical slices. It is not a substitute for a full fairness review.\n"
    )
    md.append("### Slice: ProductCD\n")
    md.append(product_table.to_markdown(index=False) if not product_table.empty else "_Not available._")
    md.append("\n\n### Slice: DeviceType\n")
    md.append(device_table.to_markdown(index=False) if not device_table.empty else "_Not available._")
    md.append("\n")

    out_path = out_dir / "bias_audit.md"
    out_path.write_text("\n".join(md))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

