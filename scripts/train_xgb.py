from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


TARGET = "isFraud"


def _read_joined(raw_dir: Path, max_rows: int | None) -> pd.DataFrame:
    tx = pd.read_csv(raw_dir / "train_transaction.csv", nrows=max_rows)
    ident_path = raw_dir / "train_identity.csv"
    if ident_path.exists():
        ident = pd.read_csv(ident_path, nrows=max_rows)
        df = tx.merge(ident, on="TransactionID", how="left")
    else:
        df = tx
    return df


def _select_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # Compact, stable set (keeps model lightweight + portable)
    numeric = [
        "TransactionAmt",
        "dist1",
        "dist2",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D10",
        "D11",
        "D15",
    ]
    categorical = [
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
    ]

    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    keep = [TARGET] + numeric + categorical
    keep = [c for c in keep if c in df.columns]

    out = df[keep].copy()
    return out, numeric, categorical


def _freq_encode(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[Any, int]]]:
    mappings: Dict[str, Dict[Any, int]] = {}
    tr = train.copy()
    te = test.copy()

    for c in cols:
        vc = tr[c].astype("string").fillna("NA").value_counts(dropna=False)
        mapping = vc.to_dict()
        mappings[c] = mapping

        tr[c] = tr[c].astype("string").fillna("NA").map(mapping).fillna(0).astype(float)
        te[c] = te[c].astype("string").fillna("NA").map(mapping).fillna(0).astype(float)

    return tr, te, mappings


@dataclass
class Artifacts:
    model: XGBClassifier
    imputer: SimpleImputer
    feature_names: List[str]
    freq_mappings: Dict[str, Dict[Any, int]]
    categorical_cols: List[str]


def train(df: pd.DataFrame, numeric: List[str], categorical: List[str], seed: int) -> Tuple[Artifacts, Dict[str, Any]]:
    df = df.dropna(subset=[TARGET]).copy()
    y = df[TARGET].astype(int).to_numpy()
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    X_train_enc, X_test_enc, mappings = _freq_encode(X_train, X_test, categorical)

    feature_names = list(X_train_enc.columns)
    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(X_train_enc)
    Xte = imputer.transform(X_test_enc)

    model = XGBClassifier(
        n_estimators=900,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=seed,
    )
    model.fit(Xtr, y_train)

    proba = model.predict_proba(Xte)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "n_test": int(len(y_test)),
        "seed": int(seed),
    }

    artifacts = Artifacts(
        model=model,
        imputer=imputer,
        feature_names=feature_names,
        freq_mappings=mappings,
        categorical_cols=categorical,
    )
    return artifacts, metrics


def save(artifacts: Artifacts, metrics: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": artifacts.model,
            "imputer": artifacts.imputer,
            "feature_names": artifacts.feature_names,
            "freq_mappings": artifacts.freq_mappings,
            "categorical_cols": artifacts.categorical_cols,
        },
        out_dir / "model.joblib",
    )
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rows", type=int, default=200000)
    p.add_argument("--raw_dir", type=str, default=str(Path("data/raw")))
    p.add_argument("--out_dir", type=str, default=str(Path("artifacts")))
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    df = _read_joined(raw_dir, max_rows=args.max_rows if args.max_rows > 0 else None)
    df, numeric, categorical = _select_columns(df)

    artifacts, metrics = train(df, numeric, categorical, seed=args.seed)
    save(artifacts, metrics, Path(args.out_dir))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
