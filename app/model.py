from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import shap


@dataclass(frozen=True)
class ScoreResult:
    probability: float
    decision: str
    top_factors: List[Dict[str, Any]]


class FraudModel:
    def __init__(self, artifacts_path: Path):
        payload = joblib.load(artifacts_path)
        self.model = payload["model"]
        self.imputer = payload["imputer"]
        self.feature_names = list(payload["feature_names"])
        self.freq_mappings = payload["freq_mappings"]
        self.categorical_cols = list(payload.get("categorical_cols") or [])

        self.explainer = shap.TreeExplainer(self.model)

    def _decision(self, p: float) -> str:
        return "ALLOW" if p < 0.2 else "REVIEW" if p < 0.6 else "BLOCK"

    def _freq_encode_one(self, row: Dict[str, Any]) -> np.ndarray:
        vals: List[float] = []
        for col in self.feature_names:
            v = row.get(col, None)
            if col in self.categorical_cols:
                mapping = self.freq_mappings.get(col, {})
                key = "NA" if v is None else str(v)
                vals.append(float(mapping.get(key, 0)))
            else:
                vals.append(np.nan if v is None else float(v))

        X = np.array(vals, dtype=float).reshape(1, -1)
        X = self.imputer.transform(X)
        return X

    def score(self, row: Dict[str, Any], top_k: int = 8) -> ScoreResult:
        X = self._freq_encode_one(row)
        p = float(self.model.predict_proba(X)[:, 1][0])

        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = np.array(shap_vals).reshape(-1)

        order = np.argsort(np.abs(shap_vals))[::-1][:top_k]
        top = []
        for idx in order:
            top.append(
                {
                    "feature": self.feature_names[int(idx)],
                    "contribution": float(shap_vals[int(idx)]),
                    "abs_contribution": float(abs(shap_vals[int(idx)])),
                }
            )

        return ScoreResult(probability=p, decision=self._decision(p), top_factors=top)

