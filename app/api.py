from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.model import FraudModel


ARTIFACTS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"

app = FastAPI(title="Fraud Pattern Detector", version="1.0.0")
engine: Optional[FraudModel] = None


class Transaction(BaseModel):
    TransactionAmt: float = Field(..., ge=0)
    ProductCD: Optional[str] = None
    card1: Optional[str] = None
    card2: Optional[str] = None
    card3: Optional[str] = None
    card5: Optional[str] = None
    card6: Optional[str] = None
    addr1: Optional[str] = None
    addr2: Optional[str] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None

    # C and D style numeric features (optional)
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    C7: Optional[float] = None
    C8: Optional[float] = None
    C9: Optional[float] = None
    C10: Optional[float] = None
    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C14: Optional[float] = None
    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D5: Optional[float] = None
    D10: Optional[float] = None
    D11: Optional[float] = None
    D15: Optional[float] = None


@app.on_event("startup")
def _load() -> None:
    global engine
    if ARTIFACTS_PATH.exists():
        engine = FraudModel(ARTIFACTS_PATH)
    else:
        engine = None


@app.get("/health")
def health() -> Dict[str, Any]:
    ok = ARTIFACTS_PATH.exists()
    return {"ok": ok, "artifacts_path": str(ARTIFACTS_PATH)}


@app.post("/score")
def score(tx: Transaction) -> Dict[str, Any]:
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Train first (python -m scripts.train_xgb) to create artifacts/model.joblib.",
        )
    res = engine.score(tx.model_dump(), top_k=8)
    return {
        "probability_fraud": res.probability,
        "decision": res.decision,
        "top_factors": res.top_factors,
    }

