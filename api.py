"""API FastAPI de scoring de fraude explicable.

Lancer :  uvicorn api:app --reload        puis ouvrir http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.config import DEPLOYED
from src.serving import FraudScorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scorer = FraudScorer()
    yield


app = FastAPI(
    title="API de détection de fraude explicable",
    description="Score une transaction (jeu Sparkov), indique si elle déclenche une alerte et "
                "renvoie les « codes motifs » (variables qui ont le plus contribué, via SHAP).",
    version="1.0.0",
    lifespan=lifespan,
)


class Transaction(BaseModel):
    cc_num: int = Field(..., description="Identifiant de la carte")
    trans_date_trans_time: datetime = Field(..., description="Date et heure de la transaction")
    amt: float = Field(..., ge=0, description="Montant ($)")
    category: str = Field(..., description="Catégorie du marchand (ex. shopping_net)")
    gender: str = Field("F", description="F ou M")
    dob: date = Field(..., description="Date de naissance du client")
    city_pop: int = Field(10000, ge=0, description="Population de la ville du client")
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    merchant: str | None = None

    model_config = {"json_schema_extra": {"examples": [{
        "cc_num": 2703186189652095, "trans_date_trans_time": "2020-07-02T02:14:00", "amt": 842.5,
        "category": "shopping_net", "gender": "F", "dob": "1988-03-09", "city_pop": 3495,
        "lat": 36.0788, "long": -81.1781, "merch_lat": 36.011293, "merch_long": -82.048315,
        "merchant": "fraud_Rippin, Kub and Mann"}]}}


class Reason(BaseModel):
    feature: str
    label: str
    value: float
    shap: float
    description: str


class Prediction(BaseModel):
    fraud_probability: float
    alert: bool
    risk_level: str
    threshold: float
    reason_codes: list[Reason]
    warnings: list[str]


def scorer(request: Request) -> FraudScorer:
    return request.app.state.scorer


@app.get("/health", tags=["service"])
def health():
    return {"status": "ok"}


@app.get("/model", tags=["service"])
def model_info(request: Request):
    """Modèle déployé, seuil d'alerte, métriques sur le test et variables utilisées."""
    s = scorer(request)
    m = s.meta["models"][DEPLOYED]
    return {
        "dataset": s.meta["dataset"], "demo": s.meta["demo"], "trained_at": s.meta["created"],
        "model": m["label"], "imbalance_strategy": s.meta["imbalance"]["chosen"],
        "threshold": s.threshold, "review_cost": s.meta["review_cost"],
        "test_metrics": {k: m["test"][k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f2", "mcc", "cost")},
        "features": s.features, "known_categories": s.categories,
    }


@app.post("/predict", response_model=Prediction, tags=["scoring"])
def predict(
    tx: Transaction, request: Request,
    threshold: float | None = Query(None, ge=0, le=1, description="Seuil d'alerte (défaut : seuil à coût minimal)"),
    top_reasons: int = Query(5, ge=1, le=10, description="Nombre de codes motifs"),
    update_history: bool = Query(False, description="Ajouter la transaction à l'historique de la carte"),
):
    """Score une transaction. Les variables de vélocité sont calculées à partir de l'historique connu de la carte."""
    try:
        return scorer(request).score(tx.model_dump(), threshold=threshold, top_reasons=top_reasons,
                                     explain=True, update_history=update_history)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", tags=["scoring"])
def predict_batch(
    txs: list[Transaction], request: Request,
    threshold: float | None = Query(None, ge=0, le=1),
    explain: bool = Query(False, description="Ajouter les codes motifs (plus lent)"),
):
    """Score jusqu'à 1000 transactions (traitées dans l'ordre reçu, sans mise à jour de l'historique)."""
    if len(txs) > 1000:
        raise HTTPException(status_code=413, detail="Maximum 1000 transactions par appel.")
    s = scorer(request)
    out = []
    for i, tx in enumerate(txs):
        r = s.score(tx.model_dump(), threshold=threshold, explain=explain)
        r["index"] = i
        out.append(r)
    return {"n": len(out), "n_alerts": sum(r["alert"] for r in out), "results": out}
