"""Scoring d'une transaction (API et application) avec les mêmes variables qu'à l'entraînement."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from .config import AMT, ART_DIR, CARD, DEPLOYED, TIME_COL
from .explain import describe_reason, reason_codes, shap_values_of
from .features import _window_features, engineer, static_features, BEHAVIOR

REQUIRED_TX = [CARD, TIME_COL, AMT, "category", "dob", "lat", "long", "merch_lat", "merch_long"]


class FraudScorer:
    def __init__(self, art_dir: Path = ART_DIR):
        self.art = Path(art_dir)
        self.meta = json.loads((self.art / "metadata.json").read_text(encoding="utf-8"))
        self.features: list[str] = self.meta["features"]
        self.categories: list[str] = self.meta["categories"]
        self.model = joblib.load(self.art / "model_xgb.joblib")
        self.threshold = float(self.meta["models"][DEPLOYED]["threshold"])
        self.explainer = shap.TreeExplainer(self.model)
        self.base_value = float(np.ravel(self.explainer.expected_value)[0])
        self._load_history()

    # ------------------------------------------------------------------ historique des cartes
    def _load_history(self):
        h = np.load(self.art / "history.npz")
        cards, ts, amt = h["cards"], h["ts"], h["amt"]
        edges = np.flatnonzero(np.r_[True, cards[1:] != cards[:-1], True])
        self.hist = {int(cards[b0]): (ts[b0:b1].copy(), amt[b0:b1].copy()) for b0, b1 in zip(edges[:-1], edges[1:])}

    def _behavior(self, card: int, ts: int, amt: float) -> np.ndarray:
        t, a = self.hist.get(int(card), (np.empty(0, "int64"), np.empty(0)))
        return np.array([v[0] for v in _window_features(t, a, [ts], [amt])], dtype="float64")

    def _remember(self, card: int, ts: int, amt: float):
        t, a = self.hist.get(int(card), (np.empty(0, "int64"), np.empty(0)))
        i = int(np.searchsorted(t, ts, side="right"))
        self.hist[int(card)] = (np.insert(t, i, ts), np.insert(a, i, amt))

    # ------------------------------------------------------------------ variables
    def features_of(self, tx: dict) -> pd.DataFrame:
        missing = [k for k in REQUIRED_TX if tx.get(k) is None]
        if missing:
            raise ValueError(f"Champs manquants : {missing}")
        row = dict(tx)
        row.setdefault("gender", "F")
        row.setdefault("city_pop", 10000)
        df1 = pd.DataFrame([row])
        df1[TIME_COL] = pd.to_datetime(df1[TIME_COL])
        df1["dob"] = pd.to_datetime(df1["dob"])
        st = static_features(df1, self.categories)
        ts = int((df1[TIME_COL].iloc[0] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))
        beh = self._behavior(int(row[CARD]), ts, float(row[AMT]))
        for name, v in zip(BEHAVIOR, beh):
            st[name] = v
        return st[self.features].astype("float32")

    # ------------------------------------------------------------------ scoring
    def risk_level(self, p: float, thr: float) -> str:
        return "élevé" if p >= thr else ("moyen" if p >= 0.5 * thr else "faible")

    def score(self, tx: dict, threshold: float | None = None, top_reasons: int = 5,
              explain: bool = True, update_history: bool = False) -> dict:
        thr = self.threshold if threshold is None else float(threshold)
        X = self.features_of(tx)
        p = float(self.model.predict_proba(X)[:, 1][0])
        warnings = []
        if str(tx.get("category")) not in self.categories:
            warnings.append(f"Catégorie inconnue : {tx.get('category')}")
        out = {
            "fraud_probability": p,
            "alert": bool(p >= thr),
            "risk_level": self.risk_level(p, thr),
            "threshold": thr,
            "reason_codes": [],
            "warnings": warnings,
        }
        if explain:
            sv = shap_values_of(self.explainer, X)[0]
            rc = reason_codes(sv, X.iloc[0].to_numpy(), self.features, top=top_reasons)
            for r in rc:
                r["description"] = describe_reason(r)
            out["reason_codes"] = rc
        if update_history:
            ts = int((pd.to_datetime(tx[TIME_COL]) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))
            self._remember(int(tx[CARD]), ts, float(tx[AMT]))
        return out

    def score_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Scoring par lot d'un DataFrame au format Sparkov (l'historique est celui du fichier)."""
        X = engineer(df, self.categories)[self.features]
        return self.model.predict_proba(X)[:, 1]
