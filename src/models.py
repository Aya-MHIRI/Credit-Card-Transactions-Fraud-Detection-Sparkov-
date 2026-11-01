"""Modèles supervisés et non supervisés."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def make_xgb(scale_pos_weight: float = 1.0, n_estimators: int = 300, seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=seed, verbosity=0,
    )


def make_logreg(seed: int = 42):
    """Régression logistique pondérée (class_weight='balanced') sur variables standardisées."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=500, C=1.0, random_state=seed),
    )


class AnomalyScorer:
    """Détecteurs d'anomalies sans étiquettes. `score(X)` : plus la valeur est grande, plus c'est anormal.

    kind : "iforest" | "lof" | "autoencoder"
    L'ajustement n'utilise JAMAIS la cible (approche non supervisée).
    """

    def __init__(self, kind: str, seed: int = 42, max_fit: int | None = None):
        self.kind, self.seed, self.max_fit = kind, seed, max_fit

    def _t(self, X):
        return np.clip(self.scaler_.transform(X), -10, 10)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        if self.max_fit and len(X) > self.max_fit:
            X = X[rng.choice(len(X), self.max_fit, replace=False)]
        if self.kind == "iforest":
            self.model_ = IsolationForest(
                n_estimators=300, max_samples=min(2048, len(X)), contamination="auto",
                n_jobs=-1, random_state=self.seed).fit(X)
        elif self.kind == "lof":
            self.scaler_ = StandardScaler().fit(X)
            self.model_ = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1).fit(self._t(X))
        elif self.kind == "autoencoder":
            self.scaler_ = StandardScaler().fit(X)
            Xs = self._t(X)
            self.model_ = MLPRegressor(
                hidden_layer_sizes=(24, 8, 24), activation="relu", batch_size=512, max_iter=60,
                early_stopping=True, n_iter_no_change=5, random_state=self.seed).fit(Xs, Xs)
        else:
            raise ValueError(self.kind)
        return self

    def score(self, X, batch: int = 50_000) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype="float64")
        for i in range(0, len(X), batch):
            xb = X[i:i + batch]
            if self.kind == "iforest":
                out[i:i + batch] = -self.model_.score_samples(xb)
            elif self.kind == "lof":
                out[i:i + batch] = -self.model_.score_samples(self._t(xb))
            else:
                xs = self._t(xb)
                out[i:i + batch] = ((xs - self.model_.predict(xs)) ** 2).mean(axis=1)
        return out
