"""Stratégies contre le déséquilibre. Elles ne sont appliquées QU'AU jeu d'entraînement."""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGIES = {
    "aucune": "Aucune (référence)",
    "poids": "Pondération des classes (scale_pos_weight)",
    "sous_echantillonnage": "Sous-échantillonnage aléatoire (10 normales : 1 fraude)",
    "sur_echantillonnage": "Sur-échantillonnage aléatoire (1 fraude : 10 normales)",
    "smote": "SMOTE (1 fraude synthétique : 10 normales)",
}


def apply_strategy(X: pd.DataFrame, y: np.ndarray, strategy: str, seed: int = 42, ratio: float = 0.1):
    """Retourne (X', y', scale_pos_weight) pour la stratégie demandée."""
    y = np.asarray(y).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)

    if strategy == "aucune":
        return X, y, 1.0
    if strategy == "poids":
        return X, y, float(len(neg) / max(len(pos), 1))
    if strategy == "sous_echantillonnage":
        keep_neg = rng.choice(neg, size=min(len(neg), int(len(pos) / ratio)), replace=False)
        idx = rng.permutation(np.concatenate([pos, keep_neg]))
        return X.iloc[idx], y[idx], 1.0
    if strategy == "sur_echantillonnage":
        target = int(ratio * len(neg))
        extra = rng.choice(pos, size=max(target - len(pos), 0), replace=True)
        idx = rng.permutation(np.concatenate([neg, pos, extra]))
        return X.iloc[idx], y[idx], 1.0
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE

        k = max(1, min(5, len(pos) - 1))
        Xr, yr = SMOTE(sampling_strategy=ratio, k_neighbors=k, random_state=seed).fit_resample(X, y)
        return Xr, np.asarray(yr).astype(int), 1.0
    raise ValueError(f"Stratégie inconnue : {strategy}")
