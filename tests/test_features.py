"""Vérifie que les variables comportementales sont correctes et sans fuite (passé strict uniquement).

    python -m pytest tests -q
"""
import numpy as np
import pandas as pd

from src.config import AMT, CARD, TIME_COL
from src.demo_data import generate_demo
from src import data as D
from src.features import MAX_GAP_H, behavioral_features, engineer, feature_names, CATEGORIES


def brute(df):
    out = []
    for i, r in df.iterrows():
        h = df[(df[CARD] == r[CARD]) & (df[TIME_COL] < r[TIME_COL])]
        dt = (r[TIME_COL] - h[TIME_COL]).dt.total_seconds()
        n1 = int((dt <= 3600).sum())
        n24 = int((dt <= 86400).sum())
        s24 = float(h.loc[dt <= 86400, AMT].sum())
        h30 = h[dt <= 30 * 86400]
        ratio = r[AMT] / h30[AMT].mean() if len(h30) and h30[AMT].mean() > 0 else 1.0
        gap = min(dt.min() / 3600, MAX_GAP_H) if len(h) else MAX_GAP_H
        out.append([n1, n24, s24, ratio, gap])
    return np.array(out)


def test_behavioral_matches_bruteforce():
    df = D.clean(generate_demo(n_cards=6, days=40, seed=1))
    df = df.sample(frac=1.0, random_state=0)  # ordre quelconque en entrée
    got = behavioral_features(df).loc[df.index].to_numpy()
    want = brute(df)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_no_future_information():
    df = D.clean(generate_demo(n_cards=5, days=30, seed=2))
    full = behavioral_features(df)
    cut = len(df) // 2
    part = behavioral_features(df.iloc[:cut])
    # les variables des premières lignes ne changent pas si on supprime le futur
    np.testing.assert_allclose(full.iloc[:cut].to_numpy(), part.to_numpy())


def test_feature_matrix_columns():
    df = D.clean(generate_demo(n_cards=5, days=20, seed=3))
    X = engineer(df)
    assert list(X.columns) == feature_names(CATEGORIES)
    assert not X.isna().any().any()
