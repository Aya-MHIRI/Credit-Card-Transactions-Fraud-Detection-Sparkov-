"""Générateur de données de DÉMONSTRATION au format Sparkov.

Sert uniquement à tester le pipeline sans télécharger Kaggle. Les résultats obtenus avec ces données
n'ont AUCUNE valeur scientifique : pour le rendu, entraîne sur le vrai jeu (fraudTrain/fraudTest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import CATEGORIES

STATES = ["NY", "CA", "TX", "FL", "PA", "OH", "IL", "MI", "GA", "NC", "VA", "WA", "MO", "MN", "AL"]
MERCHANTS = [f"fraud_Store {i:02d}" for i in range(40)]
# poids des catégories pour les transactions normales / frauduleuses (ordre de CATEGORIES)
P_NORMAL = np.array([.07, .07, .10, .09, .10, .06, .10, .08, .03, .07, .05, .04, .09, .05])
P_FRAUD = np.array([.03, .02, .06, .03, .20, .02, .03, .02, .15, .06, .02, .26, .07, .03])
MU_CAT = np.array([3.6, 3.3, 3.8, 3.4, 3.9, 3.4, 4.0, 3.6, 3.5, 3.4, 3.4, 3.7, 3.8, 4.1])
HOUR_P = np.array([.4, .3, .2, .2, .3, .6, 1.5, 3.0, 4.5, 5.5, 6.0, 6.0, 6.5, 6.5, 6.0, 5.5, 5.5, 5.5, 5.5, 5.0, 4.0, 2.5, 1.2, .7])
HOUR_P = HOUR_P / HOUR_P.sum()


def generate_demo(n_cards: int = 400, days: int = 365, seed: int = 7, start: str = "2019-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    cc = rng.permutation(np.unique(rng.integers(10**15, 10**16, n_cards * 2, dtype="int64")))[:n_cards]
    lat0, lon0 = rng.uniform(28, 48, n_cards), rng.uniform(-122, -72, n_cards)
    gender = rng.choice(["F", "M"], n_cards)
    age = rng.integers(19, 85, n_cards)
    city_pop = (np.exp(rng.normal(8.5, 1.8, n_cards)) + 50).astype(int)
    state = rng.choice(STATES, n_cards)
    rate = rng.gamma(2.0, 0.28, n_cards)  # transactions / jour
    scale = np.exp(rng.normal(0, 0.35, n_cards))

    rows = []
    # --- transactions normales
    for i in range(n_cards):
        n = rng.poisson(rate[i] * days)
        day = rng.integers(0, days, n)
        hour = rng.choice(24, n, p=HOUR_P)
        sec = day * 86400 + hour * 3600 + rng.integers(0, 3600, n)
        cat = rng.choice(len(CATEGORIES), n, p=P_NORMAL / P_NORMAL.sum())
        amt = np.clip(np.exp(rng.normal(MU_CAT[cat] + np.log(scale[i]), 0.75)), 1.0, 4000.0)
        rows.append(pd.DataFrame({"card": i, "sec": sec, "cat": cat, "amt": amt, "fraud": 0}))
    # --- épisodes de fraude (rafales de quelques transactions)
    n_ep = int(n_cards * days / 365 * 0.62)
    for _ in range(n_ep):
        i = rng.integers(0, n_cards)
        day = rng.integers(0, days)
        night = rng.random() < 0.8
        h0 = rng.choice([22, 23, 0, 1, 2, 3]) if night else rng.integers(8, 20)
        k = int(np.clip(rng.geometric(0.33), 1, 7))
        gaps = rng.exponential(25 * 60, k).cumsum()
        sec = day * 86400 + h0 * 3600 + rng.integers(0, 3600) + gaps.astype(int)
        cat = rng.choice(len(CATEGORIES), k, p=P_FRAUD / P_FRAUD.sum())
        disguised = rng.random(k) < 0.2  # fraudes « camouflées » : montant banal
        amt = np.where(disguised,
                       np.exp(rng.normal(MU_CAT[cat] + np.log(scale[i]) + 0.4, 0.6)),
                       np.exp(rng.normal(6.0, 0.6, k)))
        rows.append(pd.DataFrame({"card": i, "sec": sec, "cat": cat, "amt": np.clip(amt, 1.0, 6000.0), "fraud": 1}))

    t = pd.concat(rows, ignore_index=True).sort_values("sec", kind="stable").reset_index(drop=True)
    c = t["card"].to_numpy()
    ts = t0 + pd.to_timedelta(t["sec"], unit="s")
    n = len(t)
    merch_lat = lat0[c] + rng.normal(0, 0.45, n)
    merch_lon = lon0[c] + rng.normal(0, 0.45, n)
    dob = np.asarray((t0 - pd.to_timedelta(age * 365.25, unit="D")).strftime("%Y-%m-%d"))
    df = pd.DataFrame({
        "Unnamed: 0": np.arange(n),
        "trans_date_trans_time": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": cc[c],
        "merchant": np.array(MERCHANTS)[rng.integers(0, len(MERCHANTS), n)],
        "category": np.array(CATEGORIES)[t["cat"].to_numpy()],
        "amt": t["amt"].round(2),
        "first": "Demo", "last": "Client", "gender": gender[c], "street": "1 Demo street",
        "city": "Demoville", "state": state[c], "zip": 10000,
        "lat": lat0[c].round(4), "long": lon0[c].round(4), "city_pop": city_pop[c],
        "job": "Demo job", "dob": dob[c],
        "trans_num": [f"{x:032x}" for x in rng.integers(0, 2**62, n)],
        "unix_time": (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1),
        "merch_lat": merch_lat.round(6), "merch_long": merch_lon.round(6),
        "is_fraud": t["fraud"].to_numpy(),
    })
    return df
