"""Construction des variables, sans fuite de données.

Toutes les variables comportementales ne regardent que des transactions STRICTEMENT antérieures
de la même carte. Le même code (`_window_features`) sert à l'entraînement (par lot) et à l'API
(transaction unique + historique) pour éviter tout décalage entre entraînement et production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AMT, CARD, TIME_COL

CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
    "shopping_net", "shopping_pos", "travel",
]
MAX_GAP_H = 720.0  # 30 jours : plafond / valeur si aucune transaction précédente

STATIC_BASE = [
    "amt", "log_amt", "hour_sin", "hour_cos", "is_night", "dow", "is_weekend",
    "age", "gender_f", "log_city_pop", "distance_km",
]
BEHAVIOR = ["n_tx_1h", "n_tx_24h", "amt_sum_24h", "amt_ratio_30d", "hours_since_prev"]


def feature_names(categories=CATEGORIES):
    return STATIC_BASE + BEHAVIOR + [f"cat_{c}" for c in categories]


# nom -> (libellé court, description, groupe)
FEATURE_INFO = {
    "amt": ("Montant", "Montant de la transaction ($)", "Montant"),
    "log_amt": ("Log du montant", "log(1 + montant), réduit l'asymétrie", "Montant"),
    "hour_sin": ("Heure (sin)", "Codage cyclique de l'heure : sin(2π·h/24)", "Temps"),
    "hour_cos": ("Heure (cos)", "Codage cyclique de l'heure : cos(2π·h/24)", "Temps"),
    "is_night": ("Transaction de nuit", "1 si l'heure est entre 22 h et 6 h", "Temps"),
    "dow": ("Jour de la semaine", "0 = lundi … 6 = dimanche", "Temps"),
    "is_weekend": ("Week-end", "1 si samedi ou dimanche", "Temps"),
    "age": ("Âge du client", "Âge à la date de la transaction", "Client"),
    "gender_f": ("Client = femme", "1 si genre F", "Client"),
    "log_city_pop": ("Taille de la ville", "log(1 + population de la ville)", "Client"),
    "distance_km": ("Distance client-marchand", "Distance haversine (km) domicile ↔ marchand", "Localisation"),
    "n_tx_1h": ("Nb transactions (1 h)", "Transactions de la carte dans l'heure précédente", "Vélocité"),
    "n_tx_24h": ("Nb transactions (24 h)", "Transactions de la carte dans les 24 h précédentes", "Vélocité"),
    "amt_sum_24h": ("Montant cumulé (24 h)", "Somme des montants de la carte sur les 24 h précédentes", "Vélocité"),
    "amt_ratio_30d": ("Montant / moyenne 30 j", "Montant rapporté à la moyenne des 30 jours précédents (1 si pas d'historique)", "Écart au profil"),
    "hours_since_prev": ("Heures depuis la précédente", "Temps écoulé depuis la transaction précédente (plafonné à 720 h)", "Vélocité"),
}


def feature_label(name: str) -> str:
    if name in FEATURE_INFO:
        return FEATURE_INFO[name][0]
    if name.startswith("cat_"):
        return f"Catégorie : {name[4:]}"
    return name


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    lat1, lon1, lat2, lon2 = (np.asarray(v, dtype="float64") for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def epoch_seconds(s: pd.Series) -> np.ndarray:
    """Secondes depuis 1970, indépendant de la résolution (ns/us/s) de la colonne datetime."""
    return ((s - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(dtype="int64")


def _window_features(t, a, qt, qa):
    """Variables de vélocité pour des requêtes (qt, qa) à partir d'un historique trié (t, a).

    On ne compte que les transactions d'historique strictement antérieures à qt.
    """
    t = np.asarray(t, dtype="int64")
    a = np.asarray(a, dtype="float64")
    qt = np.asarray(qt, dtype="int64")
    qa = np.asarray(qa, dtype="float64")
    if len(t) == 0:
        m = len(qt)
        return (np.zeros(m), np.zeros(m), np.zeros(m), np.ones(m), np.full(m, MAX_GAP_H))
    cs = np.concatenate(([0.0], np.cumsum(a)))
    hi = np.searchsorted(t, qt, side="left")
    l1 = np.searchsorted(t, qt - 3600, side="left")
    l24 = np.searchsorted(t, qt - 86400, side="left")
    l30 = np.searchsorted(t, qt - 30 * 86400, side="left")
    n1 = (hi - l1).astype("float64")
    n24 = (hi - l24).astype("float64")
    s24 = cs[hi] - cs[l24]
    cnt = hi - l30
    mean30 = np.where(cnt > 0, (cs[hi] - cs[l30]) / np.maximum(cnt, 1), 0.0)
    ratio = np.where(mean30 > 0, qa / np.where(mean30 > 0, mean30, 1.0), 1.0)
    prev_t = t[np.maximum(hi - 1, 0)]
    gap = np.where(hi > 0, (qt - prev_t) / 3600.0, MAX_GAP_H)
    gap = np.minimum(gap, MAX_GAP_H)
    return n1, n24, s24, ratio, gap


def behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables de vélocité / écart au profil, calculées par carte (passé uniquement)."""
    n = len(df)
    card = df[CARD].to_numpy()
    ts = epoch_seconds(df[TIME_COL])
    amt = df[AMT].to_numpy(dtype="float64")
    order = np.lexsort((ts, card))
    c, t, a = card[order], ts[order], amt[order]
    edges = np.flatnonzero(np.r_[True, c[1:] != c[:-1], True])
    res = np.empty((n, len(BEHAVIOR)), dtype="float64")
    for b0, b1 in zip(edges[:-1], edges[1:]):
        tt, aa = t[b0:b1], a[b0:b1]
        res[b0:b1] = np.column_stack(_window_features(tt, aa, tt, aa))
    out = np.empty_like(res)
    out[order] = res
    return pd.DataFrame(out, columns=BEHAVIOR, index=df.index)


def static_features(df: pd.DataFrame, categories=CATEGORIES) -> pd.DataFrame:
    ts = pd.to_datetime(df[TIME_COL])
    h = ts.dt.hour + ts.dt.minute / 60.0
    out = pd.DataFrame(index=df.index)
    amt = df[AMT].astype("float64")
    out["amt"] = amt
    out["log_amt"] = np.log1p(amt)
    out["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    out["is_night"] = ((ts.dt.hour >= 22) | (ts.dt.hour < 6)).astype("float64")
    out["dow"] = ts.dt.dayofweek.astype("float64")
    out["is_weekend"] = (out["dow"] >= 5).astype("float64")
    out["age"] = (ts - pd.to_datetime(df["dob"])).dt.days / 365.25
    out["gender_f"] = (df["gender"].astype(str).str.upper().str[:1] == "F").astype("float64")
    out["log_city_pop"] = np.log1p(df["city_pop"].astype("float64"))
    out["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    for c in categories:
        out[f"cat_{c}"] = (df["category"] == c).astype("float64")
    return out


def engineer(df: pd.DataFrame, categories=CATEGORIES) -> pd.DataFrame:
    """Matrice de variables finale (float32), colonnes dans l'ordre de `feature_names`."""
    st = static_features(df, categories)
    be = behavioral_features(df)
    X = pd.concat([st[STATIC_BASE], be, st[[f"cat_{c}" for c in categories]]], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X["age"] = X["age"].fillna(40.0)
    return X.fillna(0.0).astype("float32")
