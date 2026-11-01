"""Métriques adaptées aux événements rares, budget d'alertes et seuil à coût métier minimal.

Coût métier = montants des fraudes manquées + coût d'analyse × nombre d'alertes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _prep(y, s, amt):
    return np.asarray(y).astype(int), np.asarray(s, dtype="float64"), np.asarray(amt, dtype="float64")


def pr_auc(y, s) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")


def roc_auc(y, s) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def k_from_rate(n: int, rate: float) -> int:
    return max(1, int(np.ceil(n * rate)))


def precision_recall_at_k(y, s, k: int):
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype="float64")
    top = np.argpartition(-s, min(k, len(s)) - 1)[:k]
    tp = int(y[top].sum())
    return tp / k, tp / max(int(y.sum()), 1)


def threshold_for_budget(s, k: int) -> float:
    s = np.asarray(s, dtype="float64")
    k = min(max(k, 1), len(s))
    return float(np.partition(s, len(s) - k)[len(s) - k])


def alert_metrics(y, s, amt, thr: float, review_cost: float) -> dict:
    """Matrice de confusion, précision, rappel, F2, MCC et coût pour une alerte déclenchée si s >= thr."""
    y, s, amt = _prep(y, s, amt)
    a = s >= thr
    tp = int((a & (y == 1)).sum())
    fp = int((a & (y == 0)).sum())
    fn = int((~a & (y == 1)).sum())
    tn = int((~a & (y == 0)).sum())
    alerts, pos = tp + fp, tp + fn
    prec = tp / alerts if alerts else 0.0
    rec = tp / pos if pos else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    den = float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    missed = float(amt[~a & (y == 1)].sum())
    caught = float(amt[a & (y == 1)].sum())
    cost = missed + review_cost * alerts
    baseline = missed + caught  # coût sans modèle : toutes les fraudes sont manquées
    return {
        "threshold": float(thr), "alerts": alerts, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f2": f2, "mcc": mcc,
        "missed_amount": missed, "caught_amount": caught, "review_cost_total": review_cost * alerts,
        "cost": cost, "baseline_cost": baseline,
        "saving": baseline - cost, "saving_pct": (baseline - cost) / baseline if baseline > 0 else 0.0,
    }


def _sorted_arrays(y, s, amt):
    y, s, amt = _prep(y, s, amt)
    order = np.argsort(-s, kind="stable")
    return s[order], np.cumsum(y[order]), np.cumsum((amt * y)[order]), int(y.sum()), float((amt * y).sum())


def best_cost_threshold(y, s, amt, review_cost: float) -> dict:
    """Seuil qui minimise le coût métier (à calculer sur la validation)."""
    s_d, cum_y, cum_amt, pos, total = _sorted_arrays(y, s, amt)
    n = len(s_d)
    m = np.arange(1, n + 1)
    cost = total - cum_amt + review_cost * m
    valid = np.r_[s_d[:-1] > s_d[1:], True]  # on ne coupe pas au milieu d'un lot d'ex æquo
    cost = np.where(valid, cost, np.inf)
    j = int(np.argmin(cost))
    if total <= cost[j]:  # aucune alerte n'est rentable
        return {"threshold": float(s_d[0]) + 1e-9, "cost": total, "alerts": 0, "baseline_cost": total}
    return {"threshold": float(s_d[j]), "cost": float(cost[j]), "alerts": j + 1, "baseline_cost": total}


def cost_curve(y, s, amt, review_cost: float) -> pd.DataFrame:
    """Coût, précision et rappel en fonction du seuil (pour les graphiques)."""
    s_d, cum_y, cum_amt, pos, total = _sorted_arrays(y, s, amt)
    s_all = np.asarray(s, dtype="float64")
    thr = np.unique(np.r_[np.linspace(0, 1, 201), np.quantile(s_all, np.linspace(0.97, 1, 150))])
    m = np.searchsorted(-s_d, -thr, side="right")
    idx = np.maximum(m - 1, 0)
    tp = np.where(m > 0, cum_y[idx], 0)
    caught = np.where(m > 0, cum_amt[idx], 0.0)
    return pd.DataFrame({
        "threshold": thr, "alerts": m,
        "precision": np.where(m > 0, tp / np.maximum(m, 1), np.nan),
        "recall": tp / max(pos, 1),
        "cost": total - caught + review_cost * m,
    })


def thin_pr_curve(y, s, max_points: int = 1500):
    """Courbe précision-rappel allégée pour l'affichage (garde toute la zone des scores élevés)."""
    from sklearn.metrics import precision_recall_curve

    p, r, _ = precision_recall_curve(y, s)
    n = len(p)
    idx = np.unique(np.r_[np.linspace(0, n - 1, max_points).astype(int), np.arange(max(0, n - 1500), n)])
    return p[idx].astype("float32"), r[idx].astype("float32")
