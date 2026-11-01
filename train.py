#!/usr/bin/env python
"""Entraîne, compare, évalue et explique les modèles ; écrit tout dans artifacts/.

Exemples :
    python train.py --demo                  # test rapide sur données simulées (sans Kaggle)
    python train.py --max-cards 200 --fast  # vrai jeu, sous-échantillon de cartes, rapide
    python train.py                         # vrai jeu complet (data/fraudTrain.csv + fraudTest.csv)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap

from src import data as D
from src import evaluation as E
from src.config import (AMT, ART_DIR, BUDGET_RATE, DATA_DIR, DEPLOYED, MODELS, RANDOM_STATE, REVIEW_COST,
                        SPLIT, TARGET, TIME_COL, CARD)
from src.demo_data import generate_demo
from src.explain import beeswarm_figure, importance_bar_figure, shap_values_of
from src.features import CATEGORIES, FEATURE_INFO, engineer, epoch_seconds, feature_names
from src.imbalance import STRATEGIES, apply_strategy
from src.models import AnomalyScorer, make_logreg, make_xgb

T0 = time.time()


def log(msg: str):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    raise TypeError(type(o))


def md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.4g}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def split_info(df, y, a, b):
    return {
        "n": int(b - a), "frauds": int(y[a:b].sum()), "fraud_rate": float(y[a:b].mean()),
        "start": str(df[TIME_COL].iloc[a]), "end": str(df[TIME_COL].iloc[b - 1]),
        "fraud_amount": float(df[AMT].iloc[a:b][y[a:b] == 1].sum()),
    }


class ScoreNorm:
    """Transforme un score brut en rang percentile (ECDF de la validation) -> échelle [0, 1] commune."""

    def __init__(self, s_val):
        self.v = np.sort(np.asarray(s_val, dtype="float64"))

    def __call__(self, s):
        return np.searchsorted(self.v, np.asarray(s, dtype="float64"), side="right") / len(self.v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="données simulées (test du pipeline)")
    ap.add_argument("--max-cards", type=int, default=None, help="n'utiliser que N cartes tirées au hasard")
    ap.add_argument("--fast", action="store_true", help="moins d'arbres / d'échantillons (rapide)")
    ap.add_argument("--review-cost", type=float, default=REVIEW_COST, help="coût d'analyse d'une alerte ($)")
    ap.add_argument("--budget", type=float, default=BUDGET_RATE, help="budget d'alertes (part des transactions)")
    ap.add_argument("--no-download", action="store_true", help="ne pas essayer Kaggle si data/ est vide")
    args = ap.parse_args()
    ART_DIR.mkdir(exist_ok=True, parents=True)
    rc, kr = args.review_cost, args.budget
    n_est = 100 if args.fast else 300

    # ------------------------------------------------------------------ 1. données
    if args.demo:
        log("Génération des données de DÉMONSTRATION (synthétiques, sans valeur scientifique)")
        raw = generate_demo()
        (DATA_DIR / "demo").mkdir(parents=True, exist_ok=True)
        raw.to_csv(DATA_DIR / "demo" / "demo_transactions.csv", index=False)
        df = D.clean(raw)
    else:
        log("Chargement du jeu de données Sparkov")
        df = D.load_dataset(DATA_DIR, download=not args.no_download, max_cards=args.max_cards)
    y_all = df[TARGET].to_numpy().astype(int)
    log(f"{len(df):,} transactions, {int(y_all.sum()):,} fraudes ({y_all.mean():.3%}), "
        f"{df[CARD].nunique():,} cartes, du {df[TIME_COL].min()} au {df[TIME_COL].max()}")

    # échantillon pour l'exploration dans l'application
    raw_sample = df if len(df) <= 400_000 else df.sample(400_000, random_state=RANDOM_STATE).sort_index()
    raw_sample.to_pickle(ART_DIR / "raw_sample.pkl")

    # ------------------------------------------------------------------ 2. variables + séparation chronologique
    log("Construction des variables (temps, vélocité, écart au profil)")
    X = engineer(df, CATEGORIES)
    names = list(X.columns)
    assert names == feature_names(CATEGORIES)
    amt_all = df[AMT].to_numpy()
    n = len(df)
    i1, i2 = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
    sl = {"train": (0, i1), "val": (i1, i2), "test": (i2, n)}
    Xtr, ytr = X.iloc[:i1], y_all[:i1]
    Xva, yva, ava = X.iloc[i1:i2], y_all[i1:i2], amt_all[i1:i2]
    Xte, yte, ate = X.iloc[i2:], y_all[i2:], amt_all[i2:]
    split = {k: split_info(df, y_all, a, b) for k, (a, b) in sl.items()}
    for k, v in split.items():
        log(f"  {k:5s}: {v['n']:>9,} tx | {v['frauds']:>6,} fraudes ({v['fraud_rate']:.3%}) | {v['start'][:10]} → {v['end'][:10]}")
        if v["frauds"] == 0:
            raise SystemExit(f"Aucune fraude dans le jeu « {k} » : agrandis l'échantillon.")
    k_val, k_test = E.k_from_rate(len(yva), kr), E.k_from_rate(len(yte), kr)
    days_test = max((pd.Timestamp(split["test"]["end"]) - pd.Timestamp(split["test"]["start"])).days + 1, 1)

    # ------------------------------------------------------------------ 3. déséquilibre : comparaison en validation
    log("Comparaison des stratégies contre le déséquilibre (XGBoost, évaluation sur la validation)")
    imb_rows, imb_models = [], {}
    for key, label in STRATEGIES.items():
        try:
            Xr, yr, spw = apply_strategy(Xtr, ytr, key, seed=RANDOM_STATE)
        except Exception as e:  # SMOTE indisponible, etc.
            log(f"  {key}: ignorée ({e})")
            continue
        m = make_xgb(spw, n_est, RANDOM_STATE).fit(Xr, yr)
        s = m.predict_proba(Xva)[:, 1]
        pk, rk = E.precision_recall_at_k(yva, s, k_val)
        best = E.best_cost_threshold(yva, s, ava, rc)
        imb_rows.append({"Stratégie": label, "clé": key, "taille entraînement": len(yr),
                         "PR-AUC (val)": E.pr_auc(yva, s), "ROC-AUC (val)": E.roc_auc(yva, s),
                         f"Précision@k (val)": pk, f"Rappel@k (val)": rk, "Coût min (val)": best["cost"]})
        imb_models[key] = m
        log(f"  {key:22s} PR-AUC val = {imb_rows[-1]['PR-AUC (val)']:.4f} | coût min val = {best['cost']:,.0f}")
    imb = pd.DataFrame(imb_rows)
    chosen = imb.loc[imb["PR-AUC (val)"].idxmax(), "clé"]
    log(f"  -> stratégie retenue : {chosen}")
    xgb = imb_models[chosen]
    del imb_models

    # ------------------------------------------------------------------ 4. autres modèles
    scores_val, scores_test = {}, {}
    scores_val["xgboost"], scores_test["xgboost"] = xgb.predict_proba(Xva)[:, 1], xgb.predict_proba(Xte)[:, 1]

    log("Régression logistique pondérée")
    lr = make_logreg(RANDOM_STATE).fit(Xtr, ytr)
    scores_val["logreg"], scores_test["logreg"] = lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1]

    fit_caps = {"iforest": None, "lof": 8_000 if args.fast else 20_000, "autoencoder": 60_000 if args.fast else 150_000}
    unsup = {}
    for key in ("iforest", "lof", "autoencoder"):
        log(f"{MODELS[key][0]} (non supervisé, sans utiliser la cible)")
        det = AnomalyScorer(key, RANDOM_STATE, fit_caps[key]).fit(Xtr)
        unsup[key] = det
        sv_, st_ = det.score(Xva), det.score(Xte)
        norm = ScoreNorm(sv_)  # échelle commune [0,1] : rang percentile de la validation
        scores_val[key], scores_test[key] = norm(sv_), norm(st_)

    # ------------------------------------------------------------------ 5. évaluation + seuils
    log("Évaluation : PR-AUC, ROC-AUC, précision/rappel@k, seuil à coût minimal (validation) puis test")
    meta_models, rows, curves = {}, [], {}
    for key, (label, kind) in MODELS.items():
        sv_, st_ = scores_val[key], scores_test[key]
        best = E.best_cost_threshold(yva, sv_, ava, rc)
        thr = best["threshold"]
        mv = E.alert_metrics(yva, sv_, ava, thr, rc)
        mt = E.alert_metrics(yte, st_, ate, thr, rc)
        oracle = E.best_cost_threshold(yte, st_, ate, rc)["cost"]
        pk_v, rk_v = E.precision_recall_at_k(yva, sv_, k_val)
        pk_t, rk_t = E.precision_recall_at_k(yte, st_, k_test)
        meta_models[key] = {
            "label": label, "type": kind, "threshold": thr,
            "val": {**mv, "pr_auc": E.pr_auc(yva, sv_), "roc_auc": E.roc_auc(yva, sv_), "p_at_k": pk_v, "r_at_k": rk_v},
            "test": {**mt, "pr_auc": E.pr_auc(yte, st_), "roc_auc": E.roc_auc(yte, st_), "p_at_k": pk_t, "r_at_k": rk_t,
                     "oracle_cost": oracle},
        }
        t = meta_models[key]["test"]
        rows.append({
            "Modèle": label, "Type": kind, "PR-AUC (val)": meta_models[key]["val"]["pr_auc"],
            "PR-AUC (test)": t["pr_auc"], "ROC-AUC (test)": t["roc_auc"],
            "Précision@k (test)": pk_t, "Rappel@k (test)": rk_t, "Seuil (choisi sur val)": thr,
            "F2 (test)": t["f2"], "MCC (test)": t["mcc"], "Alertes (test)": t["alerts"],
            "Coût val ($)": mv["cost"], "Coût test ($)": t["cost"], "Coût test optimal ($)": oracle,
            "Économie vs sans modèle (test)": t["saving_pct"],
        })
        curves[key] = {"val": E.thin_pr_curve(yva, sv_), "test": E.thin_pr_curve(yte, st_)}
        log(f"  {label:32s} PR-AUC test = {t['pr_auc']:.4f} | P@k = {pk_t:.3f} | R@k = {rk_t:.3f} | coût test = {t['cost']:,.0f}")
    comp = pd.DataFrame(rows)
    deployed_thr = meta_models[DEPLOYED]["threshold"]

    budgets = []
    for r_ in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05):
        k_ = E.k_from_rate(len(yte), r_)
        p_, rr_ = E.precision_recall_at_k(yte, scores_test[DEPLOYED], k_)
        budgets.append({"Budget (% transactions)": r_ * 100, "k (alertes)": k_, "Alertes / jour": k_ / days_test,
                        "Précision@k": p_, "Rappel@k": rr_})
    budgets = pd.DataFrame(budgets)

    # ------------------------------------------------------------------ 6. explicabilité (SHAP)
    log("SHAP : explication globale (échantillon du test)")
    explainer = shap.TreeExplainer(xgb)
    rng = np.random.default_rng(RANDOM_STATE)
    rand_idx = rng.choice(len(Xte), size=min(6000, len(Xte)), replace=False)
    fr_idx = np.flatnonzero(yte == 1)
    fr_idx = fr_idx[~np.isin(fr_idx, rand_idx)]
    fr_idx = rng.choice(fr_idx, size=min(len(fr_idx), 800), replace=False) if len(fr_idx) else fr_idx
    idx = np.concatenate([rand_idx, fr_idx])
    Xs = Xte.iloc[idx]
    sv = shap_values_of(explainer, Xs).astype("float32")
    pd.to_pickle({"values": sv, "X": Xs, "is_random": np.r_[np.ones(len(rand_idx), bool), np.zeros(len(fr_idx), bool)],
                  "base_value": float(np.ravel(explainer.expected_value)[0])}, ART_DIR / "shap_sample.pkl")
    fig = beeswarm_figure(sv, Xs)
    fig.savefig(ART_DIR / "beeswarm.png", dpi=130, bbox_inches="tight")
    fig = importance_bar_figure(sv[: len(rand_idx)], names)
    fig.savefig(ART_DIR / "shap_bar.png", dpi=130, bbox_inches="tight")
    imp = np.abs(sv[: len(rand_idx)]).mean(axis=0)
    top_feats = [(names[j], float(imp[j])) for j in np.argsort(-imp)[:8]]

    # ------------------------------------------------------------------ 7. sauvegarde
    log("Sauvegarde des artefacts")
    joblib.dump(xgb, ART_DIR / "model_xgb.joblib")
    joblib.dump(lr, ART_DIR / "model_logreg.joblib")
    for key, det in unsup.items():
        joblib.dump(det, ART_DIR / f"model_{key}.joblib")
    display_cols = [c for c in (TIME_COL, CARD, "merchant", "category", AMT, "gender", "dob", "city", "state", "city_pop",
                                "lat", "long", "merch_lat", "merch_long", "trans_num") if c in df.columns]
    for name, (a, b), sc in (("val", sl["val"], scores_val), ("test", sl["test"], scores_test)):
        part = df.iloc[a:b][display_cols + [TARGET]].reset_index(drop=True)
        for key in MODELS:
            part[f"s_{key}"] = sc[key]
        part.to_pickle(ART_DIR / f"scored_{name}.pkl")
    Xte.reset_index(drop=True).to_pickle(ART_DIR / "X_test.pkl")
    pd.to_pickle(curves, ART_DIR / "pr_curves.pkl")
    ts_all = epoch_seconds(df[TIME_COL])
    o = np.lexsort((ts_all, df[CARD].to_numpy()))
    np.savez_compressed(ART_DIR / "history.npz", cards=df[CARD].to_numpy()[o], ts=ts_all[o], amt=amt_all[o])
    comp.to_csv(ART_DIR / "comparison.csv", index=False)
    imb.to_csv(ART_DIR / "imbalance.csv", index=False)
    budgets.to_csv(ART_DIR / "budgets.csv", index=False)

    meta = {
        "demo": bool(args.demo), "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": "Credit Card Transactions Fraud Detection (Sparkov)" + (" — DONNÉES DE DÉMONSTRATION" if args.demo else ""),
        "n_rows": n, "n_cards": int(df[CARD].nunique()), "fraud_rate": float(y_all.mean()),
        "features": names, "categories": CATEGORIES, "split": split,
        "review_cost": rc, "budget_rate": kr, "k_val": k_val, "k_test": k_test, "days_test": days_test,
        "imbalance": {"chosen": chosen, "labels": STRATEGIES}, "deployed": DEPLOYED, "models": meta_models,
        "top_features": top_feats,
        "args": {"max_cards": args.max_cards, "fast": args.fast, "n_estimators": n_est},
    }
    (ART_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=jdefault), encoding="utf-8")

    # rapport de résultats prêt à copier dans le compte rendu
    d = meta_models[DEPLOYED]
    md = f"""# Résultats d'entraînement{' — DONNÉES DE DÉMONSTRATION' if args.demo else ''}

Généré le {meta['created']} — {n:,} transactions, {int(y_all.sum()):,} fraudes ({y_all.mean():.3%}), {meta['n_cards']:,} cartes.

## Séparation chronologique 60/20/20
{md_table(pd.DataFrame(split).T.reset_index().rename(columns={'index': 'jeu'}))}

## Stratégies contre le déséquilibre (appliquées à l'entraînement, comparées en validation)
Retenue : **{STRATEGIES[chosen]}** (meilleure PR-AUC en validation).

{md_table(imb.drop(columns=['clé']))}

## Comparaison de tous les modèles
Budget k = {kr:.2%} des transactions ({k_test} alertes sur le test, ≈ {k_test / days_test:.1f} alertes/jour). Coût d'analyse = {rc} $ par alerte.
Le seuil de chaque modèle est choisi sur la **validation** (coût minimal) puis appliqué tel quel au **test**.

{md_table(comp)}

## Modèle déployé : {d['label']}
- Seuil (coût minimal sur la validation) : **{d['threshold']:.3f}**
- Test : précision {d['test']['precision']:.3f}, rappel {d['test']['recall']:.3f}, F2 {d['test']['f2']:.3f}, MCC {d['test']['mcc']:.3f}
- Coût test au seuil retenu : {d['test']['cost']:,.0f} $ (sans modèle : {d['test']['baseline_cost']:,.0f} $ ; seuil optimal a posteriori : {d['test']['oracle_cost']:,.0f} $)

### Précision / rappel selon le budget d'alertes (test)
{md_table(budgets)}

## Variables les plus influentes (|SHAP| moyen)
{md_table(pd.DataFrame(top_feats, columns=['variable', '|SHAP| moyen']))}
"""
    (ART_DIR / "rapport_resultats.md").write_text(md, encoding="utf-8")
    log("Terminé. Lance maintenant :  streamlit run app.py   puis   uvicorn api:app --reload")


if __name__ == "__main__":
    main()
