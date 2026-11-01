"""Application web interactive de détection de fraude explicable (Streamlit).

Lancer :  streamlit run app.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import data as D
from src import evaluation as E
from src.config import AMT, ART_DIR, CARD, DEPLOYED, MODELS, TARGET, TIME_COL
from src.explain import describe_reason, reason_codes, shap_values_of, waterfall_figure
from src.features import FEATURE_INFO, feature_label
from src.imbalance import STRATEGIES
from src.serving import FraudScorer

st.set_page_config(page_title="Détection de fraude explicable", page_icon="🛡️", layout="wide")


# ----------------------------------------------------------------------------- utilitaires
def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # anciennes versions de Streamlit
        st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def pct(x, d=1):
    return f"{x * 100:.{d}f} %"


def money(x):
    return f"{x:,.0f} $".replace(",", " ")


if not (ART_DIR / "metadata.json").exists():
    st.title("🛡️ Détection de fraude explicable")
    st.error("Aucun modèle entraîné. Dans un terminal, lance d'abord :  `python train.py`  "
             "(ou `python train.py --demo` pour un test rapide), puis recharge cette page.")
    st.stop()

STAMP = (ART_DIR / "metadata.json").stat().st_mtime


@st.cache_data
def load_meta(stamp):
    return json.loads((ART_DIR / "metadata.json").read_text(encoding="utf-8"))


@st.cache_data
def load_pkl(name, stamp):
    return pd.read_pickle(ART_DIR / name)


@st.cache_data
def load_csv(name, stamp):
    return pd.read_csv(ART_DIR / name)


@st.cache_resource
def load_scorer(stamp):
    return FraudScorer()


meta = load_meta(STAMP)
RC0 = meta["review_cost"]

# ----------------------------------------------------------------------------- barre latérale
st.sidebar.title("🛡️ Fraude explicable")
PAGES = ["1 · Données", "2 · Préparation", "3 · Modèles", "4 · Seuil et coût", "5 · Explicabilité", "6 · API"]
page = st.sidebar.radio("Navigation", PAGES, key="page")
st.sidebar.caption(f"Jeu : {meta['dataset']}")
st.sidebar.caption(f"{meta['n_rows']:,} transactions · taux de fraude {meta['fraud_rate']:.2%}".replace(",", " "))
st.sidebar.caption(f"Entraîné le {meta['created'][:16].replace('T', ' ')}")
if meta["demo"]:
    st.sidebar.warning("DONNÉES DE DÉMONSTRATION simulées : les résultats n'ont aucune valeur. "
                       "Lance `python train.py` sur le vrai jeu pour le rendu.")

if meta["demo"]:
    st.warning("⚠️ Données de démonstration simulées — ne pas utiliser pour le rendu.")


# ============================================================================= 1. DONNÉES
if page == PAGES[0]:
    st.title("Module 1 · Exploration des données")
    src = st.radio("Source", ["Jeu utilisé pour l'entraînement", "Importer un fichier CSV (format Sparkov)"], horizontal=True)
    df = None
    if src.startswith("Jeu"):
        df = load_pkl("raw_sample.pkl", STAMP)
        if len(df) < meta["n_rows"]:
            st.caption(f"Échantillon aléatoire de {len(df):,} transactions sur {meta['n_rows']:,} (pour la fluidité).".replace(",", " "))
    else:
        up = st.file_uploader("Fichier CSV avec les colonnes du jeu Sparkov (fraudTrain.csv, fraudTest.csv…)", type="csv")
        if up is not None:
            try:
                df = D.read_csv_any(up)
            except Exception as e:
                st.error(f"Fichier invalide : {e}")
    with st.expander("Télécharger le jeu depuis Kaggle (nécessite un compte et `kagglehub`)"):
        st.write("Les fichiers sont copiés dans le dossier `data/`. Relance ensuite `python train.py`.")
        if st.button("Télécharger Kaggle → data/"):
            with st.spinner("Téléchargement…"):
                try:
                    files = D.download_from_kaggle()
                    st.success("Téléchargé : " + ", ".join(f.name for f in files))
                except Exception as e:
                    st.error(f"Téléchargement impossible ({e}). Télécharge manuellement fraudTrain.csv et "
                             "fraudTest.csv sur Kaggle (kartik2112/fraud-detection) et place-les dans data/.")

    if df is not None and len(df):
        df = df.copy()
        df["hour"] = df[TIME_COL].dt.hour
        df["jour_semaine"] = df[TIME_COL].dt.day_name()
        n, nf = len(df), int(df[TARGET].sum())
        c = st.columns(5)
        c[0].metric("Transactions", f"{n:,}".replace(",", " "))
        c[1].metric("Fraudes", f"{nf:,}".replace(",", " "))
        c[2].metric("Taux de fraude", pct(nf / n, 2))
        c[3].metric("Montant moyen (normal)", money(df.loc[df[TARGET] == 0, AMT].mean()))
        c[4].metric("Montant moyen (fraude)", money(df.loc[df[TARGET] == 1, AMT].mean()) if nf else "—")

        st.subheader("Distribution des montants par classe")
        samp = df.sample(min(len(df), 150_000), random_state=0).copy()
        samp["classe"] = np.where(samp[TARGET] == 1, "Fraude", "Normale")
        samp["log10(montant)"] = np.log10(samp[AMT] + 1)
        a, b = st.columns(2)
        with a:
            fig = px.histogram(samp, x="log10(montant)", color="classe", barmode="overlay", histnorm="probability",
                               nbins=60, opacity=0.65, color_discrete_map={"Normale": "#3B6FB6", "Fraude": "#D64545"})
            fig.update_layout(height=340, yaxis_title="proportion", margin=dict(t=30, b=10))
            show(fig)
        with b:
            fig = px.box(samp, x="classe", y=AMT, color="classe", log_y=True,
                         color_discrete_map={"Normale": "#3B6FB6", "Fraude": "#D64545"})
            fig.update_layout(height=340, showlegend=False, margin=dict(t=30, b=10))
            show(fig)

        st.subheader("Évolution du taux de fraude dans le temps")
        freq = st.radio("Période", ["Semaine", "Mois"], horizontal=True)
        g = df.set_index(TIME_COL)[TARGET].resample("W" if freq == "Semaine" else "MS").agg(["mean", "sum", "count"])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=g.index, y=g["count"], name="Transactions", marker_color="#C9D6EA", secondary_y=True)
        fig.add_scatter(x=g.index, y=g["mean"], name="Taux de fraude", mode="lines+markers",
                        line=dict(color="#D64545"), secondary_y=False)
        fig.update_yaxes(title_text="Taux de fraude", tickformat=".2%", secondary_y=False)
        fig.update_yaxes(title_text="Transactions", secondary_y=True, showgrid=False)
        fig.update_layout(height=360, margin=dict(t=30, b=10), legend=dict(orientation="h", y=1.12))
        show(fig)

        st.subheader("Taux de fraude par variable")
        options = {"Heure": "hour", "Catégorie": "category", "Jour de la semaine": "jour_semaine",
                   "Genre": "gender", "État": "state", "Marchand": "merchant"}
        options = {k: v for k, v in options.items() if v in df.columns}
        var = st.selectbox("Variable", list(options))
        col = options[var]
        minc = st.slider("Effectif minimum par modalité", 1, 500, 30)
        r = df.groupby(col)[TARGET].agg(taux="mean", fraudes="sum", transactions="count").reset_index()
        r = r[r["transactions"] >= minc]
        r = r.sort_values("hour" if col == "hour" else "taux", ascending=(col == "hour")) if col != "jour_semaine" else r
        if col in ("state", "merchant"):
            r = r.sort_values("taux", ascending=False).head(25)
        fig = px.bar(r, x=col, y="taux", hover_data=["fraudes", "transactions"], color_discrete_sequence=["#D64545"])
        fig.update_layout(height=360, yaxis_tickformat=".2%", margin=dict(t=30, b=10))
        fig.add_hline(y=nf / n, line_dash="dash", annotation_text="taux global")
        show(fig)
    elif src.startswith("Importer"):
        st.info("Importe un fichier pour l'explorer.")


# ============================================================================= 2. PRÉPARATION
elif page == PAGES[1]:
    st.title("Module 2 · Préparation et variables")
    st.subheader("Séparation chronologique 60 / 20 / 20")
    sp = pd.DataFrame(meta["split"]).T.reset_index().rename(columns={"index": "jeu"})
    sp["jeu"] = sp["jeu"].map({"train": "Entraînement", "val": "Validation", "test": "Test"})
    sp["fraud_rate"] = sp["fraud_rate"].map(lambda v: f"{v:.3%}")
    sp["fraud_amount"] = sp["fraud_amount"].map(money)
    sp["start"], sp["end"] = sp["start"].str[:10], sp["end"].str[:10]
    table(sp.rename(columns={"n": "transactions", "frauds": "fraudes", "fraud_rate": "taux de fraude",
                             "start": "début", "end": "fin", "fraud_amount": "montant des fraudes"}))
    st.caption("Le jeu est trié par date puis découpé : le modèle est toujours évalué sur des transactions POSTÉRIEURES "
               "à celles de l'entraînement (comme en production). Les variables comportementales ne regardent que le passé "
               "de la carte : aucune fuite du futur.")

    st.subheader("Variables construites")
    info = pd.DataFrame([{"Variable": k, "Libellé": v[0], "Groupe": v[2], "Description": v[1]} for k, v in FEATURE_INFO.items()])
    info = pd.concat([info, pd.DataFrame([{"Variable": "cat_*", "Libellé": "Catégorie du marchand", "Groupe": "Marchand",
                                           "Description": f"Codage one-hot des {len(meta['categories'])} catégories"}])])
    table(info)

    st.subheader("Stratégie contre le déséquilibre")
    imb = load_csv("imbalance.csv", STAMP)
    chosen = meta["imbalance"]["chosen"]
    st.markdown(f"Appliquée **au seul jeu d'entraînement**, comparée sur la **validation**. "
                f"Stratégie retenue : **{STRATEGIES[chosen]}** (meilleure PR-AUC en validation).")
    imb["Retenue"] = np.where(imb["clé"] == chosen, "✅", "")
    table(imb.drop(columns=["clé"]).style.format({c: "{:.4f}" for c in imb.columns if "AUC" in c or "@k" in c} |
                                                 {"Coût min (val)": "{:,.0f}"}))
    fig = px.bar(imb, x="Stratégie", y="PR-AUC (val)", color=np.where(imb["clé"] == chosen, "retenue", "autre"),
                 color_discrete_map={"retenue": "#2E8B57", "autre": "#9FB3CC"})
    fig.update_layout(height=320, showlegend=False, margin=dict(t=20, b=10), xaxis_title=None)
    show(fig)


# ============================================================================= 3. MODÈLES
elif page == PAGES[2]:
    st.title("Module 3 · Comparaison des modèles")
    comp = load_csv("comparison.csv", STAMP)
    k_pct = meta["budget_rate"]
    st.caption(f"Budget k = {pct(k_pct, 2)} des transactions du jeu de test ({meta['k_test']} alertes ≈ "
               f"{meta['k_test'] / meta['days_test']:.1f} par jour). Seuil choisi sur la validation (coût minimal), "
               f"coût d'analyse {RC0:g} $ par alerte.")
    fmt = {c: "{:.4f}" for c in comp.columns if "AUC" in c or "@k" in c or "F2" in c or "MCC" in c}
    fmt.update({"Seuil (choisi sur val)": "{:.3f}", "Coût val ($)": "{:,.0f}", "Coût test ($)": "{:,.0f}",
                "Coût test optimal ($)": "{:,.0f}", "Économie vs sans modèle (test)": "{:.1%}"})
    table(comp.style.format(fmt))
    st.download_button("Télécharger le tableau (CSV)", comp.to_csv(index=False).encode("utf-8"), "comparaison_modeles.csv")

    st.subheader("Courbes précision-rappel superposées")
    which = st.radio("Jeu", ["Test", "Validation"], horizontal=True)
    curves = load_pkl("pr_curves.pkl", STAMP)
    sc = load_pkl("scored_test.pkl" if which == "Test" else "scored_val.pkl", STAMP)
    prev = sc[TARGET].mean()
    fig = go.Figure()
    for key, (label, kind) in MODELS.items():
        p, r = curves[key]["test" if which == "Test" else "val"]
        ap = comp.loc[comp["Modèle"] == label, "PR-AUC (test)" if which == "Test" else "PR-AUC (val)"].iloc[0]
        fig.add_scatter(x=r, y=p, mode="lines", name=f"{label} (AP={ap:.3f})",
                        line=dict(dash="solid" if kind == "supervisé" else "dash"))
    fig.add_hline(y=prev, line_dash="dot", line_color="gray", annotation_text=f"hasard = {prev:.2%}")
    fig.update_layout(height=480, xaxis_title="Rappel", yaxis_title="Précision", xaxis_range=[0, 1.0],
                      yaxis_range=[0, 1.02], legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
    show(fig)
    st.info("**Lecture :** en détection de fraude, la PR-AUC est plus informative que la ROC-AUC car la classe positive est "
            "très rare. Les méthodes non supervisées (traits pointillés) n'utilisent pas les étiquettes : elles servent de "
            "référence et peuvent repérer des fraudes d'un type nouveau, mais restent en général moins précises que le modèle supervisé.")


# ============================================================================= 4. SEUIL ET COÛT
elif page == PAGES[3]:
    st.title("Module 4 · Seuil d'alerte, budget et coût métier")
    sv, stt = load_pkl("scored_val.pkl", STAMP), load_pkl("scored_test.pkl", STAMP)

    left, right = st.columns([1, 2])
    with left:
        key = st.selectbox("Modèle", list(MODELS), format_func=lambda k: f"{MODELS[k][0]} ({MODELS[k][1]})",
                           index=list(MODELS).index(DEPLOYED))
        rc = st.slider("Coût d'analyse d'une alerte ($)", 0.0, 100.0, float(RC0), 0.5, key="rc")
        best = E.best_cost_threshold(sv[TARGET], sv[f"s_{key}"], sv[AMT], rc)
        thr_key = f"thr_{key}"
        if thr_key not in st.session_state:
            st.session_state[thr_key] = float(min(best["threshold"], 1.0))
        mode = st.radio("Piloter par", ["Seuil", "Budget d'alertes"], horizontal=True)
        st.slider("Seuil d'alerte (score ≥ seuil)", 0.0, 1.0, step=0.001, key=thr_key, disabled=(mode != "Seuil"))
        st.button("↺ Appliquer le seuil optimal (coût minimal sur la validation)",
                  on_click=lambda: st.session_state.__setitem__(thr_key, float(min(best["threshold"], 1.0))))
        budget = st.slider("Budget d'alertes (% des transactions)", 0.05, 5.0, float(meta["budget_rate"] * 100), 0.05,
                           disabled=(mode != "Budget d'alertes")) / 100
        st.caption(f"Seuil optimal sur la validation : **{best['threshold']:.3f}** (coût {money(best['cost'])}, "
                   f"sans modèle {money(best['baseline_cost'])}).")
        if key in ("iforest", "lof", "autoencoder"):
            st.caption("Score non supervisé converti en rang percentile (échelle de la validation) : un seuil de 0,99 "
                       "≈ les 1 % de transactions les plus atypiques.")

    thr_ui = st.session_state[thr_key]
    res = {}
    for name, d in (("Validation", sv), ("Test", stt)):
        s = d[f"s_{key}"].to_numpy()
        if mode == "Seuil":
            thr = thr_ui
        else:
            thr = E.threshold_for_budget(s, E.k_from_rate(len(s), budget))
        m = E.alert_metrics(d[TARGET], s, d[AMT], thr, rc)
        res[name] = m
        res[name]["days"] = max((d[TIME_COL].max() - d[TIME_COL].min()).days + 1, 1)

    with right:
        t = res["Test"]
        c = st.columns(4)
        c[0].metric("Alertes (test)", f"{t['alerts']:,}".replace(",", " "), f"{t['alerts'] / t['days']:.1f} / jour", delta_color="off")
        c[1].metric("Précision", pct(t["precision"]))
        c[2].metric("Rappel", pct(t["recall"]))
        c[3].metric("Coût total", money(t["cost"]), f"{t['saving_pct']:+.1%} vs sans modèle".replace("+", "−" if t["saving_pct"] > 0 else "+"),
                    delta_color="off")
        rows = []
        for lab, f in (("Alertes", lambda m: f"{m['alerts']:,}"), ("Vraies fraudes détectées (TP)", lambda m: f"{m['tp']:,}"),
                       ("Fausses alertes (FP)", lambda m: f"{m['fp']:,}"), ("Fraudes manquées (FN)", lambda m: f"{m['fn']:,}"),
                       ("Précision", lambda m: pct(m["precision"], 2)), ("Rappel", lambda m: pct(m["recall"], 2)),
                       ("F2", lambda m: f"{m['f2']:.3f}"), ("MCC", lambda m: f"{m['mcc']:.3f}"),
                       ("Montant des fraudes manquées", lambda m: money(m["missed_amount"])),
                       ("Coût d'analyse des alertes", lambda m: money(m["review_cost_total"])),
                       ("Coût total", lambda m: money(m["cost"])), ("Coût sans modèle", lambda m: money(m["baseline_cost"])),
                       ("Seuil appliqué", lambda m: f"{m['threshold']:.4f}")):
            rows.append({"Indicateur": lab, "Validation": f(res["Validation"]), "Test": f(res["Test"])})
        table(pd.DataFrame(rows))
        if mode == "Seuil":
            st.caption("Le même seuil est appliqué à la validation (où il a été choisi) et au test (où on le confirme).")

    st.subheader("Coût en fonction du seuil")
    fig = go.Figure()
    for name, d, colr in (("Validation", sv, "#3B6FB6"), ("Test", stt, "#D64545")):
        cc = E.cost_curve(d[TARGET], d[f"s_{key}"], d[AMT], rc)
        fig.add_scatter(x=cc["threshold"], y=cc["cost"], mode="lines", name=f"Coût {name.lower()}", line=dict(color=colr))
        fig.add_scatter(x=[res[name]["threshold"]], y=[res[name]["cost"]], mode="markers", showlegend=False,
                        marker=dict(size=11, color=colr, line=dict(width=2, color="white")))
    fig.add_vline(x=best["threshold"], line_dash="dash", line_color="#2E8B57", annotation_text="optimum (val)")
    fig.update_layout(height=380, xaxis_title="Seuil", yaxis_title="Coût ($)", margin=dict(t=20), legend=dict(orientation="h", y=1.12))
    show(fig)

    st.subheader("Précision et rappel en fonction du seuil (test)")
    cc = E.cost_curve(stt[TARGET], stt[f"s_{key}"], stt[AMT], rc)
    fig = go.Figure()
    fig.add_scatter(x=cc["threshold"], y=cc["precision"], name="Précision", line=dict(color="#3B6FB6"))
    fig.add_scatter(x=cc["threshold"], y=cc["recall"], name="Rappel", line=dict(color="#D64545"))
    fig.add_vline(x=res["Test"]["threshold"], line_dash="dot")
    fig.update_layout(height=320, xaxis_title="Seuil", yaxis_range=[0, 1.02], margin=dict(t=20), legend=dict(orientation="h", y=1.12))
    show(fig)

    st.subheader("Budget d'alertes — modèle déployé (test)")
    bud = load_csv("budgets.csv", STAMP)
    table(bud.style.format({"Budget (% transactions)": "{:.2f}", "Alertes / jour": "{:.1f}", "Précision@k": "{:.3f}", "Rappel@k": "{:.3f}"}))
    st.caption("k justifié : capacité d'analyse supposée de l'équipe (≈ nombre d'alertes/jour que les analystes peuvent traiter). "
               "À adapter dans `src/config.py` (BUDGET_RATE).")


# ============================================================================= 5. EXPLICABILITÉ
elif page == PAGES[4]:
    st.title("Module 5 · Explicabilité (SHAP)")
    scorer = load_scorer(STAMP)
    st.subheader("Explication globale")
    a, b = st.columns(2)
    with a:
        st.image(str(ART_DIR / "beeswarm.png"), caption="SHAP beeswarm : chaque point est une transaction (échantillon du test)")
    with b:
        st.image(str(ART_DIR / "shap_bar.png"), caption="Importance moyenne |SHAP| (échantillon aléatoire du test)")
    st.caption("Beeswarm : position horizontale = effet sur le score (à droite = pousse vers la fraude), couleur = valeur de la "
               "variable. L'échantillon contient aussi des fraudes ajoutées pour la lisibilité ; la barre d'importance, elle, "
               "est calculée sur un échantillon aléatoire non biaisé.")

    st.subheader("Explication d'une alerte")
    stt = load_pkl("scored_test.pkl", STAMP)
    Xte = load_pkl("X_test.pkl", STAMP)
    thr = scorer.threshold
    stt = stt.assign(score=stt[f"s_{DEPLOYED}"])
    filt = st.radio("Transactions à examiner",
                    ["Alertes (score ≥ seuil)", "Vrais positifs", "Faux positifs", "Fraudes manquées (FN)", "Toutes (scores les plus élevés)"],
                    horizontal=True)
    y = stt[TARGET] == 1
    al = stt["score"] >= thr
    mask = {"Alertes (score ≥ seuil)": al, "Vrais positifs": al & y, "Faux positifs": al & ~y,
            "Fraudes manquées (FN)": ~al & y, "Toutes (scores les plus élevés)": al | ~al}[filt]
    cand = stt[mask].sort_values(AMT if filt.startswith("Fraudes manquées") else "score", ascending=False).head(150)
    if cand.empty:
        st.info("Aucune transaction dans cette catégorie.")
    else:
        def lab(i):
            r = cand.loc[i]
            return (f"#{i} · {r[TIME_COL]:%Y-%m-%d %H:%M} · {r[AMT]:,.2f} $ · {r['category']} · "
                    f"score {r['score']:.3f} · {'FRAUDE' if r[TARGET] == 1 else 'normale'}")

        i = st.selectbox(f"{len(cand)} transactions (triées {'par montant' if filt.startswith('Fraudes manquées') else 'par score'})",
                         list(cand.index), format_func=lab)
        row = stt.loc[i]
        x = Xte.loc[[i]]
        sv_row = shap_values_of(scorer.explainer, x)[0]
        p = float(row["score"])
        c = st.columns(4)
        c[0].metric("Score de fraude", f"{p:.3f}", f"seuil {thr:.3f}", delta_color="off")
        c[1].metric("Décision", "🚨 ALERTE" if p >= thr else "✅ pas d'alerte")
        c[2].metric("Réalité", "Fraude" if row[TARGET] == 1 else "Normale")
        c[3].metric("Montant", money(row[AMT]))
        left, right = st.columns([3, 2])
        with left:
            fig = waterfall_figure(sv_row, scorer.base_value, x.iloc[0].to_numpy(), list(x.columns))
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Waterfall SHAP : on part de la valeur moyenne du modèle E[f(x)] (log-odds) et chaque variable "
                       "ajoute (rouge) ou retire (bleu) du risque jusqu'au score final f(x).")
        with right:
            st.markdown("**Codes motifs** (variables qui ont le plus poussé vers l'alerte)")
            codes = reason_codes(sv_row, x.iloc[0].to_numpy(), list(x.columns), top=5)
            if codes:
                for k, r in enumerate(codes, 1):
                    st.markdown(f"{k}. {describe_reason(r)}")
            else:
                st.write("Aucune variable ne pousse vers la fraude.")
            st.markdown("**Détails de la transaction**")
            det = {"Date": f"{row[TIME_COL]:%Y-%m-%d %H:%M:%S}", "Carte": str(row[CARD])[-4:].rjust(16, "•"),
                   "Marchand": row.get("merchant", "—"), "Catégorie": row["category"], "Ville / État": f"{row.get('city', '')} {row.get('state', '')}"}
            table(pd.DataFrame({"Champ": list(det), "Valeur": [str(v) for v in det.values()]}))


# ============================================================================= 6. API
else:
    st.title("Module 6 · API FastAPI")
    scorer = load_scorer(STAMP)
    st.markdown("""
L'API expose le **modèle déployé** et le seuil à coût minimal. Pour la lancer, dans un terminal :

```bash
uvicorn api:app --reload
```
Documentation interactive : <http://localhost:8000/docs>

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | état du service |
| GET | `/model` | modèle, seuil, métriques de test, variables |
| POST | `/predict` | score + alerte + **codes motifs** d'une transaction |
| POST | `/predict/batch` | jusqu'à 1000 transactions |
""")
    st.subheader("Tester une transaction")
    stt = load_pkl("scored_test.pkl", STAMP)
    cand = stt.sort_values(f"s_{DEPLOYED}", ascending=False).head(30)
    i = st.selectbox("Transaction du jeu de test", list(cand.index),
                     format_func=lambda i: f"#{i} · {cand.loc[i, TIME_COL]:%Y-%m-%d %H:%M} · {cand.loc[i, AMT]:.2f} $ · {cand.loc[i, 'category']}")
    r = cand.loc[i]
    payload = {"cc_num": int(r[CARD]), "trans_date_trans_time": pd.Timestamp(r[TIME_COL]).isoformat(), "amt": float(r[AMT]),
               "category": r["category"], "gender": r["gender"], "dob": str(pd.Timestamp(r["dob"]).date()),
               "city_pop": int(r["city_pop"]), "lat": float(r["lat"]), "long": float(r["long"]),
               "merch_lat": float(r["merch_lat"]), "merch_long": float(r["merch_long"])}
    payload = json.loads(st.text_area("Corps de la requête (JSON modifiable)", json.dumps(payload, indent=2), height=270))
    url = st.text_input("URL de l'API", "http://localhost:8000")
    a, b = st.columns(2)
    out = None
    if a.button("Appeler l'API locale"):
        try:
            import httpx

            resp = httpx.post(f"{url}/predict", json=payload, timeout=10)
            out = resp.json()
            st.caption(f"HTTP {resp.status_code}")
        except Exception as e:
            st.error(f"API injoignable ({e}). Lance `uvicorn api:app` dans un autre terminal, "
                     "ou utilise le bouton « Calculer sans serveur ».")
    if b.button("Calculer sans serveur (même code)"):
        try:
            out = scorer.score(payload)
        except Exception as e:
            st.error(str(e))
    if out is not None:
        st.json(out)

    st.subheader("Scoring par lot d'un fichier CSV")
    st.caption("Fichier au format Sparkov. Les variables de vélocité sont calculées à partir de l'historique du fichier.")
    up = st.file_uploader("CSV à scorer", type="csv", key="batch")
    if up is not None:
        try:
            d = D.read_csv_any(up)
            d["score"] = scorer.score_dataframe(d)
            d["alerte"] = d["score"] >= scorer.threshold
            st.success(f"{len(d):,} transactions scorées, {int(d['alerte'].sum()):,} alertes (seuil {scorer.threshold:.3f}).".replace(",", " "))
            top = d.sort_values("score", ascending=False).head(200)
            table(top[[TIME_COL, CARD, "category", AMT, "score", "alerte"]])
            st.download_button("Télécharger les résultats (CSV)", d.to_csv(index=False).encode("utf-8"), "transactions_scorees.csv")
        except Exception as e:
            st.error(f"Impossible de scorer ce fichier : {e}")
