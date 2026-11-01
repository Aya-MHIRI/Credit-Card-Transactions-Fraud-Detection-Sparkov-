"""Explications SHAP : global (beeswarm), local (waterfall) et « codes motifs »."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402

from .features import feature_label  # noqa: E402


def shap_values_of(explainer, X) -> np.ndarray:
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # anciennes versions : une matrice par classe
        sv = sv[1]
    return np.asarray(sv)


def reason_codes(shap_row, x_row, names, top: int = 5) -> list[dict]:
    """Les `top` variables qui ont le plus POUSSÉ la transaction vers la fraude (contribution > 0)."""
    shap_row = np.asarray(shap_row, dtype="float64")
    x_row = np.asarray(x_row, dtype="float64")
    order = np.argsort(-shap_row)[:top]
    out = []
    for j in order:
        if shap_row[j] <= 0:
            break
        out.append({
            "feature": names[j],
            "label": feature_label(names[j]),
            "value": float(x_row[j]),
            "shap": float(shap_row[j]),
        })
    return out


def describe_reason(r: dict) -> str:
    if r["feature"].startswith("cat_"):
        return f"{r['label']} (contribution +{r['shap']:.2f})"
    if r["feature"] in ("is_night", "is_weekend", "gender_f"):
        return f"{r['label']} : {'oui' if r['value'] >= 0.5 else 'non'} (contribution +{r['shap']:.2f})"
    v = r["value"]
    txt = f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.2f}"
    return f"{r['label']} = {txt} (contribution +{r['shap']:.2f})"


def beeswarm_figure(values, X, max_display: int = 15):
    labels = [feature_label(c) for c in X.columns]
    plt.close("all")
    shap.summary_plot(values, X.to_numpy(), feature_names=labels, show=False, max_display=max_display)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    fig.tight_layout()
    return fig


def importance_bar_figure(values, names, max_display: int = 15):
    imp = np.abs(values).mean(axis=0)
    order = np.argsort(imp)[::-1][:max_display][::-1]
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_label(names[i]) for i in order], imp[order], color="#3B6FB6")
    ax.set_xlabel("Importance moyenne |SHAP| (log-odds)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def waterfall_figure(shap_row, base_value, x_row, names, max_display: int = 12):
    exp = shap.Explanation(
        values=np.asarray(shap_row, dtype="float64"),
        base_values=float(base_value),
        data=np.asarray(x_row, dtype="float64"),
        feature_names=[feature_label(n) for n in names],
    )
    plt.close("all")
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 5.5)
    fig.tight_layout()
    return fig
