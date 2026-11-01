"""Smoke test de l'application Streamlit : chaque page doit s'afficher sans exception.

    python -m pytest tests/test_app.py -q
"""
from pathlib import Path

import pytest

from src.config import ART_DIR

pytestmark = pytest.mark.skipif(not (ART_DIR / "metadata.json").exists(), reason="lance train.py d'abord")
APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.mark.parametrize("page", ["1 · Données", "2 · Préparation", "3 · Modèles", "4 · Seuil et coût", "5 · Explicabilité", "6 · API"])
def test_pages(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_threshold_and_budget_controls():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio(key="page").set_value("4 · Seuil et coût").run()
    at.session_state["thr_xgboost"] = 0.9
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    mode = next(r for r in at.radio if r.label == "Piloter par")
    mode.set_value("Budget d'alertes").run()
    assert not at.exception, [e.value for e in at.exception]
    # en mode budget, le nombre d'alertes suit le budget (~1 % par défaut)
    assert any("Alertes (test)" in m.label for m in at.metric)
