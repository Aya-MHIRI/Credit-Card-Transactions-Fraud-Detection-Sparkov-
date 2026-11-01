"""Test de l'API : cohérence entraînement / production (nécessite d'avoir lancé train.py).

    python -m pytest tests/test_api.py -q
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import ART_DIR

pytestmark = pytest.mark.skipif(not (ART_DIR / "metadata.json").exists(), reason="lance train.py d'abord")


def payload(row):
    keys = ["cc_num", "trans_date_trans_time", "amt", "category", "gender", "dob", "city_pop",
            "lat", "long", "merch_lat", "merch_long", "merchant"]
    d = {k: row[k] for k in keys if k in row.index}
    d["trans_date_trans_time"] = pd.Timestamp(d["trans_date_trans_time"]).isoformat()
    d["dob"] = str(pd.Timestamp(d["dob"]).date())
    d["cc_num"] = int(d["cc_num"])
    d["city_pop"] = int(d["city_pop"])
    return d


def test_api_matches_offline_scores():
    from api import app

    test = pd.read_pickle(ART_DIR / "scored_test.pkl")
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}
        info = c.get("/model").json()
        assert 0 <= info["threshold"] <= 1
        rows = test.sort_values("s_xgboost", ascending=False).head(3)
        rows = pd.concat([rows, test.sample(5, random_state=0)])
        for _, row in rows.iterrows():
            r = c.post("/predict", json=payload(row))
            assert r.status_code == 200, r.text
            res = r.json()
            # le score en ligne doit égaler le score calculé à l'entraînement (mêmes variables)
            assert abs(res["fraud_probability"] - row["s_xgboost"]) < 2e-3
            assert res["alert"] == (res["fraud_probability"] >= info["threshold"])
        top = c.post("/predict", json=payload(rows.iloc[0])).json()
        assert 1 <= len(top["reason_codes"]) <= 5
        batch = c.post("/predict/batch", json=[payload(r) for _, r in rows.iterrows()]).json()
        assert batch["n"] == len(rows)


def test_api_validation_error():
    from api import app

    with TestClient(app) as c:
        assert c.post("/predict", json={"amt": 10}).status_code == 422
