import pytest

pytest.importorskip("xgboost")
from fastapi.testclient import TestClient  # noqa: E402

from forecast import config  # noqa: E402

pytestmark = pytest.mark.skipif(not (config.MODEL_DIR / "model_meta.json").exists(),
                                reason="run scripts/train.py first")


@pytest.fixture(scope="module")
def client():
    from api.index import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_forecast_shape_and_plan(client):
    r = client.post("/forecast", json={"horizon": 7, "overrides": {}}).json()
    assert len(r["forecast"]) == 7 and len(r["plan"]) == 7
    assert all(row["covers"] >= 0 for row in r["forecast"])


def test_promo_override_increases_covers(client):
    base = client.post("/forecast", json={"horizon": 14}).json()["forecast"]
    day = next(d for d in base if d["covers"] > 0)["date"]
    promo = client.post("/forecast", json={"horizon": 14, "overrides": {day: {"promo": 1}}}).json()["forecast"]
    b = next(d for d in base if d["date"] == day)["covers"]
    p = next(d for d in promo if d["date"] == day)["covers"]
    assert p >= b


def test_horizon_validation(client):
    assert client.post("/forecast", json={"horizon": 30}).status_code == 422


def test_template_briefing_is_faithful(client):
    r = client.post("/explain", json={"provider": "template"}).json()
    assert r["evaluation"]["faithfulness"] == 1.0
