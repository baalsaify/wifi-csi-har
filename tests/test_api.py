import numpy as np
import pytest
from fastapi.testclient import TestClient

from csi_har.api.main import create_app


@pytest.fixture
def client(tiny_model_path):
    with TestClient(create_app(tiny_model_path)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_model_info_lists_12_activities(client):
    info = client.get("/model-info").json()
    assert len(info["classes"]) == 12
    assert info["input"]["csi_values_per_packet"] == 90
    assert "csi_amplitude" in info["input"]["accepts"]


def test_predict_returns_probabilities(client):
    amp = np.abs(np.random.default_rng(0).normal(10, 2, (200, 90))).tolist()
    r = client.post("/predict", json={"csi_amplitude": amp})
    assert r.status_code == 200
    body = r.json()
    assert body["activity"] in body["probabilities"]
    assert body["activity_code"].startswith("A")
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_rejects_wrong_row_width(client):
    r = client.post("/predict", json={"csi_amplitude": [[1.0] * 89] * 50})
    assert r.status_code == 422


def test_predict_rejects_too_few_packets(client):
    r = client.post("/predict", json={"csi_amplitude": [[1.0] * 90] * 5})
    assert r.status_code == 422


def test_predict_file(client, csv_text):
    text, _ = csv_text
    r = client.post("/predict/file", files={"file": ("E1_S01_C01_A01_T01.csv", text, "text/csv")})
    assert r.status_code == 200
    assert "activity" in r.json()


def test_predict_file_rejects_garbage(client):
    r = client.post("/predict/file", files={"file": ("x.csv", "not,a,csi,file\n1,2,3,4\n", "text/csv")})
    assert r.status_code == 422


def _complex_body(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return {"csi_real": rng.normal(size=(n, 90)).tolist(), "csi_imag": rng.normal(size=(n, 90)).tolist()}


def test_predict_accepts_complex_csi(client):
    assert client.post("/predict", json=_complex_body()).status_code == 200


def test_predict_rejects_half_complex_or_empty_body(client):
    body = _complex_body()
    assert client.post("/predict", json={"csi_real": body["csi_real"]}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


@pytest.fixture
def phase_client(tiny_phase_model_path):
    with TestClient(create_app(tiny_phase_model_path)) as c:
        yield c


def test_phase_model_needs_complex_csi(phase_client):
    amp = np.ones((100, 90)).tolist()
    r = phase_client.post("/predict", json={"csi_amplitude": amp})
    assert r.status_code == 422
    assert "csi_real" in r.json()["detail"]
    assert phase_client.post("/predict", json=_complex_body()).status_code == 200
    assert phase_client.get("/model-info").json()["input"]["accepts"] == ["csi_real + csi_imag"]


def test_phase_model_predicts_from_file(phase_client, csv_text):
    text, _ = csv_text
    r = phase_client.post("/predict/file", files={"file": ("t.csv", text, "text/csv")})
    assert r.status_code == 200


def test_service_reports_missing_model(tmp_path):
    with TestClient(create_app(tmp_path / "missing.pt")) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.get("/model-info").status_code == 503
