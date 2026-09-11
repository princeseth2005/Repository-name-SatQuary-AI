import os
import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "SatQuery AI"


def test_get_frontend_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "SatQuery AI" in res.text


def test_get_sample_catalog():
    res = client.get("/api/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 4
    assert any(s["filename"] == "agriculture_delta.jpg" for s in samples)


def test_load_sample_and_query():
    # 1. Load sample
    load_res = client.post("/api/sample/load", data={"sample_filename": "agriculture_delta.jpg"})
    assert load_res.status_code == 200
    img_data = load_res.json()
    img_id = img_data["image_id"]
    sess_id = img_data["session_id"]
    assert img_id is not None

    # 2. Run query
    query_res = client.post("/api/query", json={
        "image_id": img_id,
        "query": "How much vegetation is present?",
        "session_id": sess_id
    })
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert q_data["intent"] == "VEGETATION"
    assert q_data["statistics"]["vegetation"] > 0
    assert "vegetation" in q_data["answer"].lower()

    # 3. Check history
    hist_res = client.get("/api/history")
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) > 0


def test_invalid_image_upload():
    # Attempting to upload a txt file
    fake_file = ("fake.txt", b"This is not a satellite image", "text/plain")
    res = client.post("/api/upload", files={"file": fake_file})
    assert res.status_code == 400
    assert "Unsupported format" in res.json()["detail"]


def test_temporal_compare_endpoint():
    # Load T1
    res1 = client.post("/api/sample/load", data={"sample_filename": "temporal_t1.jpg"})
    assert res1.status_code == 200
    id1 = res1.json()["image_id"]

    # Load T2
    res2 = client.post("/api/sample/load", data={"sample_filename": "temporal_t2.jpg"})
    assert res2.status_code == 200
    id2 = res2.json()["image_id"]

    # Compare
    comp_res = client.post("/api/compare", json={
        "image_id_1": id1,
        "image_id_2": id2,
        "query": "Compare these two satellite passes"
    })
    assert comp_res.status_code == 200
    c_data = comp_res.json()
    assert c_data["change_percentage"] > 0.0
    assert "change_mask_url" in c_data
