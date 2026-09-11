import os
import pytest
import numpy as np
from backend.services.image_processor import load_image_as_rgb
from backend.services.vegetation import analyze_vegetation
from backend.services.water_detection import detect_water_bodies
from backend.services.builtup_detection import detect_builtup_and_roads
from backend.services.change_detection import analyze_change
from backend.services.vision_engine import run_full_vision_pipeline
from backend.services.query_engine import classify_query_intent
from backend.services.vlm_engine import get_vlm_engine

OUTPUT_DIR = os.path.join("backend", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_vegetation_analysis():
    path = os.path.join("backend", "samples", "agriculture_delta.jpg")
    rgb, _ = load_image_as_rgb(path)
    pct, mask_path, details = analyze_vegetation(rgb, OUTPUT_DIR, prefix="test_veg")
    assert pct > 20.0
    assert os.path.exists(mask_path)
    assert "quadrant_distribution" in details


def test_water_detection():
    path = os.path.join("backend", "samples", "coastal_port.jpg")
    rgb, _ = load_image_as_rgb(path)
    pct, mask_path, details, features = detect_water_bodies(rgb, OUTPUT_DIR, prefix="test_water")
    assert pct > 25.0
    assert os.path.exists(mask_path)
    assert len(features) > 0


def test_builtup_detection():
    path = os.path.join("backend", "samples", "urban_metropolis.jpg")
    rgb, _ = load_image_as_rgb(path)
    bu_pct, road_pct, bu_mask, rd_mask, details, clusters = detect_builtup_and_roads(
        rgb, OUTPUT_DIR, prefix="test_bu"
    )
    assert bu_pct > 20.0
    assert road_pct > 0.0
    assert os.path.exists(bu_mask)
    assert os.path.exists(rd_mask)


def test_change_detection():
    t1 = os.path.join("backend", "samples", "temporal_t1.jpg")
    t2 = os.path.join("backend", "samples", "temporal_t2.jpg")
    ch_pct, mask_path, side_path, details, features, nlp = analyze_change(
        t1, t2, OUTPUT_DIR, prefix="test_change"
    )
    assert ch_pct > 5.0
    assert os.path.exists(mask_path)
    assert os.path.exists(side_path)
    assert "Approximately" in nlp


def test_intent_classification():
    intent, conf = classify_query_intent("How much vegetation is present?")
    assert intent == "VEGETATION"
    assert conf >= 0.8

    intent2, conf2 = classify_query_intent("Are there water bodies visible?")
    assert intent2 == "WATER"

    intent3, conf3 = classify_query_intent("Compare the two satellite passes")
    assert intent3 in ["CHANGE_DETECTION", "IMAGE_COMPARISON"]


def test_full_vision_pipeline():
    path = os.path.join("backend", "samples", "agriculture_delta.jpg")
    res = run_full_vision_pipeline(path, OUTPUT_DIR, "test_sess_full")
    assert "statistics" in res
    assert "mask_paths" in res
    assert res["statistics"]["vegetation"] > 0
    assert res["dominant_class"] is not None


def test_vlm_engine():
    vlm = get_vlm_engine()
    stats = {"vegetation": 45.0, "water": 15.0, "built_up": 20.0, "barren": 20.0, "confidence": 0.9}
    ans = vlm.generate_answer({"statistics": stats, "dominant_class": "Vegetation", "dominant_percentage": 45.0}, "Summarize vegetation", "VEGETATION")
    assert "45.0%" in ans or "45%" in ans
    assert "vegetation" in ans.lower()
