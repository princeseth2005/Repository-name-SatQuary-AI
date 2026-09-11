import os
import pytest
from backend.services.image_processor import (
    is_valid_extension, get_safe_filename, extract_image_metadata, load_image_as_rgb
)


def test_valid_extensions():
    assert is_valid_extension("scene.jpg") is True
    assert is_valid_extension("scene.jpeg") is True
    assert is_valid_extension("scene.png") is True
    assert is_valid_extension("scene.tif") is True
    assert is_valid_extension("scene.tiff") is True
    assert is_valid_extension("doc.pdf") is False
    assert is_valid_extension("script.py") is False


def test_safe_filename():
    safe = get_safe_filename("My Satellite Image (1).JPEG")
    assert safe.startswith("sat_")
    assert safe.endswith(".jpeg")


def test_optical_metadata_extraction():
    sample_path = os.path.join("backend", "samples", "agriculture_delta.jpg")
    meta = extract_image_metadata(sample_path)
    assert meta["width"] == 800
    assert meta["height"] == 800
    assert meta["format"] in ["JPG", "JPEG"]
    assert meta["file_size"] > 0


def test_geotiff_metadata_extraction():
    geotiff_path = os.path.join("backend", "samples", "isro_sriharikota_geotiff.tif")
    meta = extract_image_metadata(geotiff_path)
    assert meta["width"] == 600
    assert meta["height"] == 600
    assert meta["has_georeference"] is True
    assert meta["latitude"] is not None
    assert meta["longitude"] is not None
    assert 13.0 <= meta["latitude"] <= 14.0
    assert 80.0 <= meta["longitude"] <= 81.0


def test_load_image_as_rgb():
    sample_path = os.path.join("backend", "samples", "coastal_port.jpg")
    arr, pil_img = load_image_as_rgb(sample_path)
    assert arr.ndim == 3
    assert arr.shape[2] == 3
    assert pil_img.size == (800, 800)
