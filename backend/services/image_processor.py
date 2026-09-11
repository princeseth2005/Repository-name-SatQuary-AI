import os
import uuid
import json
import numpy as np
from PIL import Image, ExifTags
import tifffile
from typing import Dict, Any, Tuple, Optional

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def is_valid_extension(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def get_safe_filename(original_filename: str) -> str:
    ext = os.path.splitext(original_filename)[1].lower()
    if not ext:
        ext = ".jpg"
    return f"sat_{uuid.uuid4().hex[:12]}{ext}"


def extract_geotiff_metadata(file_path: str) -> Dict[str, Any]:
    """Extracts geospatial tiepoints, pixel scales, and geographic coordinates from GeoTIFF."""
    geo_meta = {
        "has_georeference": False,
        "latitude": None,
        "longitude": None,
        "crs": None,
        "bounding_box": None,
        "pixel_scale": None
    }

    try:
        with tifffile.TiffFile(file_path) as tif:
            for page in tif.pages:
                tags = {tag.name: tag.value for tag in page.tags.values()}

                # Check for ModelTiepointTag (tag 33922) and ModelPixelScaleTag (tag 33550)
                tiepoints = tags.get("ModelTiepointTag")
                pixel_scale = tags.get("ModelPixelScaleTag")

                if tiepoints is not None and pixel_scale is not None:
                    # Tiepoint: (i, j, k, x, y, z)
                    if len(tiepoints) >= 6 and len(pixel_scale) >= 2:
                        min_x = float(tiepoints[3])
                        max_y = float(tiepoints[4])
                        scale_x = float(pixel_scale[0])
                        scale_y = float(pixel_scale[1])

                        w = page.shape[1] if len(page.shape) >= 2 else 512
                        h = page.shape[0] if len(page.shape) >= 2 else 512

                        max_x = min_x + (w * scale_x)
                        min_y = max_y - (h * scale_y)

                        center_lon = (min_x + max_x) / 2.0
                        center_lat = (min_y + max_y) / 2.0

                        geo_meta["has_georeference"] = True
                        geo_meta["longitude"] = round(center_lon, 6)
                        geo_meta["latitude"] = round(center_lat, 6)
                        geo_meta["crs"] = "EPSG:4326 (WGS84)"
                        geo_meta["bounding_box"] = [
                            round(min_y, 6), round(min_x, 6),
                            round(max_y, 6), round(max_x, 6)
                        ]
                        geo_meta["pixel_scale"] = [scale_x, scale_y]
                        break
    except Exception as e:
        # Fallback gracefully if not a GeoTIFF or standard tags missing
        pass

    return geo_meta


def extract_image_metadata(file_path: str) -> Dict[str, Any]:
    """Reads comprehensive dimensions, format, channels, EXIF, and GeoTIFF tags."""
    file_size = os.path.getsize(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    meta = {
        "width": 0,
        "height": 0,
        "channels": 3,
        "color_mode": "RGB",
        "file_size": file_size,
        "format": ext.replace(".", "").upper(),
        "has_georeference": False,
        "latitude": None,
        "longitude": None,
        "crs": None,
        "bounding_box": None,
        "extra_tags": {}
    }

    if ext in [".tif", ".tiff"]:
        geo = extract_geotiff_metadata(file_path)
        meta.update(geo)

    try:
        with Image.open(file_path) as img:
            meta["width"] = img.width
            meta["height"] = img.height
            meta["color_mode"] = img.mode
            meta["channels"] = len(img.getbands())

            # Check EXIF GPS info if present in standard JPEGs
            exif = img.getexif()
            if exif:
                gps_info = exif.get_ifd(0x8825)
                if gps_info:
                    # Convert EXIF GPS if available
                    meta["extra_tags"]["has_exif_gps"] = True
    except Exception as e:
        # If standard PIL fails (e.g. 16-bit GeoTIFF), use tifffile
        try:
            arr = tifffile.imread(file_path)
            shape = arr.shape
            if len(shape) == 2:
                meta["height"], meta["width"] = shape
                meta["channels"] = 1
            elif len(shape) >= 3:
                meta["height"], meta["width"] = shape[0], shape[1]
                meta["channels"] = shape[2]
        except Exception:
            pass

    return meta


def load_image_as_rgb(file_path: str, max_dimension: int = 1200) -> Tuple[np.ndarray, Image.Image]:
    """
    Loads any supported satellite image as normalized 8-bit RGB NumPy array and PIL Image.
    Downscales large imagery for fast interactive CV processing while preserving details.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".tif", ".tiff"]:
        try:
            raw = tifffile.imread(file_path)
            if raw.ndim == 2:
                # Grayscale to RGB
                normalized = ((raw - raw.min()) / (max(raw.max() - raw.min(), 1e-5)) * 255).astype(np.uint8)
                rgb_arr = np.stack([normalized] * 3, axis=-1)
            elif raw.ndim == 3:
                if raw.shape[2] >= 3:
                    rgb_part = raw[:, :, :3]
                else:
                    rgb_part = np.stack([raw[:, :, 0]] * 3, axis=-1)

                if rgb_part.dtype != np.uint8:
                    rgb_arr = ((rgb_part - rgb_part.min()) / (max(rgb_part.max() - rgb_part.min(), 1e-5)) * 255).astype(np.uint8)
                else:
                    rgb_arr = rgb_part
            else:
                rgb_arr = np.zeros((512, 512, 3), dtype=np.uint8)

            pil_img = Image.fromarray(rgb_arr)
        except Exception:
            pil_img = Image.open(file_path).convert("RGB")
            rgb_arr = np.array(pil_img)
    else:
        pil_img = Image.open(file_path).convert("RGB")
        rgb_arr = np.array(pil_img)

    # Downscale if exceeding max_dimension
    w, h = pil_img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        rgb_arr = np.array(pil_img)

    return rgb_arr, pil_img
