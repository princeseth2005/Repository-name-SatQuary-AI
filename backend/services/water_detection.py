import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List


def detect_water_bodies(
    rgb_arr: np.ndarray,
    output_dir: str,
    prefix: str = "water"
) -> Tuple[float, str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Detects inland and marine water bodies using optical water indices:
    - RGB NDWI (Normalized Difference Water Index RGB): (G - R) / (G + R)
    - Blue-Red absorption ratio: (B - R) / (B + R)
    - Texture uniformity (water has very low local standard deviation)
    - HSV chromaticity filtering

    Returns:
        (water_percentage, mask_path, details, detected_water_features)
    """
    h, w, _ = rgb_arr.shape
    total_pixels = float(h * w)

    img_f = rgb_arr.astype(np.float32) / 255.0
    r = img_f[:, :, 0]
    g = img_f[:, :, 1]
    b = img_f[:, :, 2]

    # Water has higher blue/green than red, and low overall red reflectance
    denom_gr = g + r
    denom_gr[denom_gr < 1e-5] = 1e-5
    ndwi_rgb = (g - r) / denom_gr

    denom_br = b + r
    denom_br[denom_br < 1e-5] = 1e-5
    blue_ratio = (b - r) / denom_br

    # Convert to grayscale to measure local texture/variance
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian_var = cv2.Laplacian(blur, cv2.CV_32F)
    texture_low = np.abs(laplacian_var) < 15.0

    # HSV values: Water typically has Hue in blue/cyan range (85-135) or low Saturation with low Value
    hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Conditions for water:
    # 1. Deep blue/ocean/lake: Blue > Red and Blue > Green, Hue in [80, 140], Val < 220
    deep_water = (b > r * 1.15) & (b > g * 0.95) & (hue >= 80) & (hue <= 140) & (val < 200)

    # 2. Turbid / silt / shallow river water: Green > Red, Blue high, low brightness
    turbid_water = (ndwi_rgb > 0.08) & (r < 0.45) & (val < 160) & texture_low

    # 3. Very dark water bodies (high absorption in all optical bands)
    dark_water = (val < 50) & (sat > 20) & (np.abs(b - g) < 0.15)

    combined_water = (deep_water | turbid_water | dark_water) & texture_low

    # Clean mask
    water_mask = combined_water.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

    water_pixels = int(np.count_nonzero(water_mask))
    water_percentage = round((water_pixels / total_pixels) * 100.0, 2)

    # Find distinct water contours
    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_features = []

    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > (total_pixels * 0.0008):  # Only significant features (>0.08% of image)
            x, y, cw, ch = cv2.boundingRect(c)
            feat_pct = round((area / total_pixels) * 100.0, 2)
            detected_features.append({
                "id": idx + 1,
                "type": "Water Body",
                "area_percentage": feat_pct,
                "bbox": [round(y / h, 4), round(x / w, 4), round((y + ch) / h, 4), round((x + cw) / w, 4)],
                "aspect_ratio": round(float(cw) / max(float(ch), 1.0), 2)
            })

    # Sort largest first
    detected_features.sort(key=lambda x: x["area_percentage"], reverse=True)

    # Quadrant breakdown
    half_h, half_w = h // 2, w // 2
    nw_water = np.count_nonzero(water_mask[:half_h, :half_w]) / float(half_h * half_w) * 100.0
    ne_water = np.count_nonzero(water_mask[:half_h, half_w:]) / float(half_h * (w - half_w)) * 100.0
    sw_water = np.count_nonzero(water_mask[half_h:, :half_w]) / float((h - half_h) * half_w) * 100.0
    se_water = np.count_nonzero(water_mask[half_h:, half_w:]) / float((h - half_h) * (w - half_w)) * 100.0

    # Categorize water presence
    if water_percentage > 40:
        presence_label = "Extensive Open Water / Coastal or Lake Environment"
    elif water_percentage > 10:
        presence_label = "Prominent River / Estuary / Reservoir System"
    elif water_percentage > 1.5:
        presence_label = "Minor Waterways / Canals / Small Ponds"
    else:
        presence_label = "No Significant Water Bodies Detected"

    # Generate transparent RGBA overlay image (Electric Azure #00B0FF with ~170 alpha)
    overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_indices = water_mask > 0
    overlay_rgba[mask_indices, 0] = 0    # R
    overlay_rgba[mask_indices, 1] = 176  # G (#00B0FF)
    overlay_rgba[mask_indices, 2] = 255  # B
    overlay_rgba[mask_indices, 3] = 170  # Alpha

    # Save overlay mask
    mask_filename = f"{prefix}_mask.png"
    mask_path = os.path.join(output_dir, mask_filename)
    Image.fromarray(overlay_rgba).save(mask_path, format="PNG")

    details = {
        "percentage": water_percentage,
        "presence_label": presence_label,
        "feature_count": len(detected_features),
        "quadrant_distribution": {
            "north_west": round(nw_water, 2),
            "north_east": round(ne_water, 2),
            "south_west": round(sw_water, 2),
            "south_east": round(se_water, 2)
        },
        "total_water_pixels": water_pixels
    }

    return water_percentage, mask_path, details, detected_features
