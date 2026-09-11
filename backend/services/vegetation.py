import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple


def analyze_vegetation(
    rgb_arr: np.ndarray,
    output_dir: str,
    prefix: str = "veg"
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Computes vegetation estimation using optical remote sensing color indices:
    - VARI (Visible Atmospherically Resistant Index): (G - R) / (G + R - B)
    - ExG (Excess Green Index): 2*G - R - B
    - GLI (Green Leaf Index): (2*G - R - B) / (2*G + R + B)

    Returns:
        (vegetation_percentage, mask_path, detailed_metrics)
    """
    h, w, _ = rgb_arr.shape
    total_pixels = float(h * w)

    # Convert to float32 normalized [0, 1]
    img_f = rgb_arr.astype(np.float32) / 255.0
    r = img_f[:, :, 0]
    g = img_f[:, :, 1]
    b = img_f[:, :, 2]

    # VARI calculation
    denom = g + r - b
    denom[np.abs(denom) < 1e-5] = 1e-5
    vari = (g - r) / denom

    # Excess Green Index (ExG)
    exg = 2.0 * g - r - b

    # Green Leaf Index (GLI)
    denom_gli = 2.0 * g + r + b
    denom_gli[np.abs(denom_gli) < 1e-5] = 1e-5
    gli = (2.0 * g - r - b) / denom_gli

    # HSV saturation and hue filtering to eliminate water or blue roofs
    hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]  # OpenCV Hue: 0-180
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Green hue in OpenCV is roughly 35 to 85
    green_hue_mask = (hue >= 32) & (hue <= 88) & (sat >= 35) & (val >= 35)

    # Multi-condition vegetation classification
    # Pixels where VARI > 0.08 or (GLI > 0.05 and green_hue_mask)
    veg_condition = ((vari > 0.08) & (exg > 0.02)) | (green_hue_mask & (gli > 0.03))

    # Clean with morphological opening/closing
    veg_mask = veg_condition.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kernel)
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, kernel)

    veg_pixels = int(np.count_nonzero(veg_mask))
    veg_percentage = round((veg_pixels / total_pixels) * 100.0, 2)

    # Spatial quadrant breakdown
    half_h, half_w = h // 2, w // 2
    nw_veg = np.count_nonzero(veg_mask[:half_h, :half_w]) / float(half_h * half_w) * 100.0
    ne_veg = np.count_nonzero(veg_mask[:half_h, half_w:]) / float(half_h * (w - half_w)) * 100.0
    sw_veg = np.count_nonzero(veg_mask[half_h:, :half_w]) / float((h - half_h) * half_w) * 100.0
    se_veg = np.count_nonzero(veg_mask[half_h:, half_w:]) / float((h - half_h) * (w - half_w)) * 100.0

    # Categorize canopy density
    if veg_percentage > 55:
        density_label = "Dense Canopy / Intensive Forest & Agricultural Zone"
    elif veg_percentage > 25:
        density_label = "Moderate Vegetation / Mixed Agro-Rural or Parkland"
    elif veg_percentage > 8:
        density_label = "Sparse Scrub / Scattered Vegetative Patches"
    else:
        density_label = "Minimal to Negligible Vegetation"

    # Generate transparent RGBA overlay image (Vibrant Green #00E676 with ~160 alpha)
    overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_indices = veg_mask > 0
    overlay_rgba[mask_indices, 0] = 0    # R
    overlay_rgba[mask_indices, 1] = 230  # G (#00E676)
    overlay_rgba[mask_indices, 2] = 118  # B
    overlay_rgba[mask_indices, 3] = 160  # Alpha

    # Save overlay mask
    mask_filename = f"{prefix}_mask.png"
    mask_path = os.path.join(output_dir, mask_filename)
    Image.fromarray(overlay_rgba).save(mask_path, format="PNG")

    details = {
        "percentage": veg_percentage,
        "density_label": density_label,
        "quadrant_distribution": {
            "north_west": round(nw_veg, 2),
            "north_east": round(ne_veg, 2),
            "south_west": round(sw_veg, 2),
            "south_east": round(se_veg, 2)
        },
        "mean_vari": round(float(np.mean(vari[mask_indices])) if np.any(mask_indices) else 0.0, 3),
        "mean_gli": round(float(np.mean(gli[mask_indices])) if np.any(mask_indices) else 0.0, 3),
        "total_vegetated_pixels": veg_pixels
    }

    return veg_percentage, mask_path, details
