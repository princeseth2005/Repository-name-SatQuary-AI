import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List
from backend.services.image_processor import load_image_as_rgb


def analyze_change(
    image_path_1: str,
    image_path_2: str,
    output_dir: str,
    prefix: str = "change"
) -> Tuple[float, str, str, Dict[str, Any], List[Dict[str, Any]], str]:
    """
    Performs temporal change detection between two satellite images (Time 1 and Time 2):
    1. Resamples and aligns both images to common grid dimensions
    2. Computes multi-spectral / CIE-Lab Delta-E perceptual differences
    3. Structural gradient difference to catch infrastructure & deforestation changes
    4. Morphological noise filtering to suppress transient atmospheric variations
    5. Contouring and bounding box extraction for primary change hotspots
    6. Transparent fuchsia change overlay + side-by-side composite

    Returns:
        (change_percentage, change_mask_path, side_by_side_path, details, change_features, natural_language_summary)
    """
    rgb1, _ = load_image_as_rgb(image_path_1)
    rgb2, _ = load_image_as_rgb(image_path_2)

    # 1. Resize to common geometry
    h = min(rgb1.shape[0], rgb2.shape[0])
    w = min(rgb1.shape[1], rgb2.shape[1])
    img1 = cv2.resize(rgb1, (w, h), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(rgb2, (w, h), interpolation=cv2.INTER_AREA)

    total_pixels = float(h * w)

    # 2. CIE-Lab Delta-E color difference
    lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB).astype(np.float32)

    delta_l = lab1[:, :, 0] - lab2[:, :, 0]
    delta_a = lab1[:, :, 1] - lab2[:, :, 1]
    delta_b = lab1[:, :, 2] - lab2[:, :, 2]
    delta_e = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)

    # 3. Edge / gradient difference (captures newly constructed roads, cleared plots)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    sobel1 = cv2.Sobel(gray1, cv2.CV_32F, 1, 1)
    sobel2 = cv2.Sobel(gray2, cv2.CV_32F, 1, 1)
    sobel_diff = np.abs(sobel1 - sobel2)

    # 4. Threshold significant changes (Delta E > 28 or significant structural alteration)
    change_raw = (delta_e > 26.0) | (sobel_diff > 45.0)

    # Clean with morphological operations
    change_mask = change_raw.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)

    change_pixels = int(np.count_nonzero(change_mask))
    change_percentage = round((change_pixels / total_pixels) * 100.0, 2)

    # 5. Extract significant changed polygons
    contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    change_features = []

    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > (total_pixels * 0.0012):  # > 0.12% of total area
            x, y, cw, ch = cv2.boundingRect(c)
            feat_pct = round((area / total_pixels) * 100.0, 2)
            change_features.append({
                "id": idx + 1,
                "type": "Altered Surface Region",
                "area_percentage": feat_pct,
                "bbox": [round(y / h, 4), round(x / w, 4), round((y + ch) / h, 4), round((x + cw) / w, 4)]
            })

    change_features.sort(key=lambda x: x["area_percentage"], reverse=True)

    # Quadrant breakdown
    half_h, half_w = h // 2, w // 2
    nw_ch = np.count_nonzero(change_mask[:half_h, :half_w]) / float(half_h * half_w) * 100.0
    ne_ch = np.count_nonzero(change_mask[:half_h, half_w:]) / float(half_h * (w - half_w)) * 100.0
    sw_ch = np.count_nonzero(change_mask[half_h:, :half_w]) / float((h - half_h) * half_w) * 100.0
    se_ch = np.count_nonzero(change_mask[half_h:, half_w:]) / float((h - half_h) * (w - half_w)) * 100.0

    quadrants = {
        "North-West": round(nw_ch, 2),
        "North-East": round(ne_ch, 2),
        "South-West": round(sw_ch, 2),
        "South-East": round(se_ch, 2)
    }

    # Find top changed sectors
    sorted_quads = sorted(quadrants.items(), key=lambda x: x[1], reverse=True)
    top_quads = [q[0] for q in sorted_quads if q[1] > 2.0]
    quad_desc = " and ".join(top_quads[:2]) if top_quads else "isolated uniform patches"

    # 6. Synthesize natural language explanation
    if change_percentage > 25.0:
        severity = "high-intensity surface transformation"
    elif change_percentage > 10.0:
        severity = "moderate regional land-cover change"
    elif change_percentage > 2.0:
        severity = "localized localized surface modifications"
    else:
        severity = "minimal visual change, mostly within baseline environmental tolerance"

    summary_nl = (
        f"Approximately {change_percentage}% of the analyzed image area exhibits significant visual change. "
        f"The comparison reveals {severity}, with primary hotspots concentrated across the {quad_desc} "
        f"({len(change_features)} distinct altered sectors identified). "
        f"Notice: This is an optical image-based visual difference analysis and should be corroborated with calibrated remote sensing products."
    )

    # 7. Generate transparent RGBA change overlay (Electric Fuchsia/Magenta #E040FB with ~170 alpha)
    overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    ch_indices = change_mask > 0
    overlay_rgba[ch_indices, 0] = 224  # R (#E040FB)
    overlay_rgba[ch_indices, 1] = 64   # G
    overlay_rgba[ch_indices, 2] = 251  # B
    overlay_rgba[ch_indices, 3] = 170  # Alpha

    mask_filename = f"{prefix}_mask.png"
    mask_path = os.path.join(output_dir, mask_filename)
    Image.fromarray(overlay_rgba).save(mask_path, format="PNG")

    # 8. Create Side-by-Side comparison image with labeled divider
    divider_w = 6
    composite_w = w * 2 + divider_w
    composite = np.zeros((h, composite_w, 3), dtype=np.uint8)
    composite[:, :w] = img1
    composite[:, w:w + divider_w] = [255, 215, 0]  # Gold divider
    composite[:, w + divider_w:] = img2

    side_by_side_filename = f"{prefix}_side_by_side.jpg"
    side_by_side_path = os.path.join(output_dir, side_by_side_filename)
    Image.fromarray(composite).save(side_by_side_path, quality=90)

    details = {
        "change_percentage": change_percentage,
        "quadrant_changes": quadrants,
        "primary_quadrant": sorted_quads[0][0],
        "significant_clusters_count": len(change_features),
        "total_changed_pixels": change_pixels
    }

    return change_percentage, mask_path, side_by_side_path, details, change_features, summary_nl
