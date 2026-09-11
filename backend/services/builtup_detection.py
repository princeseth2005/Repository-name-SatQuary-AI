import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List


def detect_builtup_and_roads(
    rgb_arr: np.ndarray,
    output_dir: str,
    veg_mask_path: str = None,
    water_mask_path: str = None,
    prefix: str = "builtup"
) -> Tuple[float, float, str, str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Detects built-up, urban infrastructure, and linear road networks using:
    - High-frequency spatial gradients (Sobel / Laplacian edge density)
    - Texture variance (Gray-Level morphological features)
    - Spectral exclusion of water bodies and dense vegetation
    - Canny + morphological skeletonization for road corridors

    Returns:
        (builtup_percentage, road_percentage, builtup_mask_path, road_mask_path, details, detected_clusters)
    """
    h, w, _ = rgb_arr.shape
    total_pixels = float(h * w)

    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

    # 1. Edge & Texture Analysis
    edges = cv2.Canny(gray, 40, 130)

    # Compute local edge density by averaging over a 15x15 window
    edge_density = cv2.boxFilter(edges.astype(np.float32) / 255.0, -1, (15, 15))

    # Laplacian variance across 9x9 window
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    lap_density = cv2.boxFilter(lap, -1, (9, 9))

    # HSV characteristics for built-up: low-to-moderate saturation, light-to-medium value
    hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Built-up conditions:
    # Noticeable texture/edge concentration, moderate brightness, not deeply saturated
    builtup_raw = ((edge_density > 0.08) | (lap_density > 14.0)) & (sat < 95) & (val > 65)

    # If masks for vegetation and water already exist, exclude them
    if veg_mask_path and os.path.exists(veg_mask_path):
        veg_img = np.array(Image.open(veg_mask_path))
        if veg_img.ndim == 3 and veg_img.shape[2] == 4:
            builtup_raw[veg_img[:, :, 3] > 50] = False

    if water_mask_path and os.path.exists(water_mask_path):
        water_img = np.array(Image.open(water_mask_path))
        if water_img.ndim == 3 and water_img.shape[2] == 4:
            builtup_raw[water_img[:, :, 3] > 50] = False

    # Morphological grouping
    builtup_mask = builtup_raw.astype(np.uint8) * 255
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    builtup_mask = cv2.morphologyEx(builtup_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    builtup_mask = cv2.morphologyEx(builtup_mask, cv2.MORPH_OPEN, kernel_open)

    builtup_pixels = int(np.count_nonzero(builtup_mask))
    builtup_percentage = round((builtup_pixels / total_pixels) * 100.0, 2)

    # 2. Road / Linear Corridor Detection
    # Long contiguous edges with low width
    kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
    kernel_line_w = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    roads_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_line_h)
    roads_w = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_line_w)
    combined_roads = cv2.bitwise_or(roads_h, roads_w)
    combined_roads = cv2.dilate(combined_roads, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    road_pixels = int(np.count_nonzero(combined_roads))
    road_percentage = round((road_pixels / total_pixels) * 100.0, 2)

    # Extract major built-up clusters
    contours, _ = cv2.findContours(builtup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_clusters = []

    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > (total_pixels * 0.001):  # > 0.1% of scene
            x, y, cw, ch = cv2.boundingRect(c)
            feat_pct = round((area / total_pixels) * 100.0, 2)
            detected_clusters.append({
                "id": idx + 1,
                "type": "Built-up Cluster",
                "area_percentage": feat_pct,
                "bbox": [round(y / h, 4), round(x / w, 4), round((y + ch) / h, 4), round((x + cw) / w, 4)]
            })

    detected_clusters.sort(key=lambda x: x["area_percentage"], reverse=True)

    # Quadrant breakdown
    half_h, half_w = h // 2, w // 2
    nw_bu = np.count_nonzero(builtup_mask[:half_h, :half_w]) / float(half_h * half_w) * 100.0
    ne_bu = np.count_nonzero(builtup_mask[:half_h, half_w:]) / float(half_h * (w - half_w)) * 100.0
    sw_bu = np.count_nonzero(builtup_mask[half_h:, :half_w]) / float((h - half_h) * half_w) * 100.0
    se_bu = np.count_nonzero(builtup_mask[half_h:, half_w:]) / float((h - half_h) * (w - half_w)) * 100.0

    # Categorize urban density
    if builtup_percentage > 45:
        density_label = "High-Density Urban / Commercial / Industrial District"
    elif builtup_percentage > 18:
        density_label = "Medium-Density Residential / Mixed Suburban Area"
    elif builtup_percentage > 3:
        density_label = "Low-Density Rural Settlement / Isolated Structures"
    else:
        density_label = "Uninhabited / Minimal Built-up Footprint"

    # Transparent RGBA overlay for built-up (Orange/Coral #FF5722 with ~160 alpha)
    overlay_builtup = np.zeros((h, w, 4), dtype=np.uint8)
    bu_indices = builtup_mask > 0
    overlay_builtup[bu_indices, 0] = 255  # R (#FF5722)
    overlay_builtup[bu_indices, 1] = 87   # G
    overlay_builtup[bu_indices, 2] = 34   # B
    overlay_builtup[bu_indices, 3] = 160  # Alpha

    bu_mask_path = os.path.join(output_dir, f"{prefix}_mask.png")
    Image.fromarray(overlay_builtup).save(bu_mask_path, format="PNG")

    # Transparent RGBA overlay for roads (Electric Yellow #FFD600 with ~180 alpha)
    overlay_roads = np.zeros((h, w, 4), dtype=np.uint8)
    road_indices = combined_roads > 0
    overlay_roads[road_indices, 0] = 255  # R (#FFD600)
    overlay_roads[road_indices, 1] = 214  # G
    overlay_roads[road_indices, 2] = 0    # B
    overlay_roads[road_indices, 3] = 180  # Alpha

    road_mask_path = os.path.join(output_dir, f"{prefix}_roads_mask.png")
    Image.fromarray(overlay_roads).save(road_mask_path, format="PNG")

    details = {
        "builtup_percentage": builtup_percentage,
        "road_percentage": road_percentage,
        "density_label": density_label,
        "cluster_count": len(detected_clusters),
        "quadrant_distribution": {
            "north_west": round(nw_bu, 2),
            "north_east": round(ne_bu, 2),
            "south_west": round(sw_bu, 2),
            "south_east": round(se_bu, 2)
        },
        "total_builtup_pixels": builtup_pixels
    }

    return builtup_percentage, road_percentage, bu_mask_path, road_mask_path, details, detected_clusters
