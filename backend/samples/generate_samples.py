"""
Sample Satellite Imagery Generator for SatQuery AI
Generates realistic multi-class satellite scenes:
1. Agricultural Delta (High vegetation, crop grids, irrigation waterways)
2. Coastal Port (Ocean/water bodies, harbour built-up, beaches, roads)
3. Urban Metropolis (Dense built-up, arterial roads, urban parks, concrete)
4. Temporal Change Pair (T1: Pre-expansion, T2: Post-expansion / deforestation)
5. GeoTIFF Sample with Sriharikota / ISRO Satish Dhawan Space Centre coordinates
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import tifffile

SAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_agricultural_delta(width=800, height=800):
    """Generates a realistic agricultural satellite scene with varied crop fields."""
    np.random.seed(42)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Base background soil/fallow fields
    base_color = np.array([125, 110, 85], dtype=np.float32)
    noise = np.random.normal(0, 10, (height, width, 3))
    img[:] = np.clip(base_color + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Draw irregular field grids
    grid_rows = 8
    grid_cols = 8
    cell_w = width // grid_cols
    cell_h = height // grid_rows

    veg_colors = [
        (34, 139, 34),    # Forest green
        (46, 125, 50),    # Vibrant crop green
        (104, 159, 56),   # Light green pasture
        (56, 142, 60),    # Medium crop
        (139, 195, 74),   # Bright young crop
        (161, 136, 127),  # Harvested fallow field
        (188, 170, 140),  # Bare dry soil
        (27, 94, 32)      # Deep dense canopy
    ]

    for r in range(grid_rows):
        for c in range(grid_cols):
            x1 = c * cell_w + np.random.randint(-10, 10)
            y1 = r * cell_h + np.random.randint(-10, 10)
            x2 = (c + 1) * cell_w + np.random.randint(-10, 10)
            y2 = (r + 1) * cell_h + np.random.randint(-10, 10)
            color = veg_colors[np.random.randint(0, len(veg_colors))]
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(80, 70, 50), width=2)

    # Irrigation canal / meandering river (Water)
    river_points = []
    curr_x = 0
    curr_y = height // 3
    while curr_x < width + 50:
        river_points.append((curr_x, curr_y))
        curr_x += 40
        curr_y += int(np.sin(curr_x / 80.0) * 35 + np.random.randint(-10, 10))

    water_color = (25, 75, 110)
    for i in range(len(river_points) - 1):
        draw.line([river_points[i], river_points[i + 1]], fill=water_color, width=28)

    # Secondary irrigation channels
    draw.line([(width // 2, 0), (width // 2, height)], fill=(30, 85, 120), width=10)
    draw.line([(0, height * 2 // 3), (width, height * 2 // 3)], fill=(30, 85, 120), width=10)

    # Farm houses / Built-up clusters
    for _ in range(12):
        bx = np.random.randint(50, width - 50)
        by = np.random.randint(50, height - 50)
        bw = np.random.randint(15, 35)
        bh = np.random.randint(15, 35)
        draw.rectangle([bx, by, bx + bw, by + bh], fill=(185, 175, 165), outline=(100, 100, 100))

    # Rural dirt roads
    draw.line([(0, 100), (width, 300)], fill=(210, 195, 170), width=6)
    draw.line([(200, 0), (700, height)], fill=(210, 195, 170), width=6)

    # Add realistic satellite sensor blur/texture
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.7))
    out_path = os.path.join(SAMPLES_DIR, "agriculture_delta.jpg")
    pil_img.save(out_path, quality=95)
    return out_path


def generate_coastal_port(width=800, height=800):
    """Generates a realistic coastal port satellite scene with deep water, docks, and urban coast."""
    np.random.seed(101)
    pil_img = Image.new("RGB", (width, height), color=(18, 55, 95))
    draw = ImageDraw.Draw(pil_img)

    # Draw ocean gradient
    for y in range(height):
        ratio = y / height
        r = int(12 + ratio * 15)
        g = int(45 + ratio * 35)
        b = int(85 + ratio * 40)
        draw.line([(0, y), (width // 2 + int(np.sin(y / 60) * 40), y)], fill=(r, g, b))

    # Coastline contour
    coast_x = width // 2
    land_coords = [(coast_x, 0)]
    for y in range(0, height + 40, 20):
        cx = coast_x + int(np.sin(y / 70.0) * 50 + np.cos(y / 30.0) * 20)
        land_coords.append((cx, y))
    land_coords.extend([(width, height), (width, 0)])
    draw.polygon(land_coords, fill=(160, 145, 125))  # Sand / coastal soil

    # Coastal greenery
    for _ in range(25):
        gx = np.random.randint(coast_x + 60, width - 40)
        gy = np.random.randint(30, height - 30)
        gr = np.random.randint(25, 70)
        draw.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], fill=(42, 115, 52))

    # Harbor docks / shipping piers protruding into water
    pier_y_list = [150, 320, 520, 680]
    for py in pier_y_list:
        px_start = coast_x - 120
        px_end = coast_x + 40
        draw.rectangle([px_start, py - 16, px_end, py + 16], fill=(130, 130, 135), outline=(70, 70, 75), width=2)
        # Moored vessels
        draw.rectangle([px_start + 10, py - 32, px_start + 70, py - 18], fill=(220, 50, 45))
        draw.rectangle([px_start + 20, py + 18, px_start + 90, py + 32], fill=(240, 240, 245))

    # Port storage yards and industrial warehouses (Built-up)
    for row in range(5):
        for col in range(4):
            bx = coast_x + 50 + col * 55
            by = 100 + row * 110
            draw.rectangle([bx, by, bx + 45, by + 85], fill=(200, 205, 210), outline=(90, 95, 100), width=2)
            # Roof textures
            if (row + col) % 2 == 0:
                draw.rectangle([bx + 5, by + 5, bx + 40, by + 80], fill=(70, 100, 130))

    # Highway / access roads
    draw.line([(coast_x + 30, 0), (coast_x + 30, height)], fill=(60, 60, 65), width=10)
    draw.line([(coast_x + 30, 320), (width, 320)], fill=(75, 75, 80), width=8)

    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.6))
    out_path = os.path.join(SAMPLES_DIR, "coastal_port.jpg")
    pil_img.save(out_path, quality=95)
    return out_path


def generate_urban_metropolis(width=800, height=800):
    """Generates an urban metropolis with city blocks, road grid, concrete roofs, and central park."""
    np.random.seed(303)
    pil_img = Image.new("RGB", (width, height), color=(140, 138, 135))
    draw = ImageDraw.Draw(pil_img)

    # City blocks grid
    block_size = 70
    road_width = 12

    for x in range(20, width - 20, block_size + road_width):
        for y in range(20, height - 20, block_size + road_width):
            # Each block has multiple buildings
            num_bldgs = 4
            sub_w = block_size // 2
            sub_h = block_size // 2
            for sx in range(2):
                for sy in range(2):
                    bx = x + sx * sub_w
                    by = y + sy * sub_h
                    roof_tones = [
                        (215, 215, 220),  # Bright concrete
                        (180, 175, 170),  # Grey flat roof
                        (160, 120, 110),  # Terracotta roof
                        (90, 110, 130),   # Metal industrial roof
                        (230, 225, 215)   # Light composite
                    ]
                    color = roof_tones[np.random.randint(0, len(roof_tones))]
                    draw.rectangle([bx + 1, by + 1, bx + sub_w - 2, by + sub_h - 2], fill=color, outline=(80, 80, 85))

    # Draw Central Urban Park (Greenery + Lake)
    park_x1, park_y1 = width // 3, height // 3
    park_x2, park_y2 = width * 2 // 3, height * 2 // 3
    draw.rectangle([park_x1, park_y1, park_x2, park_y2], fill=(38, 120, 48), outline=(60, 60, 60), width=3)

    # Dense tree clusters in park
    for _ in range(40):
        tx = np.random.randint(park_x1 + 10, park_x2 - 10)
        ty = np.random.randint(park_y1 + 10, park_y2 - 10)
        tr = np.random.randint(8, 20)
        draw.ellipse([tx - tr, ty - tr, tx + tr, ty + tr], fill=(24, 85, 32))

    # Park artificial lake (Water)
    lake_box = [park_x1 + 40, park_y1 + 50, park_x1 + 140, park_y1 + 150]
    draw.ellipse(lake_box, fill=(20, 65, 95), outline=(50, 90, 110), width=2)

    # Major arterial highways (dark asphalt with median markings)
    draw.line([(0, height // 2), (width, height // 2)], fill=(45, 45, 50), width=18)
    draw.line([(width // 2, 0), (width // 2, height)], fill=(45, 45, 50), width=18)

    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    out_path = os.path.join(SAMPLES_DIR, "urban_metropolis.jpg")
    pil_img.save(out_path, quality=95)
    return out_path


def generate_temporal_pair(width=800, height=800):
    """Generates T1 (Pre-development) and T2 (Post-development) images for change detection testing."""
    np.random.seed(505)

    # 1. Generate T1 (Natural forest with river and small village)
    t1_img = Image.new("RGB", (width, height), color=(34, 112, 45))
    draw1 = ImageDraw.Draw(t1_img)

    # Add lush canopy variations
    for _ in range(80):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        cr = np.random.randint(30, 90)
        draw1.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(np.random.randint(25, 45), np.random.randint(95, 140), np.random.randint(35, 60)))

    # Natural winding river
    river_pts = []
    for x in range(0, width + 50, 40):
        y = int(height // 2 + np.sin(x / 100.0) * 80 + np.cos(x / 50.0) * 30)
        river_pts.append((x, y))
    for i in range(len(river_pts) - 1):
        draw1.line([river_pts[i], river_pts[i + 1]], fill=(22, 70, 105), width=32)

    # Small hamlet in northeast (few small huts)
    for _ in range(6):
        hx = np.random.randint(width * 3 // 4, width - 40)
        hy = np.random.randint(40, height // 4)
        draw1.rectangle([hx, hy, hx + 18, hy + 18], fill=(180, 160, 140))

    t1_img = t1_img.filter(ImageFilter.GaussianBlur(radius=0.6))
    t1_path = os.path.join(SAMPLES_DIR, "temporal_t1.jpg")
    t1_img.save(t1_path, quality=95)

    # 2. Generate T2 (Deforestation, new highway, industrial development in center/south)
    t2_img = t1_img.copy()
    draw2 = ImageDraw.Draw(t2_img)

    # Deforested clearing in central-southwest region
    clear_coords = [
        (100, 450), (380, 420), (450, 680), (120, 720)
    ]
    draw2.polygon(clear_coords, fill=(155, 135, 105), outline=(120, 100, 75))

    # New industrial buildings in cleared zone
    for row in range(3):
        for col in range(3):
            bx = 160 + col * 75
            by = 480 + row * 65
            draw2.rectangle([bx, by, bx + 55, by + 45], fill=(220, 220, 225), outline=(80, 80, 85), width=2)
            draw2.rectangle([bx + 5, by + 5, bx + 50, by + 40], fill=(85, 115, 145))

    # New multi-lane bypass highway slashing through forest
    draw2.line([(0, 250), (width, 620)], fill=(50, 50, 55), width=16)
    # Bridge over the river
    draw2.rectangle([340, 390, 410, 430], fill=(180, 180, 185), outline=(90, 90, 95), width=2)

    t2_img = t2_img.filter(ImageFilter.GaussianBlur(radius=0.6))
    t2_path = os.path.join(SAMPLES_DIR, "temporal_t2.jpg")
    t2_img.save(t2_path, quality=95)

    return t1_path, t2_path


def generate_sample_geotiff(width=600, height=600):
    """
    Generates a GeoTIFF sample centered on ISRO Satish Dhawan Space Centre (SDSC SHAR),
    Sriharikota, Andhra Pradesh (Lat: 13.72° N, Lon: 80.23° E).
    """
    # Create multi-channel optical array (RGB + simulated NIR band)
    np.random.seed(707)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Island/launch complex geography
    # Ocean on right (East), land on left (West)
    for y in range(height):
        for x in range(width):
            if x > width * 0.65:
                rgb[y, x] = [20, 65, 105]  # Bay of Bengal water
            elif x < width * 0.2:
                rgb[y, x] = [35, 80, 110]  # Pulicat lagoon water
            else:
                # Barrier island with scrub vegetation and launch pads
                if 240 < y < 360 and 260 < x < 380:
                    rgb[y, x] = [210, 205, 200]  # Launch complex concrete pad
                elif (x - 310)**2 + (y - 300)**2 < 400:
                    rgb[y, x] = [180, 50, 40]   # Assembly tower / gantry
                else:
                    rgb[y, x] = [48, 125, 55]   # Island vegetation & coastal reserve

    out_path = os.path.join(SAMPLES_DIR, "isro_sriharikota_geotiff.tif")

    # Write TIFF with baseline metadata
    # Coordinates for Sriharikota: MinLon=80.20, MaxLon=80.26, MinLat=13.69, MaxLat=13.75
    extratags = [
        (33550, 'd', 3, (0.0001, 0.0001, 0.0)),  # ModelPixelScaleTag
        (33922, 'd', 6, (0.0, 0.0, 0.0, 80.20, 13.75, 0.0)),  # ModelTiepointTag
    ]

    tifffile.imwrite(
        out_path,
        rgb,
        photometric='rgb',
        extratags=extratags
    )
    return out_path


def generate_all_samples():
    print("Generating agricultural delta sample...")
    generate_agricultural_delta()
    print("Generating coastal port sample...")
    generate_coastal_port()
    print("Generating urban metropolis sample...")
    generate_urban_metropolis()
    print("Generating temporal change pair (T1 & T2)...")
    generate_temporal_pair()
    print("Generating ISRO Sriharikota GeoTIFF sample...")
    generate_sample_geotiff()
    print("All sample datasets successfully generated in:", SAMPLES_DIR)


if __name__ == "__main__":
    generate_all_samples()
