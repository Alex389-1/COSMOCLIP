"""ArcGIS World Imagery Wayback Historical Satellite Acquisition Engine.

Provides zero-key access to over 190+ historical high-resolution global satellite/aerial
imagery releases dating from 2014 to the present day using the official ESRI ArcGIS
Wayback WMTS (Web Map Tile Service) tile archive.
"""

import datetime
import io
import logging
import math
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageEnhance

logger = logging.getLogger("riva.vqa.wayback")

USER_AGENT = "COSMOCLIP/1.2 (https://cosmoclip.ai; contact: dev@cosmoclip.ai)"

# Wayback WMTS tile template
WAYBACK_TILE_TEMPLATE = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
    "default028mm/MapServer/tile/{release_num}/{zoom}/{row}/{col}"
)

# Curated catalog of major ArcGIS Wayback releases across the 2014-2026 archive
WAYBACK_RELEASES: List[Tuple[str, str]] = [
    ("2014-02-20", "10"),
    ("2014-06-11", "1431"),
    ("2014-12-03", "2730"),
    ("2015-05-13", "3026"),
    ("2015-09-02", "3515"),
    ("2016-01-20", "3630"),
    ("2016-06-08", "4230"),
    ("2017-01-18", "5232"),
    ("2017-07-05", "5844"),
    ("2018-01-17", "6354"),
    ("2018-08-01", "8781"),
    ("2019-01-09", "9203"),
    ("2019-06-19", "10443"),
    ("2019-10-30", "10850"),
    ("2020-02-19", "11019"),
    ("2020-05-27", "11033"),
    ("2020-08-12", "11092"),
    ("2020-11-18", "11262"),
    ("2021-03-17", "11952"),
    ("2021-07-21", "14720"),
    ("2021-12-08", "15084"),
    ("2022-04-13", "16513"),
    ("2022-09-28", "19085"),
    ("2023-03-15", "19819"),
    ("2023-08-09", "19930"),
    ("2024-01-17", "20222"),
    ("2024-05-22", "20443"),
    ("2024-09-18", "22692"),
    ("2025-02-19", "23383"),
    ("2025-06-18", "23880"),
    ("2025-10-22", "24007"),
    ("2026-02-18", "25586"),
    ("2026-06-30", "31144"),
    ("2026-08-05", "32337"),
]


def find_closest_wayback_release(target_date_str: str) -> Tuple[str, str]:
    """Finds the closest historical ArcGIS Wayback release date and release number.

    Args:
        target_date_str: Target date string (e.g. '2019-05', '2021', '2023-08-31', 'past year').

    Returns:
        Tuple of (actual_release_date, release_number).
    """
    cleaned = target_date_str.strip().lower()

    # Parse year or YYYY-MM-DD
    year_match = re.search(r"\b(201[4-9]|202[0-6])\b", cleaned)
    target_dt: Optional[datetime.datetime] = None

    # Try full ISO date
    date_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", cleaned)
    if date_match:
        try:
            target_dt = datetime.datetime(
                int(date_match.group(1)),
                int(date_match.group(2)),
                int(date_match.group(3)),
            )
        except ValueError:
            target_dt = None

    if not target_dt and year_match:
        target_dt = datetime.datetime(int(year_match.group(1)), 6, 1)

    if not target_dt:
        if "past" in cleaned or "previous" in cleaned or "old" in cleaned:
            # Default to ~2 years past
            target_dt = datetime.datetime.now() - datetime.timedelta(days=730)
        else:
            # Latest release
            return WAYBACK_RELEASES[-1]

    # Find closest release by timedelta
    best_diff = float("inf")
    best_release = WAYBACK_RELEASES[-1]

    for rel_date_str, rel_num in WAYBACK_RELEASES:
        rel_dt = datetime.datetime.strptime(rel_date_str, "%Y-%m-%d")
        diff = abs((rel_dt - target_dt).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_release = (rel_date_str, rel_num)

    return best_release


def latlon_to_tile_float(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    """Converts geographic WGS84 coordinates to fractional Slippy Map tile numbers."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = (lon + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def fetch_wayback_satellite_image_sync(
    lat: float,
    lon: float,
    release_num: str,
    output_path: Path,
    zoom: int = 16,
) -> bool:
    """Fetches and stitches a precision dead-centered 3x3 grid of historical ArcGIS Wayback tiles into a 1024x1024 scene.

    Args:
        lat: Center latitude.
        lon: Center longitude.
        release_num: ArcGIS Wayback release ID.
        output_path: Target PNG/JPG path.
        zoom: Web mercator zoom level (default 16 for ~1m/pixel crisp remote sensing).

    Returns:
        True if successfully fetched and saved, False otherwise.
    """
    center_x, center_y = latlon_to_tile_float(lat, lon, zoom)
    base_x = int(math.floor(center_x))
    base_y = int(math.floor(center_y))

    # Fractional pixel offset of target inside center tile
    px_in_tile = int((center_x - base_x) * 256)
    py_in_tile = int((center_y - base_y) * 256)

    # 3x3 tile grid (768x768 pixels total)
    grid = Image.new("RGB", (768, 768), color=(20, 25, 30))
    tiles_fetched = 0

    for dy, y_off in enumerate([-1, 0, 1]):
        for dx, x_off in enumerate([-1, 0, 1]):
            tx = base_x + x_off
            ty = base_y + y_off
            url = WAYBACK_TILE_TEMPLATE.format(
                release_num=release_num,
                zoom=zoom,
                row=ty,
                col=tx,
            )
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    tile_bytes = resp.read()
                    if len(tile_bytes) > 400:
                        tile_img = Image.open(io.BytesIO(tile_bytes)).convert("RGB")
                        grid.paste(tile_img, (dx * 256, dy * 256))
                        tiles_fetched += 1
            except Exception as e:
                logger.warning(f"Failed to fetch Wayback tile {url}: {e}")

    if tiles_fetched == 0:
        return False

    # Center target coordinate in the 768x768 grid
    target_gx = 256 + px_in_tile
    target_gy = 256 + py_in_tile

    # Crop 512x512 box centered exactly on the target coordinates
    left = max(0, min(grid.width - 512, target_gx - 256))
    top = max(0, min(grid.height - 512, target_gy - 256))
    cropped = grid.crop((left, top, left + 512, top + 512))

    # Upscale and enhance clarity
    canvas_hd = cropped.resize((1024, 1024), Image.Resampling.LANCZOS)
    canvas_hd = ImageEnhance.Sharpness(canvas_hd).enhance(1.25)
    canvas_hd = ImageEnhance.Contrast(canvas_hd).enhance(1.08)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_hd.save(output_path, format="PNG")
    logger.info(f"Saved precision dead-centered ArcGIS Wayback satellite scene ({tiles_fetched}/9 tiles) -> {output_path}")
    return True
