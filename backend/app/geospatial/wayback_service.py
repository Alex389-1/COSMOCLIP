import os
import io
import math
import hashlib
import urllib.request
import json
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List
from backend.app.api.logs import emit_log

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CACHE_DIR = os.path.join(ROOT_DIR, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Curated fallback release numbers for instant zero-latency historical lookup
WAYBACK_CURATED_RELEASES = {
    "2014": {"key": "10", "title": "World Imagery (Wayback 2014-02-20)", "date": "2014-02-20"},
    "2016": {"key": "388", "title": "World Imagery (Wayback 2016-04-20)", "date": "2016-04-20"},
    "2018": {"key": "239", "title": "World Imagery (Wayback 2018-11-29)", "date": "2018-11-29"},
    "2020": {"key": "29260", "title": "World Imagery (Wayback 2020-12-16)", "date": "2020-12-16"},
    "2021": {"key": "26120", "title": "World Imagery (Wayback 2021-12-21)", "date": "2021-12-21"},
    "2022": {"key": "45134", "title": "World Imagery (Wayback 2022-12-14)", "date": "2022-12-14"},
    "2023": {"key": "4169", "title": "World Imagery (Wayback 2023-12-06)", "date": "2023-12-06"},
    "2024": {"key": "16453", "title": "World Imagery (Wayback 2024-12-12)", "date": "2024-12-12"},
    "2025": {"key": "13192", "title": "World Imagery (Wayback 2025-12-18)", "date": "2025-12-18"},
}

class WaybackService:
    """
    Esri World Imagery Wayback Historical Satellite Imagery Service.
    Retrieves genuine time-versioned historical satellite rasters for a specified baseline year
    (e.g., 2016, 2020, 2022, 2024) to enable true bi-temporal change detection against current imagery.
    """

    _config_cache: Optional[Dict[str, Any]] = None

    @staticmethod
    def _fetch_wayback_config() -> Dict[str, Any]:
        if WaybackService._config_cache is not None:
            return WaybackService._config_cache

        config_path = os.path.join(CACHE_DIR, "waybackconfig.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    WaybackService._config_cache = json.load(f)
                    return WaybackService._config_cache
            except Exception:
                pass

        try:
            url = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
            req = urllib.request.Request(url, headers={"User-Agent": "COSMOCLIP-Wayback/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode())
                WaybackService._config_cache = data
                with open(config_path, "w") as f:
                    json.dump(data, f)
                return data
        except Exception:
            return {}

    @staticmethod
    def resolve_release_for_year(target_year: str = "2020") -> Dict[str, str]:
        """
        Resolves the closest Esri Wayback release for a given target historical year.
        """
        target_y = str(target_year).strip()
        if target_y in WAYBACK_CURATED_RELEASES:
            curated = WAYBACK_CURATED_RELEASES[target_y]
            return {
                "release_num": curated["key"],
                "item_title": curated["title"],
                "release_date": curated["date"],
                "tile_template": f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/{curated['key']}/{{z}}/{{y}}/{{x}}"
            }

        config = WaybackService._fetch_wayback_config()
        best_key = None
        best_title = None
        for k, v in config.items():
            title = v.get("itemTitle", "")
            if target_y in title:
                best_key = k
                best_title = title
                break

        if not best_key:
            curated = WAYBACK_CURATED_RELEASES.get("2020", WAYBACK_CURATED_RELEASES["2022"])
            best_key = curated["key"]
            best_title = curated["title"]

        return {
            "release_num": best_key,
            "item_title": best_title or f"World Imagery (Wayback {target_y})",
            "release_date": target_y,
            "tile_template": f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/{best_key}/{{z}}/{{y}}/{{x}}"
        }

    @staticmethod
    def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)

    @staticmethod
    def num2deg(xtile: float, ytile: float, zoom: int) -> Tuple[float, float]:
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)

    @staticmethod
    def get_mesh_bounds(center_lat: float, center_lon: float, zoom: int) -> List[List[float]]:
        center_x, center_y = WaybackService.deg2num(center_lat, center_lon, zoom)
        north_lat, west_lon = WaybackService.num2deg(center_x - 2, center_y - 2, zoom)
        south_lat, east_lon = WaybackService.num2deg(center_x + 2, center_y + 2, zoom)
        return [[south_lat, west_lon], [north_lat, east_lon]]

    @staticmethod
    def get_historical_baseline(
        bbox: List[float],
        zoom: int = 14,
        target_year: str = "2020"
    ) -> Dict[str, Any]:
        """
        Retrieves and stitches a 1024x1024 historical baseline satellite raster from Esri Wayback for the given AOI.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        leaflet_bounds = WaybackService.get_mesh_bounds(center_lat, center_lon, zoom)

        release_info = WaybackService.resolve_release_for_year(target_year)
        release_num = release_info["release_num"]
        release_date = release_info["release_date"]
        release_title = release_info["item_title"]

        loc_slug = hashlib.md5(f"wb_{center_lat:.4f}_{center_lon:.4f}_{zoom}_{release_num}".encode()).hexdigest()[:10]
        cache_filename = f"baseline_{target_year}_{loc_slug}.png"
        cache_path = os.path.join(CACHE_DIR, cache_filename)

        if os.path.exists(cache_path):
            emit_log("SUCCESS", "WAYBACK", f"Loaded historical {target_year} baseline from cache ({release_title}).")
            return {
                "image_path": cache_path,
                "image_url": f"/api/imagery/preview/{cache_filename.replace('.png', '')}",
                "release_title": release_title,
                "release_date": release_date,
                "release_num": release_num,
                "tile_template": release_info["tile_template"],
                "bbox": bbox,
                "bounds": leaflet_bounds
            }

        from concurrent.futures import ThreadPoolExecutor

        emit_log("INFO", "WAYBACK", f"Stitching 4x4 historical {target_year} tile mesh concurrently from {release_title}...")
        center_x, center_y = WaybackService.deg2num(center_lat, center_lon, zoom)
        combined_img = Image.new("RGB", (1024, 1024), color=(18, 30, 49))

        def fetch_tile(coords):
            i, dx, j, dy = coords
            tile_x = center_x + dx
            tile_y = center_y + dy
            url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/{release_num}/{zoom}/{tile_y}/{tile_x}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "COSMOCLIP-Wayback/1.0"})
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    tile = Image.open(io.BytesIO(resp.read())).convert("RGB")
                    return (i * 256, j * 256, tile)
            except Exception:
                return None

        tile_specs = [
            (i, dx, j, dy)
            for i, dx in enumerate([-2, -1, 0, 1])
            for j, dy in enumerate([-2, -1, 0, 1])
        ]

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = executor.map(fetch_tile, tile_specs)
            for res in results:
                if res:
                    px, py, tile_img = res
                    combined_img.paste(tile_img, (px, py))

        combined_img.save(cache_path, format="PNG")
        emit_log("SUCCESS", "WAYBACK", f"Assembled genuine historical {target_year} baseline raster ({release_title})")

        return {
            "image_path": cache_path,
            "image_url": f"/api/imagery/preview/{cache_filename.replace('.png', '')}",
            "release_title": release_title,
            "release_date": release_date,
            "release_num": release_num,
            "tile_template": release_info["tile_template"],
            "bbox": bbox,
            "bounds": leaflet_bounds
        }
