import os
import json
import httpx
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from backend.app.schemas.scene import SceneMetadata, BoundingBox

CDSE_STAC_URL = os.getenv("CDSE_STAC_URL", "https://catalogue.dataspace.copernicus.eu/stac/search")

# Predefined high-value demo scenes covering distinct remote sensing features
SAMPLE_SCENES: List[Dict[str, Any]] = [
    {
        "scene_id": "S2A_MSIL2A_20260814T054641_N0500_R120_T43RER_20260814T084512",
        "scene_name": "Lake Pichola & Udaipur Basin (Water & Urban Mixed)",
        "collection": "sentinel-2-l2a",
        "acquisition_time": "2026-08-14T05:46:41.024Z",
        "cloud_cover": 2.4,
        "bbox": [73.65, 24.55, 73.72, 24.62],
        "crs": "EPSG:4326",
        "resolution_m": 10.0,
        "provider": "Copernicus Data Space Ecosystem (CDSE)",
        "thumbnail_url": "/api/imagery/preview/lake_pichola_s2",
        "local_asset_name": "lake_pichola_s2.png",
        "features": {
            "dominant_classes": ["water_body", "urban_settlement", "hilly_terrain"],
            "water_body_name": "Lake Pichola & Fateh Sagar",
            "has_large_water": True,
            "surrounding_type": "Dense historic urban settlement with surrounding Aravalli hills and scrubland."
        }
    },
    {
        "scene_id": "S2B_MSIL2A_20260818T053649_N0500_R063_T44QND_20260818T091024",
        "scene_name": "Hussain Sagar & Hyderabad Urban Core",
        "collection": "sentinel-2-l2a",
        "acquisition_time": "2026-08-18T05:36:49.027Z",
        "cloud_cover": 4.1,
        "bbox": [78.44, 17.40, 78.50, 17.46],
        "crs": "EPSG:4326",
        "resolution_m": 10.0,
        "provider": "Copernicus Data Space Ecosystem (CDSE)",
        "thumbnail_url": "/api/imagery/preview/hussain_sagar_s2",
        "local_asset_name": "hussain_sagar_s2.png",
        "features": {
            "dominant_classes": ["water_body", "dense_urban", "commercial_infrastructure"],
            "water_body_name": "Hussain Sagar Lake",
            "has_large_water": True,
            "surrounding_type": "High-density metropolitan built-up area with major road networks and parks."
        }
    },
    {
        "scene_id": "S2A_MSIL2A_20260821T055631_N0500_R020_T43RDK_20260821T090115",
        "scene_name": "Sabarmati River & Ahmedabad Urban Corridor",
        "collection": "sentinel-2-l2a",
        "acquisition_time": "2026-08-21T05:56:31.024Z",
        "cloud_cover": 1.2,
        "bbox": [72.54, 23.00, 72.62, 23.08],
        "crs": "EPSG:4326",
        "resolution_m": 10.0,
        "provider": "Copernicus Data Space Ecosystem (CDSE)",
        "thumbnail_url": "/api/imagery/preview/sabarmati_ahmedabad_s2",
        "local_asset_name": "sabarmati_ahmedabad_s2.png",
        "features": {
            "dominant_classes": ["river_corridor", "urban_grid", "industrial"],
            "water_body_name": "Sabarmati River",
            "has_large_water": True,
            "surrounding_type": "Structured urban grid, embankments, bridges, and commercial developments."
        }
    },
    {
        "scene_id": "S2B_MSIL2A_20260805T043659_N0500_R133_T45QXF_20260805T081544",
        "scene_name": "Sundarbans Mangrove Delta & Tidal Estuaries",
        "collection": "sentinel-2-l2a",
        "acquisition_time": "2026-08-05T04:36:59.028Z",
        "cloud_cover": 8.7,
        "bbox": [88.80, 21.80, 88.95, 21.95],
        "crs": "EPSG:4326",
        "resolution_m": 10.0,
        "provider": "Copernicus Data Space Ecosystem (CDSE)",
        "thumbnail_url": "/api/imagery/preview/sundarbans_delta_s2",
        "local_asset_name": "sundarbans_delta_s2.png",
        "features": {
            "dominant_classes": ["mangrove_forest", "estuarine_channels", "wetlands"],
            "water_body_name": "Tidal Estuarine Channels",
            "has_large_water": True,
            "surrounding_type": "Dense protected mangrove forest and dynamic intertidal mudflats."
        }
    },
    {
        "scene_id": "S2A_MSIL2A_20260829T060621_N0500_R077_T42RWV_20260829T093210",
        "scene_name": "Bhadla Mega Solar Park & Thar Arid Zone",
        "collection": "sentinel-2-l2a",
        "acquisition_time": "2026-08-29T06:06:21.025Z",
        "cloud_cover": 0.0,
        "bbox": [71.85, 27.50, 71.95, 27.60],
        "crs": "EPSG:4326",
        "resolution_m": 10.0,
        "provider": "Copernicus Data Space Ecosystem (CDSE)",
        "thumbnail_url": "/api/imagery/preview/bhadla_solar_s2",
        "local_asset_name": "bhadla_solar_s2.png",
        "features": {
            "dominant_classes": ["photovoltaic_arrays", "sand_dunes", "arid_scrub"],
            "water_body_name": "None",
            "has_large_water": False,
            "surrounding_type": "Vast photovoltaic solar collector grids flanked by sandy desert terrain."
        }
    }
]

class CopernicusSTACClient:
    """
    Client for searching Sentinel-2 L2A optical scenes from CDSE STAC
    with fallback to curated cache for offline resiliency.
    """

    def __init__(self, timeout_secs: float = 6.0):
        self.timeout = timeout_secs
        self.cached_scenes = SAMPLE_SCENES

    async def search(
        self,
        bbox: Optional[List[float]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_cloud_cover: float = 20.0,
        limit: int = 6
    ) -> List[SceneMetadata]:
        """
        Query CDSE STAC or match against curated scene database.
        """
        # Attempt live STAC search if network is available
        live_results = await self._query_live_stac(bbox, start_date, end_date, max_cloud_cover, limit)
        if live_results:
            return live_results

        # Fallback to smart proximity/cloud matching in curated scenes
        return self._search_cached(bbox, max_cloud_cover, limit)

    async def _query_live_stac(
        self,
        bbox: Optional[List[float]],
        start_date: Optional[str],
        end_date: Optional[str],
        max_cloud_cover: float,
        limit: int
    ) -> Optional[List[SceneMetadata]]:
        if not bbox:
            return None

        # Build STAC search payload
        if not start_date or not end_date:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=60)
            datetime_range = f"{start_dt.strftime('%Y-%m-%d')}T00:00:00Z/{end_dt.strftime('%Y-%m-%d')}T23:59:59Z"
        else:
            datetime_range = f"{start_date}/{end_date}"

        payload = {
            "collections": ["SENTINEL-2"],
            "bbox": bbox,
            "datetime": datetime_range,
            "query": {
                "eo:cloud_cover": {"lte": max_cloud_cover}
            },
            "limit": limit
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(CDSE_STAC_URL, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    features = data.get("features", [])
                    results = []
                    for f in features:
                        props = f.get("properties", {})
                        meta = SceneMetadata(
                            scene_id=f.get("id", "S2_UNKNOWN"),
                            collection="sentinel-2-l2a",
                            acquisition_time=props.get("datetime", datetime.now(timezone.utc).isoformat()),
                            cloud_cover=float(props.get("eo:cloud_cover", 5.0)),
                            bbox=f.get("bbox", bbox),
                            crs="EPSG:4326",
                            resolution_m=10.0,
                            provider="Copernicus Data Space Ecosystem (CDSE)",
                            thumbnail_url=f.get("assets", {}).get("thumbnail", {}).get("href")
                        )
                        results.append(meta)
                    if results:
                        return results
        except Exception:
            # Live query timed out or CDSE network unreachable; silently fallback
            pass

        return None

    def _search_cached(
        self,
        bbox: Optional[List[float]],
        max_cloud_cover: float,
        limit: int
    ) -> List[SceneMetadata]:
        matched = []
        for s in self.cached_scenes:
            if s["cloud_cover"] <= max_cloud_cover:
                matched.append(
                    SceneMetadata(
                        scene_id=s["scene_id"],
                        scene_name=s["scene_name"],
                        collection=s["collection"],
                        acquisition_time=s["acquisition_time"],
                        cloud_cover=s["cloud_cover"],
                        bbox=s["bbox"],
                        crs=s["crs"],
                        resolution_m=s["resolution_m"],
                        provider=s["provider"],
                        thumbnail_url=s["thumbnail_url"],
                        assets={"features": s.get("features", {})}
                    )
                )

        if bbox and len(bbox) == 4:
            # Sort by distance from target center
            target_center_lon = (bbox[0] + bbox[2]) / 2.0
            target_center_lat = (bbox[1] + bbox[3]) / 2.0
            
            def distance(item: SceneMetadata) -> float:
                c_lon = (item.bbox[0] + item.bbox[2]) / 2.0
                c_lat = (item.bbox[1] + item.bbox[3]) / 2.0
                return ((c_lon - target_center_lon) ** 2 + (c_lat - target_center_lat) ** 2) ** 0.5
            
            matched.sort(key=distance)

        return matched[:limit]

    def get_scene_by_id(self, scene_id: str) -> Optional[Dict[str, Any]]:
        for s in self.cached_scenes:
            if s["scene_id"] == scene_id or scene_id in s["scene_id"] or s["local_asset_name"].startswith(scene_id):
                return s
        return None
