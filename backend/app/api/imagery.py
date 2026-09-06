import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from backend.app.schemas.scene import SceneMetadata, SearchSceneRequest, BoundingBox
from backend.app.geospatial.stac_client import CopernicusSTACClient, SAMPLE_SCENES
from backend.app.geospatial.realtime_imagery import RealtimeImageryService, CACHE_DIR

router = APIRouter(prefix="/api/imagery", tags=["Imagery"])
stac_client = CopernicusSTACClient()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(ROOT_DIR, "data", "sample_scenes")

class GeocodeRequest(BaseModel):
    query: str

class GeocodeResponse(BaseModel):
    name: str
    lat: float
    lon: float
    bbox: List[float]
    zoom: int
    image_url: str

from backend.app.geospatial.geocoder import GeocoderService
from backend.app.geospatial.imagery_service import ImageryService

@router.post("/geocode", response_model=GeocodeResponse)
async def geocode_location(req: GeocodeRequest):
    """
    Dynamically resolves place names from voice or text, generates real-time satellite imagery,
    and returns exact bounding coordinates for the map.
    """
    geo = GeocoderService.geocode(req.query)
    if not geo:
        raise HTTPException(status_code=404, detail=f"Location '{req.query}' could not be resolved.")
    scene = ImageryService.get_multimodal_scene(geo["name"], bbox=geo["bbox"], zoom=geo.get("zoom", 14))
    return GeocodeResponse(
        name=geo["name"],
        lat=geo["lat"],
        lon=geo["lon"],
        bbox=geo["bbox"],
        zoom=geo.get("zoom", 14),
        image_url=scene["image_url"]
    )

@router.post("/search", response_model=List[SceneMetadata])
async def search_satellite_scenes(req: SearchSceneRequest):
    """
    Search Sentinel-2 L2A optical scenes matching geographic bounding box,
    acquisition timeframe, and cloud cover thresholds from Copernicus STAC.
    """
    bbox_list = req.bbox.to_bbox_list() if req.bbox else None
    results = await stac_client.search(
        bbox=bbox_list,
        start_date=req.start_date,
        end_date=req.end_date,
        max_cloud_cover=req.max_cloud_cover,
        limit=req.limit
    )
    return results

@router.get("/scenes", response_model=List[SceneMetadata])
async def list_available_scenes():
    """
    List active satellite scenes.
    """
    results = await stac_client.search(limit=10)
    return results

@router.get("/cache/{filename}")
async def get_cached_realtime_image(filename: str):
    """
    Serve dynamically generated real-time satellite mosaic or bi-temporal comparison image.
    """
    import urllib.parse
    clean_filename = urllib.parse.unquote(filename)
    file_path = os.path.join(CACHE_DIR, clean_filename)
    
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"}
        )

    # Check raw filename
    raw_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(raw_path):
        return FileResponse(
            raw_path,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"}
        )

    # Fallback to any active cache file if present
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".png"):
            return FileResponse(os.path.join(CACHE_DIR, f), media_type="image/png")

    raise HTTPException(status_code=404, detail=f"Cached image '{filename}' not found.")

@router.get("/preview/{scene_name}")
async def get_scene_preview(scene_name: str):
    """
    Serve high-resolution satellite scene visual preview PNG.
    """
    clean_name = scene_name.replace(".png", "")
    
    # Check in cache first
    cached_path = os.path.join(CACHE_DIR, f"{clean_name}.png")
    if os.path.exists(cached_path):
        return FileResponse(cached_path, media_type="image/png")

    # Match in SAMPLE_SCENES or data/sample_scenes
    matched_file = None
    for s in SAMPLE_SCENES:
        if clean_name in s["local_asset_name"] or clean_name in s["scene_id"] or s["local_asset_name"].startswith(clean_name):
            matched_file = s["local_asset_name"]
            break

    if not matched_file:
        matched_file = f"{clean_name}.png"

    file_path = os.path.join(DATA_DIR, matched_file)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    
    default_path = os.path.join(DATA_DIR, "lake_pichola_s2.png")
    if os.path.exists(default_path):
        return FileResponse(default_path, media_type="image/png")

    raise HTTPException(status_code=404, detail=f"Scene preview '{scene_name}' not found.")
