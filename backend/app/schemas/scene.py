from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    min_lon: float = Field(..., description="Minimum longitude (West)")
    min_lat: float = Field(..., description="Minimum latitude (South)")
    max_lon: float = Field(..., description="Maximum longitude (East)")
    max_lat: float = Field(..., description="Maximum latitude (North)")

    def to_bbox_list(self) -> List[float]:
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]

class SceneMetadata(BaseModel):
    scene_id: str = Field(..., description="Unique scene identifier")
    collection: str = Field(default="sentinel-2-l2a", description="STAC collection name")
    acquisition_time: str = Field(..., description="ISO 8601 acquisition timestamp")
    cloud_cover: float = Field(..., description="Estimated cloud cover percentage (0-100)")
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    crs: str = Field(default="EPSG:4326", description="Coordinate Reference System")
    resolution_m: float = Field(default=10.0, description="Spatial resolution in meters")
    provider: str = Field(default="Copernicus Data Space Ecosystem (CDSE)", description="Data provider")
    thumbnail_url: Optional[str] = Field(None, description="Visual preview URL or local path")
    assets: Dict[str, Any] = Field(default_factory=dict, description="Dictionary of available spectral assets")
    scene_name: Optional[str] = Field(None, description="Descriptive or location label")

class SearchSceneRequest(BaseModel):
    bbox: Optional[BoundingBox] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    max_cloud_cover: float = Field(default=20.0, ge=0.0, le=100.0)
    limit: int = Field(default=6, ge=1, le=20)
    location_query: Optional[str] = None
