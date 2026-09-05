from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter(tags=["System Health"])

@router.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "service": "COSMOCLIP Remote Sensing Assistant",
        "version": "v0.1.0-mvp",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "capabilities": {
            "satellite_stac_search": True,
            "raster_preprocessing": True,
            "rs_llava_vqa": True,
            "langgraph_controller": True,
            "voice_interaction": True
        }
    }
