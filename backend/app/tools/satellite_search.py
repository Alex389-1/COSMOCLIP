from typing import Dict, Any, List, Optional
from backend.app.tools.base import BaseTool
from backend.app.geospatial.stac_client import CopernicusSTACClient

class SatelliteSearchTool(BaseTool):
    tool_id = "satellite_search_tool"

    def __init__(self):
        self.stac_client = CopernicusSTACClient()

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        bbox = inputs.get("bbox")
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")
        max_cloud_cover = float(inputs.get("max_cloud_cover", 20.0))
        limit = int(inputs.get("limit", 6))

        scenes = await self.stac_client.search(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            limit=limit
        )

        return {
            "status": "ok",
            "count": len(scenes),
            "scenes": [s.model_dump() for s in scenes]
        }
