import os
import io
import base64
from PIL import Image
from typing import Dict, Any, Optional, Tuple
from backend.app.tools.base import BaseTool
from backend.app.geospatial.raster_utils import RasterProcessor
from backend.app.geospatial.stac_client import SAMPLE_SCENES

class ImageProcessorTool(BaseTool):
    tool_id = "image_processor_tool"

    def __init__(self, data_dir: str = "data/sample_scenes"):
        self.data_dir = data_dir

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        scene_id = inputs.get("scene_id")
        image_data_url = inputs.get("image_data_url")
        aoi_bbox = inputs.get("bbox")

        raw_bytes: Optional[bytes] = None
        scene_meta: Dict[str, Any] = {}

        if image_data_url:
            # Decode base64 uploaded image
            if "," in image_data_url:
                header, encoded = image_data_url.split(",", 1)
            else:
                encoded = image_data_url
            raw_bytes = base64.b64decode(encoded)
            scene_meta = {
                "source": "user_upload",
                "scene_id": "CUSTOM_UPLOAD_" + os.urandom(3).hex()
            }
        elif scene_id:
            # Find in sample scenes
            matched = None
            for s in SAMPLE_SCENES:
                if s["scene_id"] == scene_id or scene_id in s["scene_id"] or s["local_asset_name"].startswith(scene_id):
                    matched = s
                    break
            
            if not matched:
                matched = SAMPLE_SCENES[0] # Default to Lake Pichola
            
            file_path = os.path.join(self.data_dir, matched["local_asset_name"])
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    raw_bytes = f.read()
            else:
                # Generate in-memory fallback
                img = Image.new("RGB", (512, 512), color=(70, 90, 110))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                raw_bytes = buf.getvalue()
            
            scene_meta = matched

        if not raw_bytes:
            raise ValueError("No image source provided (neither scene_id nor image_data_url).")

        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        
        # Calculate spectral indices for VLM reasoning
        spectral_indices = RasterProcessor.compute_spectral_indices(pil_img)

        # Generate standard preview PNG bytes
        preview_bytes, stats = RasterProcessor.create_rgb_preview(raw_bytes, max_dim=800)

        return {
            "status": "ok",
            "image_bytes": preview_bytes,
            "dimensions": [pil_img.width, pil_img.height],
            "spectral_indices": spectral_indices,
            "scene_metadata": scene_meta,
            "channels": stats.get("color_bands", ["Red", "Green", "Blue"])
        }
