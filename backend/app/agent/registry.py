from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ToolCapability(BaseModel):
    tool_id: str
    name: str
    description: str
    version: str = "0.1.0"
    task_types: List[str]
    modalities: List[str] = ["optical"]
    input_count: int = 1
    formats: List[str] = ["png", "jpeg", "geotiff"]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    outputs: List[str] = ["answer", "confidence", "evidence"]
    runtime: str = "local_gpu / fallback_cpu"
    is_active: bool = True

class ModelRegistry:
    """
    Central typed registry for COSMOCLIP specialist AI tools and models.
    Prevents hardcoding model invocation and enables dynamic agentic routing.
    """

    def __init__(self):
        self._tools: Dict[str, ToolCapability] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        # 1. Satellite Search Tool
        self.register(
            ToolCapability(
                tool_id="satellite_search_tool",
                name="Copernicus STAC Search Tool",
                description="Queries Copernicus Data Space STAC catalog for Sentinel-2 L2A optical scenes matching AOI and cloud parameters.",
                version="0.1.0",
                task_types=["search", "discovery"],
                modalities=["optical", "multispectral"],
                outputs=["scene_id", "scene_metadata", "thumbnail_url"]
            )
        )

        # 2. Raster Processor Tool
        self.register(
            ToolCapability(
                tool_id="image_processor_tool",
                name="Remote Sensing Raster Processor",
                description="Normalizes optical reflectance, crops bounding box patches, calculates spectral proxies, and creates preview assets.",
                version="0.1.0",
                task_types=["preprocessing", "spectral_analysis"],
                modalities=["optical", "multispectral", "geotiff"],
                outputs=["image_bytes", "spectral_indices", "preview_meta"]
            )
        )

        # 3. Satellite VQA Specialist Tool (RS-LLaVA)
        self.register(
            ToolCapability(
                tool_id="satellite_vqa",
                name="RS-LLaVA Satellite VQA Specialist",
                description="Remote Sensing Vision-Language Model adapted for natural language question answering and scene reasoning over earth observation imagery.",
                version="0.1.0",
                task_types=["vqa", "captioning"],
                modalities=["optical"],
                parameters={"max_new_tokens": 128, "temperature": 0.0, "adapter": "RS-VQA-LoRA"},
                outputs=["answer", "confidence", "evidence"],
                runtime="local_gpu"
            )
        )

        # 4. Grounding Specialist Tool (P1 / Phase 2)
        self.register(
            ToolCapability(
                tool_id="satellite_grounding_tool",
                name="Grounding DINO Remote Sensing Specialist",
                description="Language-guided spatial bounding box and polygon localization for earth observation objects.",
                version="0.1.0",
                task_types=["grounding"],
                modalities=["optical"],
                outputs=["bounding_boxes", "confidence"],
                is_active=True
            )
        )

    def register(self, tool: ToolCapability):
        self._tools[tool.tool_id] = tool

    def get_tool(self, tool_id: str) -> Optional[ToolCapability]:
        return self._tools.get(tool_id)

    def find_tools_for_task(self, task_type: str, modality: str = "optical") -> List[ToolCapability]:
        return [
            tool for tool in self._tools.values()
            if tool.is_active and task_type in tool.task_types and modality in tool.modalities
        ]

    def list_all_tools(self) -> List[ToolCapability]:
        return list(self._tools.values())

GLOBAL_REGISTRY = ModelRegistry()
