"""COSMOCLIP Remote-Sensing VQA Engine Types & Data Contracts (PRD V1.2)."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class FetchResult:
    """Result returned by fetch_satellite_image() interface (PRD §10.3)."""
    image_path: str
    latitude: float = 0.0
    longitude: float = 0.0
    scene_date: str = ""
    source: str = "sentinel-2"
    cloud_cover_pct: float = 0.0
    status: str = "success"  # "success" | "geocode_failed" | "no_scene_found" | "error"
    location_query: str = ""
    latency_ms: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "latitude": round(self.latitude, 5),
            "longitude": round(self.longitude, 5),
            "scene_date": self.scene_date,
            "source": self.source,
            "cloud_cover_pct": round(self.cloud_cover_pct, 1),
            "status": self.status,
            "location_query": self.location_query,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
        }


@dataclass
class LatencyBreakdown:
    """Decomposed latency timings in milliseconds (PRD §6.1 / N2)."""
    geocode_ms: float = 0.0
    catalog_search_ms: float = 0.0
    image_fetch_ms: float = 0.0
    image_preprocess_ms: float = 0.0
    vqa_inference_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        res = {
            "image_preprocess_ms": round(self.image_preprocess_ms, 2),
            "vqa_inference_ms": round(self.vqa_inference_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }
        if self.geocode_ms > 0:
            res["geocode_ms"] = round(self.geocode_ms, 2)
        if self.catalog_search_ms > 0:
            res["catalog_search_ms"] = round(self.catalog_search_ms, 2)
        if self.image_fetch_ms > 0:
            res["image_fetch_ms"] = round(self.image_fetch_ms, 2)
        return res


@dataclass
class VQAResult:
    """Structured result returned by run_vqa() interface (PRD §10.2)."""
    answer: str
    model: str = "TerraQ-VL"
    model_version: str = "4bit-qwen2.5-3b-v1"
    latency_ms: float = 0.0
    status: str = "success"  # "success" | "timeout" | "low_confidence" | "error"
    output_tokens: int = 0
    image_source: str = "upload"  # "upload" | "fetch"
    location_query: Optional[str] = None
    resolved_lat: Optional[float] = None
    resolved_lon: Optional[float] = None
    scene_date: Optional[str] = None
    cloud_cover_pct: Optional[float] = None
    latency_breakdown: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    raw_details: Optional[Dict[str, Any]] = None

    def to_trace_dict(self, session_id: str = "", image_id: str = "", question: str = "") -> Dict[str, Any]:
        """Formats the result into an execution trace record (PRD F6/F9)."""
        import time
        trace = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "image_id": image_id,
            "image_source": self.image_source,
            "question": question,
            "intent": "satellite_image_analysis",
            "tool": "analyze_satellite_image",
            "model": self.model,
            "model_version": self.model_version,
            "output_tokens": self.output_tokens,
            "vqa_inference_ms": round(self.latency_breakdown.vqa_inference_ms, 2),
            "total_latency_ms": round(self.latency_ms, 2),
            "status": self.status,
            "latency_breakdown": self.latency_breakdown.to_dict(),
        }
        if self.location_query:
            trace["location_query"] = self.location_query
        if self.resolved_lat is not None:
            trace["resolved_lat"] = round(self.resolved_lat, 5)
        if self.resolved_lon is not None:
            trace["resolved_lon"] = round(self.resolved_lon, 5)
        if self.scene_date:
            trace["scene_date"] = self.scene_date
        if self.cloud_cover_pct is not None:
            trace["cloud_cover_pct"] = round(self.cloud_cover_pct, 1)
        if self.latency_breakdown.geocode_ms > 0:
            trace["geocode_ms"] = round(self.latency_breakdown.geocode_ms, 2)
        if self.latency_breakdown.catalog_search_ms > 0:
            trace["catalog_search_ms"] = round(self.latency_breakdown.catalog_search_ms, 2)
        if self.latency_breakdown.image_fetch_ms > 0:
            trace["image_fetch_ms"] = round(self.latency_breakdown.image_fetch_ms, 2)
        return trace
