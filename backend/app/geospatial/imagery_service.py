import os
import hashlib
from typing import Dict, Any, List, Optional
from backend.app.geospatial.geocoder import GeocoderService
from backend.app.geospatial.sentinel2_service import Sentinel2Service
from backend.app.geospatial.wayback_service import WaybackService
from backend.app.geospatial.sar_service import SARService
from backend.app.geospatial.fusion_service import FusionService
from backend.app.geospatial.change_detection_service import ChangeDetectionService

class ImageryService:
    """
    Unified Remote Sensing Imagery & Data Router Facade.
    Orchestrates Sentinel-2 Optical, Sentinel-1 SAR, Esri Wayback Historical Archive, and Multimodal Fusion.
    """

    @staticmethod
    def geocode(location_name: str) -> Dict[str, Any]:
        return GeocoderService.geocode(location_name)

    @staticmethod
    def get_multimodal_scene(
        location_name: str,
        bbox: Optional[List[float]] = None,
        zoom: int = 14,
        baseline_year: str = "2020",
        is_fine_detail: bool = False
    ) -> Dict[str, Any]:
        # 1. Geocode location if bbox not provided
        if not bbox or len(bbox) != 4:
            geo = GeocoderService.geocode(location_name)
            if not geo:
                bbox = [77.48, 28.74, 77.51, 28.76]
                center_lat, center_lon = 28.7495, 77.4912
                resolved_name = location_name or "Observation Area"
            else:
                bbox = geo["bbox"]
                center_lat, center_lon = geo["lat"], geo["lon"]
                resolved_name = geo["name"]
        else:
            center_lat = (bbox[1] + bbox[3]) / 2.0
            center_lon = (bbox[0] + bbox[2]) / 2.0
            resolved_name = location_name

        loc_slug = hashlib.md5(f"{center_lat:.4f}_{center_lon:.4f}_{zoom}_{is_fine_detail}".encode()).hexdigest()[:10]

        # 2. Acquire Optical Imagery (Sub-Meter High-Res for fine detail queries vs Sentinel-2 MSI for macro)
        if is_fine_detail:
            s2_meta = Sentinel2Service.get_esri_highres_crop(bbox=bbox, zoom=max(zoom, 18))
        else:
            s2_meta = Sentinel2Service.get_sentinel2(bbox=bbox, zoom=zoom)
        opt_path = s2_meta["image_path"]

        # 3. Acquire Genuine Historical Baseline Raster (Esri Wayback WMTS)
        wayback_meta = WaybackService.get_historical_baseline(
            bbox=bbox,
            zoom=zoom,
            target_year=baseline_year
        )
        baseline_path = wayback_meta["image_path"]

        # 4. Acquire Sentinel-1 C-Band SAR GRD
        s1_meta = SARService.get_sentinel1(
            bbox=bbox,
            resolution=10,
            optical_reference_path=opt_path
        )

        sar_vv_path = s1_meta.get("vv_image_path") or os.path.join(os.path.dirname(opt_path), f"s1_vv_{loc_slug}.png")
        sar_vh_path = s1_meta.get("vh_image_path") or os.path.join(os.path.dirname(opt_path), f"s1_vh_{loc_slug}.png")

        # 5. Generate Cross-Modal Fusion and Difference Layers
        fused_url = FusionService.generate_optical_sar_fusion(opt_path, sar_vv_path, sar_vh_path, loc_slug)
        sar_diff_url = FusionService.generate_sar_temporal_diff(sar_vv_path, loc_slug)
        before_url, after_url, diff_overlay_url = FusionService.generate_bitemporal_optical_pair(
            optical_path=opt_path,
            location_slug=loc_slug,
            baseline_path=baseline_path
        )

        # 6. Compute Deterministic Pixel-Level Change Vector Analysis (CVA) & SAR Log-Ratio Heatmaps
        cva_result = ChangeDetectionService.compute_optical_cva(
            baseline_path=baseline_path,
            current_path=opt_path,
            bbox=bbox,
            loc_slug=loc_slug,
            baseline_year=baseline_year,
            current_year="2026",
            zoom=zoom
        )

        sar_logratio_result = ChangeDetectionService.compute_sar_logratio(
            current_vv_path=sar_vv_path,
            baseline_vv_path=None,
            bbox=bbox,
            loc_slug=loc_slug,
            baseline_year=baseline_year,
            current_year="2026",
            zoom=zoom
        )

        return {
            "name": resolved_name,
            "lat": center_lat,
            "lon": center_lon,
            "zoom": zoom,
            "bbox": bbox,
            "optical": s2_meta,
            "sar": s1_meta,
            "wayback": wayback_meta,
            "baseline_year": baseline_year,
            "baseline_period": wayback_meta.get("release_title", f"Wayback ({baseline_year})"),
            "baseline_tile_url": wayback_meta.get("tile_template"),
            "image_url": s2_meta["image_url"],
            "optical_url": s2_meta["image_url"],
            "sar_url": s1_meta["sar_dual_pol_url"],
            "sar_vv_url": s1_meta["vv_image_url"],
            "sar_vh_url": s1_meta["vh_image_url"],
            "fused_url": fused_url or s2_meta["image_url"],
            "sar_diff_url": sar_diff_url,
            "before_image_url": before_url,
            "after_image_url": after_url,
            "diff_overlay_url": diff_overlay_url,
            "heatmap_url": cva_result.get("heatmap_url"),
            "sar_heatmap_url": sar_logratio_result.get("heatmap_url"),
            "heatmap_bounds": cva_result.get("bounds"),
            "cva_metrics": cva_result.get("metrics"),
            "sar_logratio_metrics": sar_logratio_result.get("metrics")
        }
