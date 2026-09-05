import io
import os
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Dict, Any, Tuple, Optional, List

class RasterProcessor:
    """
    Handles remote-sensing raster processing, normalization,
    clipping, preview generation, and spatial coordinate mapping.
    """

    @staticmethod
    def normalize_reflectance(band_data: np.ndarray, lower_percentile: float = 2.0, upper_percentile: float = 98.0) -> np.ndarray:
        """
        Robust 2%-98% cumulative stretch for satellite optical reflectance bands.
        Maps raw values (e.g. 0-10000 DN or 0.0-1.0 surface reflectance) to 0-255 uint8.
        """
        valid_mask = ~np.isnan(band_data)
        if not np.any(valid_mask):
            return np.zeros_like(band_data, dtype=np.uint8)
        
        vmin = np.percentile(band_data[valid_mask], lower_percentile)
        vmax = np.percentile(band_data[valid_mask], upper_percentile)
        
        if vmax <= vmin:
            vmax = vmin + 1e-5
            
        stretched = np.clip((band_data - vmin) / (vmax - vmin), 0.0, 1.0)
        return (stretched * 255).astype(np.uint8)

    @staticmethod
    def create_rgb_preview(image_bytes: bytes, max_dim: int = 1024) -> Tuple[bytes, Dict[str, Any]]:
        """
        Loads an optical raster image, scales to max_dim while preserving aspect ratio,
        and returns PNG bytes along with dimensions and channel stats.
        """
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            orig_w, orig_h = pil_img.size
            
            # Scale if larger than max_dim
            if max(orig_w, orig_h) > max_dim:
                scale = max_dim / float(max(orig_w, orig_h))
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                new_w, new_h = orig_w, orig_h

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=True)
            png_bytes = buf.getvalue()

            stats = {
                "original_dimensions": [orig_w, orig_h],
                "preview_dimensions": [new_w, new_h],
                "format": "PNG",
                "color_bands": ["Red (B04)", "Green (B03)", "Blue (B02)"]
            }
            return png_bytes, stats
        except Exception as e:
            raise ValueError(f"Failed to process raster image: {str(e)}")

    @staticmethod
    def crop_aoi_patch(image: Image.Image, aoi_normalized: List[float]) -> Image.Image:
        """
        Crop a normalized [ymin, xmin, ymax, xmax] (0.0 to 1.0) box from PIL Image.
        """
        w, h = image.size
        ymin, xmin, ymax, xmax = aoi_normalized
        crop_box = (
            int(xmin * w),
            int(ymin * h),
            int(xmax * w),
            int(ymax * h)
        )
        # Ensure minimum 16x16 size
        crop_box = (
            max(0, crop_box[0]),
            max(0, crop_box[1]),
            min(w, max(crop_box[0] + 16, crop_box[2])),
            min(h, max(crop_box[1] + 16, crop_box[3]))
        )
        return image.crop(crop_box)

    @staticmethod
    def compute_spectral_indices(rgb_img: Image.Image) -> Dict[str, float]:
        """
        Extract fast proxy indices from visible RGB channels (e.g., Green-Red index for vegetation,
        Blue-Red ratio for water bodies, brightness for urban/cloud).
        """
        arr = np.array(rgb_img.convert("RGB"), dtype=np.float32)
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        total_pixels = r.size

        # Water proxy: high Blue & Green, low Red
        water_mask = (b > r * 1.15) & (g > r * 1.05) & (b > 50) & (r < 110)
        water_ratio = float(np.sum(water_mask) / total_pixels)

        # Vegetation proxy: high Green relative to Red & Blue (VARI: (G - R)/(G + R - B))
        denom = g + r - b
        denom[denom == 0] = 1e-5
        vari = (g - r) / denom
        veg_mask = (g > r * 1.1) & (g > b * 1.05) & (r < 140)
        veg_ratio = float(np.sum(veg_mask) / total_pixels)

        # Urban/Built-up proxy: high brightness, low spectral variance, moderate reflectance
        urban_mask = (r > 120) & (g > 120) & (b > 120) & (np.abs(r - g) < 25) & (np.abs(g - b) < 25)
        urban_ratio = float(np.sum(urban_mask) / total_pixels)

        # Cloud proxy: extremely high brightness in all channels
        cloud_mask = (r > 220) & (g > 220) & (b > 220)
        cloud_ratio = float(np.sum(cloud_mask) / total_pixels)

        return {
            "water_ratio": round(water_ratio, 4),
            "vegetation_ratio": round(veg_ratio, 4),
            "urban_ratio": round(urban_ratio, 4),
            "cloud_ratio": round(cloud_ratio, 4),
            "mean_reflectance": float(round(np.mean(arr) / 255.0, 3))
        }
