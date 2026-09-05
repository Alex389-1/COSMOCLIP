import pytest
import numpy as np
from PIL import Image
from backend.app.geospatial.raster_utils import RasterProcessor
from backend.app.geospatial.stac_client import CopernicusSTACClient

@pytest.mark.asyncio
async def test_stac_search_fallback():
    client = CopernicusSTACClient()
    results = await client.search(max_cloud_cover=10.0, limit=3)
    assert len(results) > 0
    assert results[0].collection == "sentinel-2-l2a"
    assert results[0].cloud_cover <= 10.0

def test_raster_spectral_indices():
    # Test water-like image (high Blue, low Red)
    water_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    water_arr[:, :, 0] = 30  # Red
    water_arr[:, :, 1] = 90  # Green
    water_arr[:, :, 2] = 140 # Blue
    img = Image.fromarray(water_arr)
    
    indices = RasterProcessor.compute_spectral_indices(img)
    assert indices["water_ratio"] > 0.8
    assert indices["vegetation_ratio"] < 0.2

def test_raster_preview_generation():
    arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()
    
    preview_bytes, stats = RasterProcessor.create_rgb_preview(raw_bytes, max_dim=150)
    assert len(preview_bytes) > 0
    assert stats["preview_dimensions"][0] <= 150
