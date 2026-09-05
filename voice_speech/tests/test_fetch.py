"""Unit tests for Sentinel-2 satellite image acquisition engine (PRD §8.1 / §10.3)."""

import pytest
import os
from pathlib import Path
from voice_speech.engine.vqa.fetch import fetch_satellite_image, calculate_bounding_box
from voice_speech.engine.vqa.types import FetchResult


def test_calculate_bounding_box():
    """Test 5km x 5km bounding box calculation."""
    min_lon, min_lat, max_lon, max_lat = calculate_bounding_box(18.947, 72.845, size_km=5.0)
    assert min_lat < 18.947 < max_lat
    assert min_lon < 72.845 < max_lon
    assert round(max_lat - min_lat, 4) > 0.03


@pytest.mark.asyncio
async def test_fetch_satellite_image_success():
    """Test successful acquisition of Sentinel-2 image for known location."""
    res = await fetch_satellite_image(
        location="Mumbai Port",
        session_id="test_fetch_sess_1",
    )
    assert isinstance(res, FetchResult)
    assert res.status == "success"
    assert res.image_path != ""
    assert res.source != ""
    assert res.cloud_cover_pct < 20.0
    assert "geocode_ms" in res.latency_ms
    assert "image_fetch_ms" in res.latency_ms


@pytest.mark.asyncio
async def test_fetch_satellite_image_unknown_location():
    """Test error handling when location cannot be geocoded (PRD F8 / §10.6)."""
    res = await fetch_satellite_image(
        location="xyz999_invalid_alien_base",
        session_id="test_fetch_sess_err",
    )
    assert isinstance(res, FetchResult)
    assert res.status == "geocode_failed"
    assert res.error_message == "I couldn't find that location."


@pytest.mark.asyncio
async def test_fetch_satellite_image_with_date():
    """Test acquisition with specific requested acquisition date via ArcGIS Wayback."""
    target_date = "2021-03-17"
    res = await fetch_satellite_image(
        location="Thar Desert",
        date=target_date,
        session_id="test_fetch_sess_date",
    )
    assert res.status == "success"
    assert "wayback" in res.source.lower()
    assert res.scene_date != ""
    assert res.latitude > 20.0
