"""Unit tests for RS-VLM VQA engine, error states, and uncertainty filtering."""

import pytest
import asyncio
from pathlib import Path
from voice_speech.engine.vqa.model import RSVQAModelEngine, run_vqa
from voice_speech.engine.vqa.types import VQAResult
from voice_speech.tests.generate_test_tiles import create_sample_tiles


@pytest.fixture(scope="module")
def sample_tiles():
    return create_sample_tiles("voice_speech/data/test_samples")


@pytest.mark.asyncio
async def test_vqa_water_presence(sample_tiles):
    """Test Presence Question: water detection in coastal tile."""
    water_tile = sample_tiles["water"]
    result = await run_vqa(
        image_path=water_tile,
        question="Is there water in this image?",
        session_id="test_session_1",
    )
    assert isinstance(result, VQAResult)
    assert result.status == "success"
    assert "yes" in result.answer.lower()
    assert result.latency_ms > 0
    assert result.latency_breakdown.vqa_inference_ms >= 0


@pytest.mark.asyncio
async def test_vqa_urban_structures(sample_tiles):
    """Test Object/Spatial Question: buildings in urban tile."""
    urban_tile = sample_tiles["urban"]
    result = await run_vqa(
        image_path=urban_tile,
        question="Are there buildings visible?",
        session_id="test_session_2",
    )
    assert result.status == "success"
    assert "building" in result.answer.lower() or "structure" in result.answer.lower()


@pytest.mark.asyncio
async def test_vqa_vegetation_land_cover(sample_tiles):
    """Test Land Cover Question: forest / agriculture tile."""
    veg_tile = sample_tiles["vegetation"]
    result = await run_vqa(
        image_path=veg_tile,
        question="What type of land dominates this area?",
        session_id="test_session_3",
    )
    assert result.status == "success"
    assert "vegetation" in result.answer.lower() or "forest" in result.answer.lower() or "land" in result.answer.lower()


@pytest.mark.asyncio
async def test_vqa_missing_image_error():
    """Test error handling when no image is provided (Requirement §10)."""
    result = await run_vqa(
        image_path="",
        question="What is this?",
        session_id="test_session_err",
    )
    assert result.status == "error"
    assert "No image uploaded yet" in result.answer or "Please upload" in result.answer


@pytest.mark.asyncio
async def test_vqa_invalid_format_error(tmp_path):
    """Test error handling for unsupported file extensions (PDF, GeoTIFF)."""
    fake_geotiff = tmp_path / "satellite.tif"
    fake_geotiff.write_bytes(b"GEOTIFF_DUMMY_HEADER")

    result = await run_vqa(
        image_path=str(fake_geotiff),
        question="Is there a river?",
    )
    assert result.status == "error"
    assert "Unsupported image format" in result.answer


@pytest.mark.asyncio
async def test_vqa_uncertainty_heuristic():
    """Test F7 Bounded Uncertainty Heuristic."""
    engine = RSVQAModelEngine.get_instance()
    ans, status = engine._apply_uncertainty_heuristic("I cannot determine if there is an airfield.")
    assert status == "low_confidence"
    assert ans == "I can't reliably determine that from this image."
