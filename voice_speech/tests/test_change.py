"""Unit tests for Bi-Temporal Satellite Change Detection (Change VQA)."""

import os
import pytest
from PIL import Image, ImageDraw
from voice_speech.engine.vqa.change import detect_satellite_changes, ChangeDetectionResult
from voice_speech.engine.gemini.tools import dispatch_tool_call


@pytest.fixture
def sample_temporal_tiles(tmp_path):
    """Creates synthetic T1 and T2 temporal image pair with visible changes."""
    t1_path = tmp_path / "t1.png"
    t2_path = tmp_path / "t2.png"

    # T1: Base scene with water and 1 vessel
    img1 = Image.new("RGB", (256, 256), color=(20, 60, 130))
    d1 = ImageDraw.Draw(img1)
    d1.rectangle([20, 20, 80, 50], fill=(240, 240, 240))  # 1 ship
    img1.save(t1_path)

    # T2: Changed scene with new ship and expanded coastline
    img2 = Image.new("RGB", (256, 256), color=(20, 60, 130))
    d2 = ImageDraw.Draw(img2)
    d2.rectangle([20, 20, 80, 50], fill=(240, 240, 240))    # 1 ship
    d2.rectangle([140, 120, 200, 150], fill=(255, 255, 255)) # New ship 2
    img2.save(t2_path)

    return str(t1_path), str(t2_path)


@pytest.mark.asyncio
async def test_detect_satellite_changes(sample_temporal_tiles):
    t1_path, t2_path = sample_temporal_tiles

    res = await detect_satellite_changes(
        t1_path=t1_path,
        t2_path=t2_path,
        location="Mumbai Port",
        date1="2026-08-31",
        date2="2026-09-02",
        question="Did maritime ship traffic change?",
        session_id="test_change_sess",
    )

    assert isinstance(res, ChangeDetectionResult)
    assert res.change_pct > 0.0
    assert os.path.exists(res.diff_overlay_path)
    assert len(res.changed_bbox) == 4
    assert "Mumbai Port" in res.summary
    assert "2026-08-31" in res.summary
    assert "2026-09-02" in res.summary


@pytest.mark.asyncio
async def test_dispatch_compare_satellite_images_tool():
    """Test invoking compare_satellite_images tool via tool registry."""
    res = await dispatch_tool_call(
        "compare_satellite_images",
        {
            "location": "Mumbai Port",
            "date1": "2026-08-31",
            "date2": "2026-09-02",
            "question": "Did ship traffic or water level change?",
        },
        context={"session_id": "test_tool_change_sess"},
    )

    assert isinstance(res, str)
    assert len(res) > 20
    assert "Mumbai Port" in res
    assert ("2026-08-31" in res or "August" in res or "31" in res)
