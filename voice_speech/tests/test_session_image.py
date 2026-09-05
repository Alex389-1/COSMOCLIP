"""Tests for satellite image upload API and session lifecycle management."""

import pytest
import io
from fastapi.testclient import TestClient
from voice_speech.web_server import app
from voice_speech.engine.conversation.state import ACTIVE_SESSIONS, get_or_create_session_state
from voice_speech.tests.generate_test_tiles import create_sample_tiles


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_tiles():
    return create_sample_tiles("voice_speech/data/test_samples")


def test_upload_valid_png(client, sample_tiles):
    """Test successful upload of valid PNG satellite image."""
    session_id = "test_upload_sess_1"
    water_path = sample_tiles["water"]

    with open(water_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        f"/session/{session_id}/image",
        files={"file": ("sentinel2_water.png", io.BytesIO(file_bytes), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["session_id"] == session_id
    assert "image_url" in data

    # Verify session state binding
    state = get_or_create_session_state(session_id)
    assert state.active_image_path is not None
    assert state.active_image_id == "sentinel2_water.png"


def test_upload_invalid_format_rejected(client):
    """Test rejection of unsupported file types (e.g. GeoTIFF / PDF)."""
    session_id = "test_upload_sess_reject"
    fake_geotiff = io.BytesIO(b"FAKE_GEOTIFF_DATA")

    response = client.post(
        f"/session/{session_id}/image",
        files={"file": ("satellite_scene.tif", fake_geotiff, "image/tiff")},
    )
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]


def test_get_uploaded_image(client, sample_tiles):
    """Test retrieving active satellite image for a session."""
    session_id = "test_upload_sess_get"
    urban_path = sample_tiles["urban"]

    with open(urban_path, "rb") as f:
        file_bytes = f.read()

    client.post(
        f"/session/{session_id}/image",
        files={"file": ("sentinel2_urban.png", io.BytesIO(file_bytes), "image/png")},
    )

    get_resp = client.get(f"/session/{session_id}/image")
    assert get_resp.status_code == 200
    assert len(get_resp.content) == len(file_bytes)
