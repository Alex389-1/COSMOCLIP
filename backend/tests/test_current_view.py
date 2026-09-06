import io
import base64
import pytest
import numpy as np
from PIL import Image
from httpx import ASGITransport, AsyncClient
from backend.app.main import app
from backend.app.agent.router import route_query, classify


def generate_dummy_screenshot_b64(r: int = 34, g: int = 139, b: int = 34) -> str:
    """Generates a small test JPEG base64 data URL (e.g. forest green)."""
    arr = np.full((128, 128, 3), [r, g, b], dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def test_two_way_router_classification():
    """
    Verifies the two-way split:
    - Named destination present -> navigation
    - Everything else -> current_view
    """
    assert classify("take me to KIET Ghaziabad") == "navigation"
    assert classify("show me Tokyo") == "navigation"
    assert classify("where is Paris") == "navigation"
    assert classify("London") == "navigation"

    assert classify("what can you see") == "current_view"
    assert classify("can you see the cars on street") == "current_view"
    assert classify("how has this changed") == "current_view"
    assert classify("count the buildings here") == "current_view"
    assert classify("what am I looking at") == "current_view"

    route_nav = route_query("show me Tokyo")
    assert route_nav["query_type"] == "navigation"
    assert route_nav.get("target_entity") is not None

    route_cv = route_query("can you see the cars")
    assert route_cv["query_type"] == "current_view"


@pytest.mark.asyncio
async def test_current_view_endpoint_never_moves_map():
    """
    Verifies /api/query/current-view endpoint:
    - Never returns recenter fields or map coordinates
    - Always sets should_recenter_map: False and is_new_location_query: False
    - Returns query_type: 'current_view'
    """
    screenshot_b64 = generate_dummy_screenshot_b64(34, 139, 34)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "question": "What land cover is visible in this satellite view?",
            "image_base64": screenshot_b64
        }
        resp = await ac.post("/api/query/current-view", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert "answer" in data
        assert len(data["answer"]) > 0
        assert data.get("query_type") == "current_view"
        assert data.get("should_recenter_map") is False
        assert data.get("is_new_location_query") is False
        assert data.get("location_meta") is None


@pytest.mark.asyncio
async def test_main_query_endpoint_routes_screenshot_to_current_view():
    """
    Verifies that POST /api/query with image_base64 routes to current-view
    when non-navigation phrasing is used, leaving the map completely untouched.
    """
    screenshot_b64 = generate_dummy_screenshot_b64(20, 100, 200) # blue water
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "question": "Can you see any boats or water in this view?",
            "image_base64": screenshot_b64
        }
        resp = await ac.post("/api/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data.get("query_type") == "current_view"
        assert data.get("should_recenter_map") is False
        assert data.get("is_new_location_query") is False
        assert data.get("location_meta") is None
        assert "answer" in data
