import pytest
from httpx import ASGITransport, AsyncClient
from backend.app.main import app

@pytest.mark.asyncio
async def test_api_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "online"
        assert data["capabilities"]["langgraph_controller"] is True

@pytest.mark.asyncio
async def test_api_imagery_search():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/imagery/search", json={"max_cloud_cover": 15.0, "limit": 2})
        assert resp.status_code == 200
        scenes = resp.json()
        assert len(scenes) > 0
        assert "scene_id" in scenes[0]

@pytest.mark.asyncio
async def test_api_query():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "question": "Is there a large water body in this area?",
            "scene_id": "S2A_MSIL2A_20260814T054641_N0500_R120_T43RER_20260814T084512"
        }
        resp = await ac.post("/api/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "confidence" in data
        assert len(data["trace"]) > 0
