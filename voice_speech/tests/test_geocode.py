"""Unit tests for Nominatim geocoding engine with caching (PRD §8.1)."""

import pytest
from voice_speech.engine.vqa.geocode import geocode_location



@pytest.mark.asyncio
async def test_geocode_known_location():
    """Test resolving canonical remote-sensing test locations."""
    res = await geocode_location("Mumbai Port")
    assert res is not None
    lat, lon, name = res
    assert 18.0 <= lat <= 20.0
    assert 72.0 <= lon <= 74.0
    assert "Mumbai" in name


@pytest.mark.asyncio
async def test_geocode_thar_desert():
    """Test geocoding Thar Desert."""
    res = await geocode_location("Thar Desert")
    assert res is not None
    lat, lon, name = res
    assert 25.0 <= lat <= 29.0
    assert 68.0 <= lon <= 74.0



@pytest.mark.asyncio
async def test_geocode_cache_hit():
    """Test LRU caching avoids duplicate external calls."""
    res1 = await geocode_location("New Delhi")
    assert res1 is not None

    res2 = await geocode_location("New Delhi")
    assert res2 == res1


@pytest.mark.asyncio
async def test_geocode_unknown_location():
    """Test handling of nonexistent/unresolvable location."""
    res = await geocode_location("xyz123_nonexistent_place_99999")
    assert res is None


@pytest.mark.asyncio
async def test_geocode_empty_query():
    """Test handling of empty string."""
    res = await geocode_location("   ")
    assert res is None
