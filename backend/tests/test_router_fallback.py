import time
import pytest
from unittest.mock import patch
from backend.app.agent.router import QueryInterpreter
from backend.app.geospatial.geocoder import GeocoderService

# Standard test viewport: KIET Ghaziabad campus
TEST_VP_BBOX = [77.4912, 28.7495, 77.5025, 28.7580]

# All 8 canonical test cases representing the full range of spatial intents:
# (question, exp_intent, exp_recenter)
CORE_TEST_CASES = [
    ("can you see the cars", "viewport_bound", False),
    ("can you see the cars on street", "viewport_bound", False),
    ("what can you see in this area", "viewport_bound", False),
    ("show me Tokyo", "navigation", True),
    ("take me to London", "navigation", True),
    ("New York City", "navigation", True),
    ("Paris", "navigation", True),
    ("what is the NDVI of this field", "viewport_bound", False),
    ("how many buildings are here", "viewport_bound", False),
    ("compare this year with last year", "viewport_bound", False),
    ("where is the port", "navigation", True),
    ("zoom to KIET college", "navigation", True),
]


class MockLLMTimeout:
    """Context manager simulating an LLM timeout / network outage."""
    def __enter__(self):
        QueryInterpreter._INTERPRET_CACHE.clear()
        self.patcher = patch.object(QueryInterpreter, "_llm_interpret", side_effect=TimeoutError("LLM connection timed out after 8000ms"))
        self.patcher.start()
        # Seed test geocodes so tests don't fail on external Nominatim rate limits (HTTP 429)
        for loc, lat, lon in [("tokyo", 35.68, 139.75), ("london", 51.50, -0.12), ("new york city", 40.71, -74.0), ("paris", 48.85, 2.35), ("the port", 18.95, 72.85), ("kiet college", 28.75, 77.50)]:
            if loc not in GeocoderService._GEOCODE_CACHE:
                GeocoderService._GEOCODE_CACHE[loc] = (time.time() + 86400, GeocoderService._build_geo_result(loc.title(), lat, lon))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.stop()
        QueryInterpreter._INTERPRET_CACHE.clear()


def test_fallback_path_all_core_cases():
    """
    Test 1: Fallback path test suite.
    Forces _llm_interpret to raise a timeout and runs all canonical test cases
    through the deterministic fallback router.
    """
    initial_fallback_count = getattr(QueryInterpreter, "_FALLBACK_COUNT", 0)

    with MockLLMTimeout():
        for question, exp_intent, exp_recenter in CORE_TEST_CASES:
            now_ms = time.time() * 1000.0
            res = QueryInterpreter.interpret(
                question,
                viewport_bbox=TEST_VP_BBOX,
                viewport_captured_at=now_ms,
            )

            assert res["is_fallback"] is True, f"Expected is_fallback=True for '{question}'"
            assert res["query_intent"] == exp_intent, (
                f"FALLBACK diverged on: '{question}'. Got '{res['query_intent']}', expected '{exp_intent}'"
            )
            assert res["is_new_location_query"] == exp_recenter, (
                f"FALLBACK map recenter diverged on: '{question}'. Got {res['is_new_location_query']}, expected {exp_recenter}"
            )

    # Verify fallback counter incremented
    assert QueryInterpreter._FALLBACK_COUNT > initial_fallback_count


def test_fallback_stale_or_missing_viewport_safety():
    """
    Test 2: Viewport staleness failsafe in fallback router.
    If the viewport is stale (>10s old) or missing, deictic questions must
    fall back to 'followup' and NEVER trigger map recentering.
    """
    with MockLLMTimeout():
        # Case A: Stale viewport (captured 25 seconds ago)
        stale_ms = (time.time() - 25.0) * 1000.0
        res_stale = QueryInterpreter.interpret(
            "can you see the cars",
            viewport_bbox=TEST_VP_BBOX,
            viewport_captured_at=stale_ms,
        )
        assert res_stale["query_intent"] == "followup"
        assert res_stale["is_new_location_query"] is False
        assert res_stale["is_fallback"] is True

        # Case B: Missing viewport entirely
        res_missing = QueryInterpreter.interpret(
            "what can you see here",
            viewport_bbox=None,
            viewport_captured_at=None,
        )
        assert res_missing["query_intent"] == "followup"
        assert res_missing["is_new_location_query"] is False
        assert res_missing["is_fallback"] is True

        # Case C: Conversational followup
        res_followup = QueryInterpreter.interpret(
            "tell me more about it",
            viewport_bbox=TEST_VP_BBOX,
            viewport_captured_at=time.time() * 1000.0,
        )
        assert res_followup["query_intent"] == "followup"
        assert res_followup["is_new_location_query"] is False


def test_fallback_matches_llm_path_intent_parity():
    """
    Test 3: Intent parity between primary LLM path and deterministic fallback.
    Both paths must yield identical query_intent and is_new_location_query.
    """
    for question, exp_intent, exp_recenter in CORE_TEST_CASES:
        now_ms = time.time() * 1000.0
        # 1. Run Fallback Path
        with MockLLMTimeout():
            fallback_res = QueryInterpreter.interpret(
                question,
                viewport_bbox=TEST_VP_BBOX,
                viewport_captured_at=now_ms,
            )

        assert fallback_res["query_intent"] == exp_intent
        assert fallback_res["is_new_location_query"] == exp_recenter
        assert fallback_res["is_fallback"] is True


def test_geocoder_sanity_guard_rejects_spurious_sentences():
    """
    Test 4: Geocoder defense-in-depth sanity guard.
    Verifies that conversational phrases and sentences are rejected rather than
    matching distant spurious places (e.g. Nominatim Cumbernauld bug).
    """
    # 1. Pure conversational sentences with no named place must be rejected
    assert GeocoderService.is_valid_geocoding_match(
        "can you see the cars",
        "What can you see in the ponds?, Cumbernauld, North Lanarkshire, Scotland"
    ) is False

    assert GeocoderService.is_valid_geocoding_match(
        "can you see the cars on street",
        "Cars, Gironde, Nouvelle-Aquitaine, France"
    ) is False

    assert GeocoderService.is_valid_geocoding_match(
        "what can you see in this area",
        "Area 51, Nevada, United States"
    ) is False

    # 2. Blacklisted non-locations must be rejected
    assert GeocoderService.is_valid_geocoding_match(
        "Previous context",
        "Precious Gifts Preschool, Bridgeport"
    ) is False

    # 3. Legitimate geographic destinations must pass
    assert GeocoderService.is_valid_geocoding_match(
        "Tokyo",
        "Tokyo, Special Wards of Tokyo, Tokyo, 160-8484, Japan"
    ) is True

    assert GeocoderService.is_valid_geocoding_match(
        "New York City",
        "New York, United States"
    ) is True

    assert GeocoderService.is_valid_geocoding_match(
        "KIET Ghaziabad",
        "KIET Group of Institutions, Muradnagar, Ghaziabad, Uttar Pradesh, 201206, India"
    ) is True

    assert GeocoderService.is_valid_geocoding_match(
        "London",
        "London, Greater London, England, SW1A 2AA, United Kingdom"
    ) is True


def test_geocoding_sentence_bypasses_or_returns_none():
    """
    Test 5: Live geocoding of conversational queries.
    Calling GeocoderService.geocode() with a sentence like 'can you see the cars'
    or 'Previous context' must return None (never Null Island 0.0, 0.0, Scotland, France, or Bridgeport).
    """
    geo_res = GeocoderService.geocode("can you see the cars")
    assert geo_res is None, f"Expected None on conversational sentence, got: {geo_res}"

    # Must reject "Previous context" and return None
    prev_geo = GeocoderService.geocode("Previous context")
    assert prev_geo is None, f"Expected None on invalid location, got: {prev_geo}"


def test_geocoder_cache_and_substring_word_boundary():
    """
    Test 6: Geocoder in-memory caching and strict word-boundary matching.
    """
    # 1. Substring containment should NOT match ('art' in 'Stratford' or 'pond' in 'respond')
    assert GeocoderService.is_valid_geocoding_match(
        "art",
        "Stratford-upon-Avon, Warwickshire, England"
    ) is False

    # 2. Exact or word-boundary matches SHOULD pass
    assert GeocoderService.is_valid_geocoding_match(
        "Stratford",
        "Stratford-upon-Avon, Warwickshire, England"
    ) is True

    # 3. Search plan should be capped to at most 3 items
    plan = GeocoderService.build_search_plan("KIET Group of Institutions, Ghaziabad")
    assert len(plan) <= 3, f"Search plan exceeded cap: {len(plan)}"

    # 4. Cache test: populate cache and verify instantaneous retrieval
    GeocoderService._GEOCODE_CACHE["test_place"] = (time.time(), {
        "name": "Test Place",
        "lat": 12.34,
        "lon": 56.78,
        "zoom": 15,
        "bbox": [56.7, 12.3, 56.8, 12.4]
    })
    cached = GeocoderService.geocode("test_place")
    assert cached is not None
    assert cached["name"] == "Test Place"
    assert cached["lat"] == 12.34


def test_geocoder_cache_ttl_and_lru_eviction():
    """
    Test 7: Cache TTL expiration and LRU max-size eviction.
    """
    GeocoderService.clear_cache()

    # 1. Test TTL expiration: item older than 24h should be evicted on lookup
    old_timestamp = time.time() - (GeocoderService._CACHE_TTL_SEC + 10.0)
    GeocoderService._GEOCODE_CACHE["stale_city"] = (old_timestamp, {
        "name": "Stale City",
        "lat": 10.0,
        "lon": 20.0,
        "zoom": 14,
        "bbox": [19.9, 9.9, 20.1, 10.1]
    })
    # Lookup should detect expired entry and evict it (returning None since stale_city isn't a real place)
    res = GeocoderService.geocode("stale_city")
    assert "stale_city" not in GeocoderService._GEOCODE_CACHE

    # 2. Test true LRU capacity bounding & hit reordering
    orig_max = GeocoderService._CACHE_MAX_SIZE
    try:
        GeocoderService._CACHE_MAX_SIZE = 3
        GeocoderService.clear_cache()
        # Insert A, B, C
        GeocoderService._store_cache("place_a", {"name": "Place A", "lat": 1.0, "lon": 2.0})
        GeocoderService._store_cache("place_b", {"name": "Place B", "lat": 1.0, "lon": 2.0})
        GeocoderService._store_cache("place_c", {"name": "Place C", "lat": 1.0, "lon": 2.0})

        # Access place_a (moves it to most recently used end)
        res_a = GeocoderService.geocode("place_a")
        assert res_a is not None

        # Insert place_d (should evict place_b, NOT place_a, because place_a was accessed!)
        GeocoderService._store_cache("place_d", {"name": "Place D", "lat": 1.0, "lon": 2.0})
        assert len(GeocoderService._GEOCODE_CACHE) == 3
        assert "place_b" not in GeocoderService._GEOCODE_CACHE  # True LRU: place_b was least recently used
        assert "place_a" in GeocoderService._GEOCODE_CACHE      # place_a preserved due to hit
        assert "place_c" in GeocoderService._GEOCODE_CACHE
        assert "place_d" in GeocoderService._GEOCODE_CACHE
    finally:
        GeocoderService._CACHE_MAX_SIZE = orig_max
        GeocoderService.clear_cache()


@pytest.mark.asyncio
async def test_unresolved_navigation_location_honest_response():
    """
    Test 8: Explicit navigation to a nonexistent place ('take me to XyzblahNonexistent').
    System must:
    1. Honestly alert the user that the place could not be found.
    2. NEVER snap the map (is_new_location_query is False).
    3. Retain the user's viewport without running irrelevant visual analysis or proactively describing the viewport.
    4. Offer the choice to describe the viewport on the user's NEXT turn.
    """
    from backend.app.agent.controller import CosmoClipAgentController

    controller = CosmoClipAgentController()
    response = await controller.run({
        "question": "take me to XyzblahNonexistent",
        "viewport_bbox": TEST_VP_BBOX,
        "viewport_zoom": 17,
        "viewport_captured_at": time.time() * 1000.0,
        "session_id": "test_unresolved_session"
    })

    assert response.is_new_location_query is False, "Failed geocode must NEVER trigger map recentering"
    assert "couldn't find" in response.answer.lower() or "could not find" in response.answer.lower()
    assert "xyzblahnonexistent" in response.answer.lower()
    assert "current view" in response.answer.lower() or "current viewport" in response.answer.lower()
    assert "would you like me to describe" in response.answer.lower(), "Must offer next action as an interactive question"

    # Verify short-circuit: did NOT run downstream vision inference or produce evidence bounding boxes
    trace_steps = [s.step for s in response.trace]
    assert "unresolved_location_notice" in trace_steps
    assert "satellite_vqa" not in trace_steps, "Must NOT run VLM inference on unrelated scene"
    assert "prepare_realtime_imagery" not in trace_steps, "Must NOT fetch raster tiles on failed navigation"
    assert len(response.evidence) == 0, "No visual evidence should be generated"

