import os
import re
import json
import time
import urllib.request
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
try:
    from app.geospatial.geocoder import GeocoderService
    from app.api.logs import emit_log
except ImportError:
    from backend.app.geospatial.geocoder import GeocoderService
    from backend.app.api.logs import emit_log

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# ---------------------------------------------------------------------------
# Spatial-intent keyword tables & guards
# ---------------------------------------------------------------------------
NAVIGATION_VERBS = [
    "take me to", "show me", "go to", "navigate to", "fly to",
    "look at", "search for", "find ", "jump to", "zoom to", "move to",
    "switch to", "head to", "where is", "locate "
]

DEICTIC_KEYWORDS = [
    "current place", "current location", "this place", "this area",
    "this location", "here", "what can you see", "what am i looking at",
    "what is this", "tell me about this", "where am i",
    "what do you see", "what are we looking at", "look at this", "this map",
    "current view", "what is here", "on the screen", "on screen",
]

CURRENT_VIEW_MARKERS = [
    "current view", "active view", "this view", "particular view", "on screen",
    "on the screen", "in view", "in this view", "detail regarding", "details regarding",
    "detail of", "details of", "detail about", "details about", "what do you see",
    "what can you see", "what am i looking at", "what is this", "tell me about this",
    "describe this", "what is visible", "features in view", "satellite view", "map view"
]

# Generic nouns that an LLM extractor or naive regex might mistake for place names.
# These should NEVER alone trigger navigation intent or be geocoded as places.
GENERIC_LOCATION_NOISE = {
    "street", "road", "avenue", "lane", "highway", "bridge", "area", "place",
    "location", "here", "there", "spot", "zone", "region", "site", "scene",
    "building", "buildings", "corner", "intersection", "view", "map",
    "park", "field", "ground", "grounds", "water", "river", "lake", "sea",
    "city", "town", "village", "country", "earth", "world", "globe",
    "car", "cars", "vehicle", "vehicles", "tree", "trees", "roof", "roofs",
}

INVALID_LOCATION_HINTS = {
    "previous context", "previous", "context", "none", "null", "undefined",
    "target location", "current location", "current viewport location", "current context",
    "global", "earth", "world"
}

STOPWORDS = {
    "show", "me", "locate", "find", "tell", "about", "view", "where", "is",
    "the", "a", "an", "in", "at", "near", "of", "to", "can", "you", "see", "previous", "context"
}

FINE_DETAIL_KEYWORDS = [
    "count", "how many", "this area", "here", "visible", "in view", "on the street",
    "car", "cars", "vehicle", "vehicles", "building", "buildings", "roof", "roofs",
    "boat", "boats", "ship", "ships", "truck", "trucks", "bus", "buses",
    "pedestrian", "pedestrians", "person", "people", "what am i looking at",
    "what is this", "corner", "lane", "parking", "pool", "tree", "trees", "intersection",
    "screen", "on the screen", "what you can see", "what do you see", "what can you see",
    "current view", "what is here", "what are we looking at", "look at this", "this map"
]


class QueryInterpreter:
    """
    Unified LLM-First Query Interpreter and Geospatial Router.

    In a single model call, it disambiguates the user question, extracts
    named destinations (guarded against generic location noise), classifies
    spatial intent (navigation vs viewport_bound vs followup), and detects
    fine-detail requirements.

    Deterministic cross-checks ensure live viewport freshness and bounds
    govern whether viewport_bound or followup takes effect.
    """

    # In-memory query interpretation cache (query+vp_active -> parsed LLM result)
    _INTERPRET_CACHE: Dict[str, Dict[str, Any]] = {}
    _FALLBACK_COUNT: int = 0

    @staticmethod
    def _viewport_is_fresh(viewport_captured_at: Optional[float], max_age_s: float = 10.0) -> bool:
        """Returns True if captured_at timestamp is within max_age_s seconds of now."""
        if not viewport_captured_at:
            return False
        # Normalize to milliseconds (frontend sends ms epoch Date.now() > 1e11, Python time.time() < 1e11)
        ts_ms = viewport_captured_at if viewport_captured_at > 1e11 else viewport_captured_at * 1000.0
        age_s = (time.time() * 1000.0 - ts_ms) / 1000.0
        return 0.0 <= age_s < max_age_s

    @staticmethod
    def _llm_interpret(question: str, has_viewport: bool = False) -> Optional[Dict[str, Any]]:
        """
        Unified LLM Query Interpretation:
        In a single LLM call, performs:
          1. Spelling & grammar normalization
          2. Named destination place extraction (strictly distinct from current viewport)
          3. Spatial intent classification ('navigation' | 'viewport_bound' | 'followup')
          4. Fine-detail requirement detection (cars, buildings, trees, people, etc.)
          5. Temporal bounds and task type detection
          6. Reasoning phrase for full observability
        """
        # 1. Try Mistral AI (Fast & reliable JSON completions)
        if MISTRAL_API_KEY:
            try:
                url = "https://api.mistral.ai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json"
                }
                prompt = f"""You are an expert Geospatial Location & Intent Interpreter for a satellite imagery map cockpit.
Analyze this user query: "{question}"
Has a live map viewport active on screen: {has_viewport}

Return ONLY a valid JSON object matching this schema:
{{
  "corrected_question": "Clean, grammatically corrected query with typos fixed",
  "named_place": "Extracted specific destination place, institution, city, or landmark name if explicitly named by user to navigate to, else null",
  "location_query": "Canonical search string (<Place Name>, <City>, <State>, <Country>) if named_place is not null, else null",
  "spatial_intent": "navigation" | "viewport_bound" | "followup",
  "wants_fine_detail": true | false,
  "task": "change_analysis" | "vqa" | "grounding" | "captioning",
  "target_entity": "education_campus" | "urban" | "water" | "vegetation" | "infrastructure" | "general",
  "is_comparison": true | false,
  "baseline_year": "2020",
  "current_year": "2026",
  "reasoning": "One short phrase explaining the spatial intent classification"
}}

Classification Rules:
1. "navigation": User explicitly names a specific place, institution, city, or landmark to search for or go to (e.g., "take me to KIET Ghaziabad", "where is Eiffel Tower", "show me Tokyo").
2. "viewport_bound": User is explicitly asking about what is currently visible on screen, visible in the map view, or the current area, without naming any specific destination — includes questions about visible objects, deictic phrasing, counting, or inspecting the current scene (e.g. "what can you see", "can you see the cars on street", "are there buildings visible", "how has this area changed", "what am I looking at", "tell me about this place", "count the trees").
3. "followup": Conversational or ambiguous query referring to prior dialogue, the previous answer, or pronouns (e.g. "tell me more", "tell me more about it", "explain further", "why did that happen", "what else") that does not explicitly mention what is visible on the map/screen. Such queries MUST be classified as "followup".
4. A generic noun like "street", "road", "area", "place", "location", "here", "there", "view", "map", "corner", "intersection", "building", "cars" is NEVER a named place unless combined with an actual proper noun (e.g. "Main Street", "Abbey Road"). If the user asks "can you see the cars on street", named_place MUST be null and spatial_intent MUST be "viewport_bound".
5. wants_fine_detail: Set to true if the user asks about small localized objects requiring high resolution: cars, vehicles, people, roof details, road lanes, trees, boats, etc.
"""
                payload = {
                    "model": "open-mistral-nemo",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=8.0) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    content_str = res_data["choices"][0]["message"]["content"]
                    norm = json.loads(content_str)
                    emit_log("SUCCESS", "MISTRAL-ROUTER", f"Unified interpretation: intent='{norm.get('spatial_intent')}', named_place='{norm.get('named_place')}', fine_detail={norm.get('wants_fine_detail')}")
                    return norm
            except Exception as me:
                emit_log("WARNING", "MISTRAL-ROUTER", f"Mistral interpreter fallback: {str(me)[:80]}")

        # 2. Try Gemini Flash if available
        if GEMINI_API_KEY:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=GEMINI_API_KEY)
                prompt = f"""You are an expert Geospatial Location & Intent Interpreter for a satellite imagery map cockpit.
Analyze this user query: "{question}"
Has a live map viewport active on screen: {has_viewport}

Return ONLY a valid JSON object matching:
{{
  "corrected_question": "Clean query",
  "named_place": "Specific destination if user named one to navigate to, else null",
  "location_query": "Canonical search string (<Place Name>, <City>, <State>, <Country>) if named_place is not null, else null",
  "spatial_intent": "navigation" | "viewport_bound" | "followup",
  "wants_fine_detail": true | false,
  "task": "change_analysis" | "vqa" | "grounding",
  "target_entity": "education_campus" | "urban" | "water" | "vegetation" | "infrastructure" | "general",
  "is_comparison": true | false,
  "baseline_year": "2020",
  "current_year": "2026",
  "reasoning": "Short explanation"
}}
Rules:
- 'navigation': user explicitly names a place to navigate to (e.g. 'where is KIET').
- 'viewport_bound': user asks about current on-screen map/view/objects without naming a destination (e.g. 'can you see the cars on street', 'what can you see', 'what is this').
- 'followup': ambiguous conversational query.
- Generic nouns like 'street', 'road', 'place', 'area', 'here' are NEVER named places.
"""
                for model_candidate in ["gemini-2.5-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest", "gemini-2.5-flash"]:
                    try:
                        resp = client.models.generate_content(
                            model=model_candidate,
                            contents=prompt,
                            config=types.GenerateContentConfig(response_mime_type="application/json")
                        )
                        if resp and resp.text:
                            data = json.loads(resp.text.strip())
                            emit_log("SUCCESS", "GEMINI-ROUTER", f"Unified interpretation: intent='{data.get('spatial_intent')}', named_place='{data.get('named_place')}'")
                            return data
                    except Exception:
                        continue
            except Exception as ge:
                emit_log("WARNING", "GEMINI-ROUTER", f"Gemini interpreter fallback: {str(ge)[:80]}")

        return None

    @staticmethod
    def _llm_normalize(question: str) -> Optional[Dict[str, Any]]:
        """Backward-compatible alias for _llm_interpret."""
        return QueryInterpreter._llm_interpret(question, has_viewport=False)

    @staticmethod
    def _extract_location_fallback(question: str) -> str:
        """
        Regex fallback to isolate target place name when LLM is unavailable.
        """
        q = question.strip()
        clean = q.lower()

        # Reject conversational question starters and visual inspection commands
        if clean.startswith(("what ", "what's ", "how ", "why ", "when ", "who ", "can you ", "could you ", "tell me ", "count ", "describe ", "detect ", "is there ", "are there ", "look at ")):
            if not any(v in clean for v in ["take me to", "navigate to", "fly to", "zoom to", "where is", "locate "]):
                return ""

        q = re.sub(r'^(take\s+me\s+to|go\s+to|fly\s+to|navigate\s+to|zoom\s+to|jump\s+to|switch\s+to|locate|search\s+for|compare|comparison|show\s+me|show|find|where\s+is|what\s+is\s+the|what\s+are\s+the|analyze|inspect|view)\s+', '', q, flags=re.IGNORECASE)
        q = re.sub(r'^(satellite\s+changes|satellite\s+view|satellite\s+imagery|changes\s+in|changes\s+for|changes\s+around|expansion\s+of|growth\s+of)\s+', '', q, flags=re.IGNORECASE)
        q = re.sub(r'^(for|around|in|near|at|over|of|to)\s+', '', q, flags=re.IGNORECASE)
        q = re.sub(r'\s+(between|from|in|during)\s+(20\d\d)\s+(and|to|vs|versus)\s+(20\d\d).*$', '', q, flags=re.IGNORECASE)
        q = re.sub(r'\s+(in\s+20\d\d|since\s+20\d\d|from\s+20\d\d).*$', '', q, flags=re.IGNORECASE)
        q = re.sub(r'\b(ollage|collage|collge|colleg)\b', 'College', q, flags=re.IGNORECASE)
        q = re.sub(r'\b(univeristy|univercity)\b', 'University', q, flags=re.IGNORECASE)
        q = q.strip(" ?.,!\"'")

        clean_sub = q.lower()
        if clean_sub in GENERIC_LOCATION_NOISE or clean_sub in INVALID_LOCATION_HINTS or len(clean_sub) < 2:
            return ""

        # Check that distinctive tokens exist (not just stopwords)
        clean_words = [w for w in re.findall(r'\b[a-z0-9]{2,}\b', clean_sub) if w not in GENERIC_LOCATION_NOISE and w not in STOPWORDS]
        if not clean_words:
            return ""

        return q

    @staticmethod
    def _contains_keyword(text: str, keywords: List[str]) -> bool:
        """Word-boundary regex match to prevent substring false positives (e.g. 'here' in 'where')."""
        for kw in keywords:
            if " " in kw:
                if kw in text:
                    return True
            else:
                if re.search(r'\b' + re.escape(kw) + r'\b', text):
                    return True
        return False

    @staticmethod
    def _classify_spatial_intent(
        question: str,
        viewport_bbox: Optional[List[float]],
        viewport_captured_at: Optional[float],
        location_hint: Optional[str],
    ) -> str:
        """
        Deterministic keyword & viewport classifier used as fallback when LLM is offline.
        """
        clean_q = question.strip().lower()
        has_nav_verb = QueryInterpreter._contains_keyword(clean_q, NAVIGATION_VERBS)
        has_loc_hint = bool(
            location_hint and
            len(location_hint.strip()) > 2 and
            location_hint.strip().lower() not in INVALID_LOCATION_HINTS and
            "global" not in location_hint.lower() and
            "earth" not in location_hint.lower()
        )
        has_deictic = QueryInterpreter._contains_keyword(clean_q, DEICTIC_KEYWORDS) or bool(re.search(r'\b(this|these)\s+[a-z0-9]+\b', clean_q))
        has_fine_detail = QueryInterpreter._contains_keyword(clean_q, FINE_DETAIL_KEYWORDS)
        has_analytical = QueryInterpreter._contains_keyword(clean_q, [
            "ndvi", "ndwi", "spectral", "vegetation", "indices", "band", "reflectance"
        ])
        has_comp = any(w in clean_q for w in ["compare", "comparison", "last year", "changes", "change", "over time"])
        has_followup_word = any(clean_q.startswith(w) or f" {w}" in clean_q for w in [
            "tell me more", "explain", "why", "what else", "elaborate", "details",
            "how so", "more info", "what happened", "continue", "next"
        ])

        # Distinctive place tokens in query
        clean_words = [
            w for w in re.findall(r'\b[a-z0-9]{2,}\b', clean_q)
            if w not in GENERIC_LOCATION_NOISE and w not in STOPWORDS
        ]
        is_standalone_place = (
            bool(clean_words) and
            not has_deictic and
            not has_fine_detail and
            not has_analytical and
            not has_comp and
            not has_followup_word and
            not clean_q.startswith(("what", "how", "why", "when", "who", "can", "could", "is", "are", "do", "does", "did"))
        )

        if has_followup_word:
            return "followup"

        if has_nav_verb or has_loc_hint or is_standalone_place:
            return "navigation"

        return "viewport_bound"

    @staticmethod
    def interpret(
        question: str,
        location_hint: Optional[str] = None,
        viewport_bbox: Optional[List[float]] = None,
        viewport_captured_at: Optional[float] = None,
        viewport_zoom: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Interprets a query and resolves spatial intent, target location, and viewport usage.
        Follows the Unified LLM-First design with deterministic cross-checks.
        """
        clean_q = question.strip().lower()
        has_vp_bounds = bool(viewport_bbox and len(viewport_bbox) == 4)
        is_fresh_vp = has_vp_bounds and QueryInterpreter._viewport_is_fresh(viewport_captured_at)

        # 1. Attempt cached or live Unified LLM interpretation
        cache_key = f"{clean_q}::vp={has_vp_bounds}::hint={location_hint or ''}"
        llm_res = QueryInterpreter._INTERPRET_CACHE.get(cache_key)
        if not llm_res:
            try:
                llm_res = QueryInterpreter._llm_interpret(question, has_viewport=has_vp_bounds)
                if llm_res:
                    if len(QueryInterpreter._INTERPRET_CACHE) > 250:
                        QueryInterpreter._INTERPRET_CACHE.clear()
                    QueryInterpreter._INTERPRET_CACHE[cache_key] = llm_res
            except Exception as e:
                emit_log("WARNING", "ROUTER-FALLBACK", f"LLM interpreter call raised exception: {e}. Engaging fallback.")
                llm_res = None

        # 2. Extract and sanitize unified outputs
        if llm_res:
            is_fallback = False
            corrected_question = llm_res.get("corrected_question") or question
            raw_named_place = llm_res.get("named_place")
            spatial_intent_raw = (llm_res.get("spatial_intent") or "").lower().strip()
            wants_fine_detail = bool(llm_res.get("wants_fine_detail"))
            task = llm_res.get("task", "vqa")
            target = llm_res.get("target_entity", "general")
            is_comp = bool(llm_res.get("is_comparison"))
            baseline_year = llm_res.get("baseline_year")
            current_year = llm_res.get("current_year", "2026")
            reasoning = llm_res.get("reasoning") or "LLM spatial classification"
            loc_search_str_llm = llm_res.get("location_query")

            # Clean and filter named place — generic nouns are never accepted as destinations
            named_place: Optional[str] = None
            if raw_named_place and isinstance(raw_named_place, str):
                cleaned_np = raw_named_place.strip()
                if cleaned_np.lower() not in GENERIC_LOCATION_NOISE and len(cleaned_np) >= 3:
                    named_place = cleaned_np
        else:
            # Deterministic Fallback if LLM is offline, rate-limited, or times out
            is_fallback = True
            QueryInterpreter._FALLBACK_COUNT += 1
            emit_log(
                "WARNING", "ROUTER-FALLBACK",
                f"⚠️ LLM interpreter unavailable/timed out (total fallbacks: {QueryInterpreter._FALLBACK_COUNT}). Engaging deterministic fallback router for: '{question[:50]}'"
            )
            corrected_question = question
            reasoning = "Rule-based fallback router (LLM offline/timeout)"
            task = "change_analysis" if any(w in clean_q for w in ["change", "changes", "compare", "vs"]) else "vqa"
            target = "general"
            is_comp = any(w in clean_q for w in ["change", "changes", "compare", "vs"])
            baseline_year = "2020"
            current_year = "2026"
            wants_fine_detail = QueryInterpreter._contains_keyword(clean_q, FINE_DETAIL_KEYWORDS)
            loc_search_str_llm = None

            spatial_intent_raw = QueryInterpreter._classify_spatial_intent(
                question, viewport_bbox, viewport_captured_at, location_hint
            )
            named_place = None
            if spatial_intent_raw == "navigation":
                raw_extracted = QueryInterpreter._extract_location_fallback(question)
                if raw_extracted and raw_extracted.strip().lower() not in GENERIC_LOCATION_NOISE and len(raw_extracted.strip()) >= 3:
                    named_place = raw_extracted.strip()

        # 3. Deterministic Cross-Check & Single Source of Truth
        has_explicit_loc_hint = bool(
            location_hint and
            len(location_hint.strip()) > 2 and
            location_hint.strip().lower() not in INVALID_LOCATION_HINTS and
            "global" not in location_hint.lower() and
            "earth" not in location_hint.lower()
        )

        is_fine_detail = wants_fine_detail or QueryInterpreter._contains_keyword(clean_q, FINE_DETAIL_KEYWORDS)

        # Authoritative Spatial Intent:
        # 1. Viewport-bound / Current-view: user is asking about current viewport or visible screen features
        # IMPORTANT: Only use classify() as a tiebreaker when spatial_intent_raw is NOT already "followup".
        # classify() (via extract_place_entity) can't distinguish true follow-ups from screen-view queries.
        _classify_current_view = False
        if spatial_intent_raw not in ("followup",):
            try:
                _classify_current_view = classify(question) == "current_view"
            except Exception:
                pass
        is_screen_query = (
            spatial_intent_raw in ("viewport_bound", "current_view") or
            any(m in clean_q for m in CURRENT_VIEW_MARKERS) or
            _classify_current_view
        )
        if is_screen_query and not named_place:
            if not is_fresh_vp:
                query_intent = "followup"
            else:
                query_intent = "viewport_bound"
            named_place = None  # Viewport/screen questions never navigate
        # 2. Navigation: explicit destination named by user OR caller explicitly requested navigation to a place
        elif named_place or (has_explicit_loc_hint and not is_screen_query) or (spatial_intent_raw == "navigation") or any(clean_q.startswith(p) for p in ["take me to ", "go to ", "fly to ", "navigate to ", "zoom to "]):
            query_intent = "navigation"
            if not named_place:
                extracted_target = QueryInterpreter._extract_location_fallback(question)
                if extracted_target:
                    named_place = extracted_target
        # 3. Followup: stale viewport, ambiguous context, or conversational reference
        else:
            query_intent = "followup"

        # Explicit year extraction fallback from question text
        explicit_year = re.search(r'\b(201\d|202[0-5])\b', question)
        if explicit_year:
            baseline_year = explicit_year.group(1)
        elif not baseline_year:
            baseline_year = "2020"

        # -------------------------------------------------------------------
        # Case 1: VIEWPORT-BOUND — live viewport / current screen in view
        # -------------------------------------------------------------------
        if query_intent in ("viewport_bound", "current_view"):
            if has_vp_bounds:
                center_lat = (viewport_bbox[1] + viewport_bbox[3]) / 2.0
                center_lon = (viewport_bbox[0] + viewport_bbox[2]) / 2.0
                resolved_place_name = GeocoderService.reverse_geocode(center_lat, center_lon) or f"Area at {center_lat:.4f}°N, {center_lon:.4f}°E"
                geocode_info = {
                    "name": resolved_place_name,
                    "lat": center_lat,
                    "lon": center_lon,
                    "zoom": 19 if is_fine_detail else 17,
                    "bbox": viewport_bbox,
                    "display_name": resolved_place_name
                }
            else:
                resolved_place_name = "Current Screen View"
                geocode_info = {
                    "name": "Current Screen View",
                    "lat": 0.0,
                    "lon": 0.0,
                    "zoom": 17,
                    "bbox": [-180, -90, 180, 90],
                    "display_name": "Current Screen View"
                }

            emit_log("SUCCESS", "ROUTER", f"[intent=viewport_bound] {reasoning} -> Locked onto live viewport: '{resolved_place_name}'")

            return {
                "task": task,
                "target": target,
                "is_comparison": is_comp,
                "is_fine_detail": is_fine_detail,
                "wants_fine_detail": is_fine_detail,
                "use_viewport_bbox": has_vp_bounds,
                "is_new_location_query": False,  # Map must NOT move
                "needs_grounding": True,
                "geocoded_location": geocode_info,
                "corrected_question": corrected_question,
                "location_query": resolved_place_name,
                "baseline_year": baseline_year,
                "current_year": current_year,
                "query_intent": "viewport_bound",
                "reasoning": reasoning,
                "is_fallback": is_fallback,
                "extracted_keywords": [w for w in re.findall(r'\b\w+\b', clean_q) if len(w) > 3]
            }

        # -------------------------------------------------------------------
        # Case 2: FOLLOWUP — stale/missing viewport, conversational reference
        # -------------------------------------------------------------------
        if query_intent == "followup":
            if has_vp_bounds:
                center_lat = (viewport_bbox[1] + viewport_bbox[3]) / 2.0
                center_lon = (viewport_bbox[0] + viewport_bbox[2]) / 2.0
                resolved_name = GeocoderService.reverse_geocode(center_lat, center_lon) or f"Area at {center_lat:.4f}°N, {center_lon:.4f}°E"
                emit_log("INFO", "ROUTER", f"[intent=followup] {reasoning} -> Stale viewport used as silent context: '{resolved_name}'")
                geocode_info = {
                    "name": resolved_name,
                    "lat": center_lat,
                    "lon": center_lon,
                    "zoom": 17,
                    "bbox": viewport_bbox,
                    "display_name": resolved_name
                }
            else:
                emit_log("INFO", "ROUTER", f"[intent=followup] {reasoning} -> No viewport available, preserving context")
                geocode_info = {
                    "name": "Current context",
                    "lat": 20.0,
                    "lon": 78.0,
                    "zoom": 14,
                    "bbox": [-180, -60, 180, 60],
                    "display_name": "Current context"
                }

            return {
                "task": task,
                "target": target,
                "is_comparison": is_comp,
                "is_fine_detail": is_fine_detail,
                "wants_fine_detail": is_fine_detail,
                "use_viewport_bbox": has_vp_bounds,
                "is_new_location_query": False,  # Followup NEVER moves the map
                "needs_grounding": True,
                "geocoded_location": geocode_info,
                "corrected_question": corrected_question,
                "location_query": geocode_info["name"] if has_vp_bounds else None,
                "baseline_year": baseline_year,
                "current_year": current_year,
                "query_intent": "followup",
                "reasoning": reasoning,
                "is_fallback": is_fallback,
                "extracted_keywords": [w for w in re.findall(r'\b\w+\b', clean_q) if len(w) > 3]
            }

        # -------------------------------------------------------------------
        # Case 3: NAVIGATION — explicit destination place requested
        # -------------------------------------------------------------------
        loc_search_str = location_hint.strip() if has_explicit_loc_hint else (loc_search_str_llm or named_place)
        emit_log("SUCCESS", "ROUTER", f"[intent=navigation] {reasoning} -> Resolving geocode for '{loc_search_str}'...")
        geocode_info = GeocoderService.geocode(loc_search_str)

        # DEFENSE-IN-DEPTH: If geocoding fails completely, do NOT snap to Null Island or recenter map.
        if not geocode_info:
            emit_log("WARNING", "ROUTER", f"Geocoding failed for navigation target '{loc_search_str}'. Safe fallback without moving map.")
            if has_vp_bounds:
                center_lat = (viewport_bbox[1] + viewport_bbox[3]) / 2.0
                center_lon = (viewport_bbox[0] + viewport_bbox[2]) / 2.0
                fallback_geo = GeocoderService._build_geo_result("Current Viewport Location", center_lat, center_lon, default_zoom=viewport_zoom or 15)
                fallback_geo["bbox"] = viewport_bbox
                return {
                    "task": task,
                    "target": target,
                    "is_comparison": is_comp,
                    "is_fine_detail": is_fine_detail,
                    "wants_fine_detail": is_fine_detail,
                    "use_viewport_bbox": True,
                    "is_new_location_query": False,  # Failed geocode must NEVER recenter map
                    "needs_grounding": True,
                    "geocoded_location": fallback_geo,
                    "corrected_question": corrected_question,
                    "location_query": None,
                    "baseline_year": baseline_year,
                    "current_year": current_year,
                    "query_intent": "navigation",
                    "geocoding_failed": True,
                    "unresolved_place": loc_search_str,
                    "reasoning": f"Navigation target '{loc_search_str}' could not be geocoded; retaining current viewport with explicit explanation.",
                    "is_fallback": True,
                    "extracted_keywords": [w for w in re.findall(r'\b\w+\b', clean_q) if len(w) > 3]
                }
            else:
                fallback_geo = {
                    "name": "Current context",
                    "lat": 20.0,
                    "lon": 78.0,
                    "zoom": 14,
                    "bbox": [-180, -60, 180, 60],
                    "display_name": "Current context"
                }
                return {
                    "task": task,
                    "target": target,
                    "is_comparison": is_comp,
                    "is_fine_detail": is_fine_detail,
                    "wants_fine_detail": is_fine_detail,
                    "use_viewport_bbox": False,
                    "is_new_location_query": False,  # Failed geocode must NEVER recenter map
                    "needs_grounding": False,
                    "geocoded_location": fallback_geo,
                    "corrected_question": corrected_question,
                    "location_query": None,
                    "baseline_year": baseline_year,
                    "current_year": current_year,
                    "query_intent": "navigation",
                    "geocoding_failed": True,
                    "unresolved_place": loc_search_str,
                    "reasoning": f"Navigation target '{loc_search_str}' could not be geocoded.",
                    "is_fallback": True,
                    "extracted_keywords": [w for w in re.findall(r'\b\w+\b', clean_q) if len(w) > 3]
                }

        return {
            "task": task,
            "target": target,
            "is_comparison": is_comp,
            "is_fine_detail": is_fine_detail,
            "wants_fine_detail": is_fine_detail,
            "use_viewport_bbox": False,
            "is_new_location_query": True,  # Authoritative trigger to recenter map
            "needs_grounding": task in ["grounding", "change_analysis"] or any(w in clean_q for w in ["detail", "where", "highlight", "box", "show me", "count"]),
            "geocoded_location": geocode_info,
            "corrected_question": corrected_question,
            "location_query": loc_search_str,
            "baseline_year": baseline_year,
            "current_year": current_year,
            "query_intent": "navigation",
            "reasoning": reasoning,
            "is_fallback": is_fallback,
            "extracted_keywords": [w for w in re.findall(r'\b\w+\b', clean_q) if len(w) > 3]
        }


def extract_place_entity(question: str) -> Optional[str]:
    """
    Extracts a specific named destination or place entity from the question.
    Returns None if the question is asking about the current screen/view or contains generic words.
    """
    clean = question.strip().lower()

    # If asking about the current/active view or visible details, it is NEVER a navigation query
    if any(marker in clean for marker in CURRENT_VIEW_MARKERS):
        if not any(v in clean for v in ["take me to", "navigate to", "fly to", "zoom to", "where is", "locate "]):
            return None

    # 1. First check if question starters reject conversational phrases
    if clean.startswith((
        "what ", "what's ", "how ", "why ", "when ", "who ", "can you ", "could you ",
        "tell me ", "count ", "describe ", "detect ", "is there ", "are there ", "look at ",
        "detail ", "details ", "give me ", "explain ", "analyze "
    )):
        # Check if there is an explicit navigation verb inside
        if not any(v in clean for v in ["take me to", "navigate to", "fly to", "zoom to", "where is", "locate "]):
            return None

    # 2. Extract with LLM if available
    try:
        llm_res = QueryInterpreter._llm_interpret(question, has_viewport=False)
    except Exception:
        llm_res = None

    if llm_res and llm_res.get("named_place"):
        np = str(llm_res["named_place"]).strip()
        if np.lower() not in GENERIC_LOCATION_NOISE and len(np) >= 3 and not any(m in np.lower() for m in CURRENT_VIEW_MARKERS):
            return np

    # 3. Rule-based / regex extraction fallback
    fallback = QueryInterpreter._extract_location_fallback(question)
    if fallback and fallback.strip().lower() not in GENERIC_LOCATION_NOISE and len(fallback.strip()) >= 3 and not any(m in fallback.lower() for m in CURRENT_VIEW_MARKERS):
        return fallback.strip()

    return None


def route_query(question: str) -> Dict[str, Any]:
    """
    Binary two-way split:
    - 'navigation': Named place present -> Geocode place -> fetch AOI -> recenter
    - 'current_view': Everything else -> Screenshot the live map, send straight to VLM
    """
    named_place = extract_place_entity(question)
    if named_place:
        return {"query_type": "navigation", "target_entity": named_place}
    return {"query_type": "current_view"}


def classify(question: str) -> str:
    """
    Binary classification between navigation and current_view.
    """
    named_place = extract_place_entity(question)
    return "navigation" if named_place else "current_view"

