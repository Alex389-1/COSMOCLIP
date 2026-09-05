import os
import re
import json
import urllib.request
from typing import Dict, Any, Optional
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

class QueryInterpreter:

    """
    Intelligent LLM-Assisted Query Normalizer and Geospatial Intent Router.
    Corrects user typos, phonetic speech-to-text confusion, and extracts canonical
    place names and temporal bounds BEFORE geocoding.
    """

    @staticmethod
    def _llm_normalize(question: str) -> Optional[Dict[str, Any]]:
        """
        Calls Mistral AI / Google Gemini to disambiguate misspelled place names,
        speech errors, and extract canonical search entities while strictly preserving
        the user-specified city and region.
        """
        # 1. Try Mistral AI (Fast & reliable JSON completions)
        if MISTRAL_API_KEY:
            try:
                url = "https://api.mistral.ai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json"
                }
                prompt = f"""You are an expert Geospatial Location & Query Disambiguator for Satellite AI.
The user provided this raw text or speech-to-text query: "{question}"

Instructions:
1. Fix all spelling mistakes and typos (e.g., 'colage'/'collage' -> 'College', 'ghziabad' -> 'Ghaziabad', 'saterlite' -> 'satellite').
2. STRICTLY PRESERVE the user's intended city, district, and state. NEVER replace or substitute with a different city.
3. Identify the true real-world geographic entity in that specified region (e.g. if the user asked for KIET in Ghaziabad, identify 'KIET Group of Institutions').
4. Formulate the 'location_query' as: '<Exact Place Name>, <City>, <State>, <Country>' for precise OpenStreetMap / STAC geocoding.

Return ONLY a valid JSON object matching this schema:
{{
  "corrected_question": "Clear, complete user question with proper grammar",
  "place_name": "Core institution, infrastructure, or natural feature name",
  "city": "City or District",
  "state": "State or Province",
  "country": "Country",
  "location_query": "Full search query in format: Place Name, City, State, Country",
  "task": "change_analysis" | "vqa" | "grounding" | "captioning",
  "target_entity": "education_campus" | "urban" | "water" | "vegetation" | "infrastructure" | "general",
  "is_comparison": true,
  "baseline_year": "2016",
  "current_year": "2026"
}}
"""
                payload = {
                    "model": "open-mistral-nemo",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    content_str = res_data["choices"][0]["message"]["content"]
                    norm = json.loads(content_str)
                    emit_log("SUCCESS", "MISTRAL-AI", f"Normalized query: '{question}' -> '{norm.get('location_query')}'")
                    return norm
            except Exception as me:
                emit_log("WARNING", "MISTRAL-AI", f"Mistral normalization fallback: {str(me)[:80]}")

        # 2. Try Gemini 2.5 / 2.0 Flash if available
        if GEMINI_API_KEY:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=GEMINI_API_KEY)
                prompt = f"""You are an expert geospatial query interpreter for satellite Earth observation.
The user asked: "{question}"
Return ONLY a valid JSON object with:
{{
  "corrected_question": "Clean, grammatically corrected query",
  "location_query": "Canonical search string: <Place Name>, <City>, <State>, <Country>",
  "task": "change_analysis" or "vqa",
  "target_entity": "education_campus" or "urban" or "water" or "infrastructure" or "general",
  "is_comparison": true or false,
  "baseline_year": "2021-2025",
  "current_year": "2026"
}}"""
                for model_candidate in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                    try:
                        resp = client.models.generate_content(
                            model=model_candidate,
                            contents=prompt,
                            config=types.GenerateContentConfig(response_mime_type="application/json")
                        )
                        if resp and resp.text:
                            data = json.loads(resp.text.strip())
                            if data.get("location_query"):
                                emit_log("SUCCESS", "GEMINI-LIVE", f"Normalized query: '{question}' -> '{data.get('location_query')}'")
                                return data
                    except Exception:
                        continue
            except Exception as ge:
                emit_log("WARNING", "GEMINI-LIVE", f"Gemini normalizer fallback: {str(ge)[:80]}")

        return None

    @staticmethod
    def _extract_location_fallback(question: str) -> str:
        """
        Strips common question phrasing, satellite keywords, and temporal intervals
        to isolate the core target place name.
        """
        q = question.strip()
        # Remove common query prefixes
        q = re.sub(r'^(take\s+me\s+to|go\s+to|fly\s+to|navigate\s+to|zoom\s+to|jump\s+to|switch\s+to|locate|search\s+for|compare|comparison|show\s+me|show|find|where\s+is|what\s+is\s+the|what\s+are\s+the|analyze|inspect|view)\s+', '', q, flags=re.IGNORECASE)
        q = re.sub(r'^(satellite\s+changes|satellite\s+view|satellite\s+imagery|changes\s+in|changes\s+for|changes\s+around|expansion\s+of|growth\s+of)\s+', '', q, flags=re.IGNORECASE)
        q = re.sub(r'^(for|around|in|near|at|over|of|to)\s+', '', q, flags=re.IGNORECASE)
        # Remove temporal suffixes
        q = re.sub(r'\s+(between|from|in|during)\s+(20\d\d)\s+(and|to|vs|versus)\s+(20\d\d).*$', '', q, flags=re.IGNORECASE)
        q = re.sub(r'\s+(in\s+20\d\d|since\s+20\d\d|from\s+20\d\d).*$', '', q, flags=re.IGNORECASE)
        # Fix common typos
        q = re.sub(r'\b(ollage|collage|collge|colleg)\b', 'College', q, flags=re.IGNORECASE)
        q = re.sub(r'\b(univeristy|univercity)\b', 'University', q, flags=re.IGNORECASE)
        q = q.strip(" ?.,!\"'")
        return q or question

    FINE_DETAIL_KEYWORDS = [
        "count", "how many", "this area", "here", "visible", "in view", "on the street",
        "car", "cars", "vehicle", "vehicles", "building", "buildings", "roof", "roofs",
        "boat", "boats", "ship", "ships", "truck", "trucks", "bus", "buses",
        "pedestrian", "pedestrians", "person", "people", "what am i looking at",
        "what is this", "corner", "lane", "parking", "pool", "tree", "trees", "intersection",
        "screen", "on the screen", "what you can see", "what do you see", "what can you see",
        "current view", "what is here", "what are we looking at", "look at this", "this map"
    ]

    @staticmethod
    def interpret(
        question: str,
        location_hint: Optional[str] = None,
        viewport_bbox: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        clean_q = question.strip().lower()
        is_fine_detail = any(kw in clean_q for kw in QueryInterpreter.FINE_DETAIL_KEYWORDS)
        
        # Check if the query asks to navigate to an explicit new named place (e.g. 'show Paris', 'now show me London', 'go to Central Park')
        explicit_nav_keywords = [
            "show me", "show ", "go to", "fly to", "navigate to", "take me to",
            "look at", "search for", "find ", "jump to", "zoom to", "move to",
            "switch to", "head to", "where is", "locate "
        ]
        has_explicit_nav = any(kw in clean_q for kw in explicit_nav_keywords) or any(
            re.search(rf'\b{re.escape(kw)}\b', clean_q) for kw in [
                "tokyo", "paris", "london", "dubai", "mumbai", "delhi", "new york", "central park",
                "manhattan", "brooklyn", "singapore", "sydney", "berlin", "rome", "cairo"
            ]
        )

        # Check if viewport is a global/wide overview
        is_global_viewport = False
        if viewport_bbox and len(viewport_bbox) == 4:
            span_lon = abs(viewport_bbox[2] - viewport_bbox[0])
            span_lat = abs(viewport_bbox[3] - viewport_bbox[1])
            if span_lon > 4.0 or span_lat > 4.0:
                is_global_viewport = True

        has_explicit_loc_hint = bool(
            location_hint and
            len(location_hint.strip()) > 2 and
            "global" not in location_hint.lower() and
            "earth" not in location_hint.lower()
        )

        # Case 1: Active Live Local Map Viewport Navigation (User is looking at or manually panned a specific neighborhood/city)
        if viewport_bbox and len(viewport_bbox) == 4 and not is_global_viewport and not has_explicit_nav and not has_explicit_loc_hint:
            center_lat = (viewport_bbox[1] + viewport_bbox[3]) / 2.0
            center_lon = (viewport_bbox[0] + viewport_bbox[2]) / 2.0

            # Dynamically reverse geocode to identify the exact neighborhood/district the user panned to
            resolved_place_name = GeocoderService.reverse_geocode(center_lat, center_lon) or f"Area at {center_lat:.4f}°N, {center_lon:.4f}°E"

            is_comp = any(w in clean_q for w in [
                "change", "changes", "compare", "difference", "differ", "before vs after", "vs",
                "historical", "reclamation", "expansion", "growth", "shrink", "deforestation"
            ])
            task = "change_analysis" if is_comp else "vqa"
            target = "general"

            year_match = re.search(r'\b(201\d|202[0-5])\b', question)
            baseline_year = year_match.group(1) if year_match else "2020"

            emit_log("INFO", "GEOCODER", f"Locked onto live map viewport ({center_lat:.4f}°N, {center_lon:.4f}°E): '{resolved_place_name}'")

            geocode_info = {
                "name": resolved_place_name,
                "lat": center_lat,
                "lon": center_lon,
                "zoom": 19 if is_fine_detail else 17,
                "bbox": viewport_bbox,
                "display_name": resolved_place_name
            }

            return {
                "task": task,
                "target": target,
                "is_comparison": is_comp,
                "is_fine_detail": is_fine_detail or True,
                "use_viewport_bbox": True,
                "is_new_location_query": False,
                "needs_grounding": True,
                "geocoded_location": geocode_info,
                "corrected_question": question,
                "location_query": resolved_place_name,
                "baseline_year": baseline_year,
                "current_year": "2026",
                "extracted_keywords": [w for w in re.findall(r'\b\w+\b', clean_q) if len(w) > 3]
            }

        # Case 2: Named Place Search / Global Search Query (Explicit Navigation Request)
        llm_norm = QueryInterpreter._llm_normalize(question)

        if llm_norm and llm_norm.get("location_query"):
            corrected_question = llm_norm.get("corrected_question", question)
            loc_search_str = llm_norm.get("location_query", question)
            task = llm_norm.get("task", "vqa")
            target = llm_norm.get("target_entity", "general")
            is_comp = llm_norm.get("is_comparison", "change" in clean_q or "vs" in clean_q or "compare" in clean_q)
            baseline_year = llm_norm.get("baseline_year")
            current_year = llm_norm.get("current_year")
        else:
            # Algorithmic Regex Fallback
            loc_search_str = location_hint.strip() if (location_hint and len(location_hint.strip()) > 2) else QueryInterpreter._extract_location_fallback(question)
            corrected_question = question
            is_comp = any(w in clean_q for w in [
                "change", "changes", "compare", "difference", "differ", "before vs after", "vs",
                "historical", "reclamation", "expansion", "growth", "shrink", "deforestation"
            ])
            task = "change_analysis" if is_comp else "vqa"
            target = "general"
            baseline_year = None
            current_year = None

        # Extract explicit baseline year from question if present (e.g., 2014, 2016, 2018, 2020, 2022, 2024)
        explicit_year_match = re.search(r'\b(201\d|202[0-5])\b', question)
        if explicit_year_match:
            baseline_year = explicit_year_match.group(1)
        elif not baseline_year:
            baseline_year = "2020"

        # Dynamic Geocoding using Normalized Location Query
        emit_log("INFO", "GEOCODER", f"Resolving coordinates for place: '{loc_search_str}'...")
        geocode_info = GeocoderService.geocode(loc_search_str)

        return {
            "task": task,
            "target": target,
            "is_comparison": is_comp,
            "is_fine_detail": is_fine_detail,
            "use_viewport_bbox": False,
            "is_new_location_query": True,
            "needs_grounding": task in ["grounding", "change_analysis"] or any(w in clean_q for w in ["detail", "where", "highlight", "box", "show me", "count"]),
            "geocoded_location": geocode_info,
            "corrected_question": corrected_question,
            "location_query": loc_search_str,
            "baseline_year": baseline_year,
            "current_year": current_year or "2026",
            "extracted_keywords": [w for w in re.findall(r'\b\w+\b', clean_q) if len(w) > 3]
        }
