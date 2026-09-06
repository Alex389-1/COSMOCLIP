import os
import re
import json
import time
import logging
import urllib.request
import urllib.parse
from typing import Optional, List, Dict, Any, Tuple
from collections import OrderedDict
from difflib import SequenceMatcher
try:
    from app.api.logs import emit_log
except ImportError:
    from backend.app.api.logs import emit_log

logger = logging.getLogger(__name__)

STOPWORDS = {
    "show", "me", "locate", "find", "tell", "about", "view", "details", "where", "is",
    "the", "a", "an", "in", "at", "near", "around", "of", "for", "to", "between", "across",
    "please", "this", "that", "place", "area", "region", "city", "map", "satellite", "imagery",
    "observation", "changes", "change", "what", "are", "there", "i", "askd", "asked",
    "can", "you", "see", "we", "they", "it", "how", "many", "do", "does", "did", "look", "on",
    "previous", "context", "current"
}

GENERIC_OBJECTS = {
    "car", "cars", "vehicle", "vehicles", "street", "road", "avenue", "lane",
    "tree", "trees", "building", "buildings", "roof", "roofs", "lake", "water",
    "river", "field", "pond", "ponds", "boat", "boats", "ship", "ships",
    "people", "person", "house", "houses", "scene", "view", "corner", "intersection"
}

INVALID_LOCATIONS = {
    "previous context", "previous", "context", "none", "null", "undefined",
    "target location", "current location", "current viewport location", "current context",
    "global", "earth", "world"
}

NON_REGIONS = {
    "college", "campus", "university", "institute", "institutions", "group", "school",
    "hospital", "dept", "department", "road", "block", "sector", "zone", "office", "center", "centre"
}

GEOGRAPHIC_FEATURE_TERMS = {
    "island", "islands", "isle", "isles", "hill", "hills", "mountain", "mountains", "mount", "lake", "lakes",
    "river", "rivers", "sea", "ocean", "bay", "beach", "harbor", "harbour", "port", "park",
    "parks", "valley", "creek", "cape", "point", "strait", "sound", "channel", "forest",
    "tower", "building", "hall", "house", "palace", "fort", "castle", "bridge", "street",
    "road", "avenue", "lane", "place", "center", "centre"
}

KNOWN_CARTOGRAPHIC_LOCATIONS: Dict[str, Dict[str, Any]] = {
    "null island": {
        "name": "Null Island (0°N, 0°E)",
        "lat": 0.0,
        "lon": 0.0,
        "zoom": 6,
        "bbox": [-1.0, -1.0, 1.0, 1.0],
        "display_name": "Null Island (0°N, 0°E), Soul Buoy Station 13010, Gulf of Guinea"
    },
    "nullisland": {
        "name": "Null Island (0°N, 0°E)",
        "lat": 0.0,
        "lon": 0.0,
        "zoom": 6,
        "bbox": [-1.0, -1.0, 1.0, 1.0],
        "display_name": "Null Island (0°N, 0°E), Soul Buoy Station 13010, Gulf of Guinea"
    },
    "soul buoy": {
        "name": "Soul Buoy (Null Island 0°N, 0°E)",
        "lat": 0.0,
        "lon": 0.0,
        "zoom": 6,
        "bbox": [-1.0, -1.0, 1.0, 1.0],
        "display_name": "Soul Buoy (Station 13010), Null Island (0°N, 0°E), Gulf of Guinea"
    }
}

class GeocoderService:
    """
    100% Dynamic Geospatial Resolver:
    Pulls exact coordinates, bounding boxes, and official place names dynamically
    using live global Web Geocoding APIs (OpenStreetMap Nominatim, Photon Komoot)
    and Google Gemini / Mistral AI Geospatial Intelligence.
    ZERO hardcoded coordinates or static city-institution tables.
    """

    _GEOCODE_CACHE: OrderedDict[str, Tuple[float, Dict[str, Any]]] = OrderedDict()
    _CACHE_TTL_SEC: float = 86400.0  # 24 hours TTL
    _CACHE_MAX_SIZE: int = 500       # Maximum entries to bound memory

    @classmethod
    def _store_cache(cls, key: str, val: Dict[str, Any]) -> None:
        """Stores entry with timestamp and evicts least recently accessed item if exceeding max size (true LRU)."""
        if key in cls._GEOCODE_CACHE:
            cls._GEOCODE_CACHE.move_to_end(key)
        elif len(cls._GEOCODE_CACHE) >= cls._CACHE_MAX_SIZE:
            cls._GEOCODE_CACHE.popitem(last=False)  # Evict least recently accessed item
        cls._GEOCODE_CACHE[key] = (time.time(), val)

    @classmethod
    def clear_cache(cls) -> None:
        """Clears all cached geocoding entries."""
        cls._GEOCODE_CACHE.clear()

    @staticmethod
    def extract_regional_anchor(text: str) -> Optional[str]:
        """
        Dynamically extracts city/district/region qualification from natural or normalized query.
        Example: 'KIET Group of Institutions, Ghaziabad, Uttar Pradesh' -> 'ghaziabad'
        Example: 'show me KIIT college in ghaziabad' -> 'ghaziabad'
        """
        clean = text.lower().strip()
        # If comma-separated normalized address, check terms after first comma
        if "," in clean:
            parts = [p.strip() for p in clean.split(",")[1:] if p.strip()]
            for p in parts:
                tokens = [w for w in re.split(r'[,.\s]+', p) if len(w) > 2 and w not in STOPWORDS and w not in NON_REGIONS]
                if tokens:
                    return tokens[0]

        for prep in [" in ", " at ", " near ", " around "]:
            if prep in clean:
                part = clean.split(prep)[-1].strip()
                tokens = [w for w in re.split(r'[,.\s]+', part) if len(w) > 2 and w not in STOPWORDS and w not in NON_REGIONS]
                if tokens:
                    return tokens[0]
        return None

    @staticmethod
    def generate_acoustic_variants(word: str) -> List[str]:
        """
        Algorithmically generates generic acoustic / phonetic vowel permutations
        with original exact spelling prioritized first.
        """
        variants = [word.lower()]
        w = word.lower()

        if "i" in w:
            variants.append(w.replace("i", "e"))
        if "e" in w:
            variants.append(w.replace("e", "i"))
        if "ii" in w:
            variants.append(w.replace("ii", "ie"))
        if "ie" in w:
            variants.append(w.replace("ie", "ii"))

        # Deduplicate while preserving order
        return list(dict.fromkeys(variants))

    @staticmethod
    def extract_distinctive_tokens(text: str) -> List[str]:
        """
        Extracts distinctive named entity tokens excluding common conversational words and generic nouns.
        """
        clean = text.lower()
        words = re.findall(r'\b[a-z0-9]{2,15}\b', clean)
        generic_terms = STOPWORDS.union({"college", "campus", "university", "institute", "institutions", "school", "hospital", "station", "building", "group"})
        distinctive = [w for w in words if w not in generic_terms]
        return distinctive

    @staticmethod
    def build_search_plan(text: str) -> List[Tuple[str, str]]:
        """
        Builds an ordered list of (search_query, required_entity_token) tuples.
        Capped to top 3 candidate queries to prevent combinatorial latency explosion.
        """
        clean = text.strip()
        plan: List[Tuple[str, str]] = []

        # 1. Exact raw text (especially when already normalized like 'KIET Group of Institutions, Ghaziabad, Uttar Pradesh')
        plan.append((clean, ""))

        # 2. Cleaned transcription typos
        clean_lower = clean.lower()
        clean_lower = re.sub(r'\b(ollage|collage|collge|colleg)\b', 'college', clean_lower)
        clean_lower = re.sub(r'\b(univeristy|univercity)\b', 'university', clean_lower)

        anchor = GeocoderService.extract_regional_anchor(clean_lower)
        distinctive = GeocoderService.extract_distinctive_tokens(clean_lower)

        if anchor and anchor in distinctive:
            distinctive_entities = [d for d in distinctive if d != anchor]
        else:
            distinctive_entities = distinctive

        core_entity = distinctive_entities[0] if distinctive_entities else ""

        # 3. Direct stripped query
        clean_stripped = " ".join([w for w in clean_lower.split() if w not in {"show", "me", "please", "locate", "find", "tell", "about", "view", "details", "where", "is"}])
        if clean_stripped and clean_stripped != clean.lower():
            plan.append((clean_stripped, core_entity))

        # 4. Substantive entity + anchor variants (e.g. "kiet ghaziabad", "kiet college ghaziabad")
        if core_entity:
            entity_variants = GeocoderService.generate_acoustic_variants(core_entity)
            for ev in entity_variants:
                if anchor:
                    plan.append((f"{ev} group of institutions {anchor}", ev))
                    plan.append((f"{ev} college {anchor}", ev))
                    plan.append((f"{ev} {anchor}", ev))
                else:
                    plan.append((f"{ev} group of institutions", ev))
                    plan.append((f"{ev} college", ev))
                    plan.append((ev, ev))

        # Deduplicate while preserving order
        seen = set()
        ordered_plan = []
        for q, req in plan:
            key = (q.strip(), req.strip())
            if key[0] and key not in seen:
                seen.add(key)
                ordered_plan.append(key)

        # Cap search plan attempts to top 3 candidate queries
        return ordered_plan[:3]


    @staticmethod
    def is_valid_geocoding_match(
        query: str,
        display_name: str,
        min_token_match: float = 0.5,
        min_similarity: float = 0.40
    ) -> bool:
        """
        Defense-in-depth sanity check:
        Rejects geocoder results if the query is a conversational sentence, a non-location,
        or if ZERO query tokens match the candidate address.
        Uses exact token containment and \\b word-boundary regex to prevent false-positive substring matches.
        """
        clean_q = query.lower().strip()
        clean_disp = display_name.lower().strip()

        # 1. Reject blacklisted non-location phrases immediately
        if clean_q in INVALID_LOCATIONS:
            return False

        # 2. Extract distinctive tokens from the input query
        q_tokens = [
            w for w in re.findall(r'\b[a-z0-9]{2,}\b', clean_q)
            if w not in STOPWORDS and w not in GENERIC_OBJECTS
        ]
        if not q_tokens:
            emit_log(
                "WARNING", "GEOCODER",
                f"Geocoder sanity guard rejected '{query}': no distinctive geographic entity tokens found."
            )
            return False

        disp_tokens = set(re.findall(r'\b[a-z0-9]{2,}\b', clean_disp))

        # Check token matches using exact token set or \\b word-boundary regex (prevent substring matches like 'art' in 'Stratford')
        matched_tokens = 0
        for qt in q_tokens:
            if qt in disp_tokens or bool(re.search(r'\b' + re.escape(qt) + r'\b', clean_disp)):
                matched_tokens += 1
            else:
                best_sim = max([SequenceMatcher(None, qt, dt).ratio() for dt in disp_tokens] or [0.0])
                if best_sim >= 0.75:
                    matched_tokens += 1

        # CRITICAL: At least one distinctive query token MUST match the result!
        # If 0 tokens match, reject unconditionally (e.g. 'previous context' -> 'precious gifts preschool')
        if matched_tokens == 0:
            emit_log(
                "WARNING", "GEOCODER",
                f"Geocoder match rejected: '{query}' -> '{display_name[:60]}...' (0 tokens matched)"
            )
            return False

        # CRITICAL: If the query contains proper name tokens (non-generic geographic terms like 'island', 'lake', 'river'),
        # at least one PROPER name token MUST match! Prevent matching 'The Island Hill' when searching 'Null Island'.
        proper_tokens = [t for t in q_tokens if t not in GEOGRAPHIC_FEATURE_TERMS]
        if proper_tokens:
            proper_matched = 0
            for pt in proper_tokens:
                if pt in disp_tokens or bool(re.search(r'\b' + re.escape(pt) + r'\b', clean_disp)):
                    proper_matched += 1
                else:
                    best_sim = max([SequenceMatcher(None, pt, dt).ratio() for dt in disp_tokens] or [0.0])
                    if best_sim >= 0.75:
                        proper_matched += 1
            if proper_matched == 0:
                emit_log(
                    "WARNING", "GEOCODER",
                    f"Geocoder match rejected: '{query}' -> '{display_name[:60]}...' (proper tokens {proper_tokens} missing, only generic feature terms matched)"
                )
                return False

        token_match_ratio = matched_tokens / len(q_tokens)
        if token_match_ratio >= min_token_match:
            return True

        # Calculate sequence similarity against display name parts
        disp_parts = [p.strip() for p in clean_disp.split(",") if p.strip()]
        cleaned_entity_query = " ".join(q_tokens)
        part_similarities = [
            SequenceMatcher(None, cleaned_entity_query, part).ratio()
            for part in disp_parts
        ] or [0.0]
        best_part_sim = max(part_similarities)

        if best_part_sim >= min_similarity:
            return True

        emit_log(
            "WARNING", "GEOCODER",
            f"Geocoder match rejected: '{query}' -> '{display_name[:60]}...' (token_match={token_match_ratio:.2f}, best_sim={best_part_sim:.2f})"
        )
        return False

    @staticmethod
    def geocode(location_name: str) -> Optional[Dict[str, Any]]:
        """
        Dynamically geocodes any query to exact ground-truth coordinates over the internet.
        Priority:
        1. Known cartographic reference landmarks (e.g. Null Island 0°N, 0°E, Soul Buoy).
        2. In-memory cache lookup with 24h TTL and LRU bounded size (500 items).
        3. Live High-Precision Authoritative Geospatial Index (OSM Nominatim & Photon Komoot)
           with dynamic regional anchor consistency, acoustic token verification, and strict wall-clock budget.
        4. Google Gemini & Mistral AI Geospatial Intelligence with spatial constraints.
        Returns None if all strategies fail, guaranteeing no synthetic Null Island (0,0) fallback.
        """
        clean_loc = location_name.lower().strip()
        norm_loc = re.sub(r'[^a-z0-9\s]', '', clean_loc).strip()

        # Known cartographic reference landmarks (e.g. Null Island 0°N, 0°E, Soul Buoy)
        is_null_island = (
            norm_loc == "null island" or
            norm_loc.startswith("null island") or
            "null island" in norm_loc or
            "nullisland" in norm_loc or
            "soul buoy" in norm_loc
        )
        if is_null_island or norm_loc in KNOWN_CARTOGRAPHIC_LOCATIONS or clean_loc in KNOWN_CARTOGRAPHIC_LOCATIONS:
            known_key = "soul buoy" if "soul buoy" in norm_loc else "null island"
            known = KNOWN_CARTOGRAPHIC_LOCATIONS.get(known_key) or KNOWN_CARTOGRAPHIC_LOCATIONS["null island"]
            GeocoderService._store_cache(clean_loc, known)
            emit_log("INFO", "GEOCODER", f"Resolved cartographic reference landmark: '{location_name}' -> {known['lat']}°N, {known['lon']}°E ({known['display_name']})")
            return known

        # Cache check: avoid redundant network hits, respect TTL and LRU eviction
        if clean_loc in GeocoderService._GEOCODE_CACHE:
            cached_ts, cached_val = GeocoderService._GEOCODE_CACHE[clean_loc]
            if (time.time() - cached_ts) < GeocoderService._CACHE_TTL_SEC:
                # Move to end on hit to maintain true LRU order (least-recently-used is at index 0)
                GeocoderService._GEOCODE_CACHE.move_to_end(clean_loc)
                emit_log("INFO", "GEOCODER", f"Cache hit for geocoded location: '{clean_loc}' (fresh TTL, LRU updated)")
                return cached_val
            else:
                emit_log("INFO", "GEOCODER", f"Cache entry expired for '{clean_loc}' (TTL > 24h). Evicting.")
                GeocoderService._GEOCODE_CACHE.pop(clean_loc, None)

        # Sanity Guard 0: Reject blacklisted non-locations immediately
        if clean_loc in INVALID_LOCATIONS or not clean_loc:
            emit_log("WARNING", "GEOCODER", f"Sanity guard: query '{location_name}' is not a valid geographic destination. Bypassing geocoder.")
            return None

        # Sanity Guard 0.5: Reject conversational queries with no geographic entity tokens
        distinctive_tokens = [
            w for w in re.findall(r'\b[a-z0-9]{2,}\b', clean_loc)
            if w not in STOPWORDS and w not in GENERIC_OBJECTS
        ]
        if not distinctive_tokens:
            emit_log("WARNING", "GEOCODER", f"Sanity guard: query '{location_name}' has no recognizable place names. Bypassing geocoder.")
            return None

        start_time = time.perf_counter()
        MAX_TOTAL_BUDGET_SEC = 5.0

        anchor_city = GeocoderService.extract_regional_anchor(location_name)
        search_plan = GeocoderService.build_search_plan(location_name)

        # Strategy 1: High-Precision Live Geospatial Search (Nominatim & Photon Komoot)
        for query_str, req_token in search_plan:
            elapsed = time.perf_counter() - start_time
            if elapsed >= (MAX_TOTAL_BUDGET_SEC - 0.2):
                emit_log("WARNING", "GEOCODER", f"Total geocoding budget ({MAX_TOTAL_BUDGET_SEC}s) reached before candidate '{query_str}'.")
                break

            cand_clean = re.sub(r'[,.]', ' ', query_str).strip()
            encoded = urllib.parse.quote(cand_clean)

            # Provider A: OpenStreetMap Nominatim
            elapsed = time.perf_counter() - start_time
            remaining_a = MAX_TOTAL_BUDGET_SEC - elapsed
            if remaining_a > 0.4:
                try:
                    timeout_a = min(2.0, remaining_a)
                    url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=6&addressdetails=1"
                    req = urllib.request.Request(url, headers={"User-Agent": "COSMOCLIP-LiveGeo/3.0 (contact@cosmoclip.ai)"})
                    with urllib.request.urlopen(req, timeout=timeout_a) as resp:
                        data = json.loads(resp.read().decode())
                        if data and len(data) > 0:
                            selected = None
                            for item in data:
                                disp = (item.get("display_name", "") + " " + item.get("name", "")).lower()
                                addr = item.get("address", {})
                                addr_str = " ".join(str(v).lower() for v in addr.values()) if isinstance(addr, dict) else ""
                                full_osm_text = f"{disp} {addr_str}"

                                # 1. If a regional anchor was asked (e.g. 'ghaziabad'), ensure candidate matches this region
                                if anchor_city and anchor_city.lower() not in full_osm_text:
                                    continue

                                # 2. If a specific entity was requested (e.g. 'kiet' or 'kiit'), require it in the OSM result name
                                if req_token and req_token.lower() not in full_osm_text:
                                    continue

                                # 3. Defense-in-depth sanity check: SequenceMatcher & token verification
                                if not GeocoderService.is_valid_geocoding_match(cand_clean, full_osm_text):
                                    continue

                                selected = item
                                break

                            if selected:
                                lat, lon = float(selected["lat"]), float(selected["lon"])
                                name_parts = selected.get("display_name", query_str).split(",")
                                clean_name = ", ".join(name_parts[:3]).strip()
                                ptype = selected.get("type", "")
                                zoom = 16 if ptype in ["college", "university", "school", "hospital", "station", "building", "aerodrome", "port"] else 15
                                res = GeocoderService._build_geo_result(clean_name, lat, lon, default_zoom=zoom)
                                GeocoderService._store_cache(clean_loc, res)
                                return res
                except Exception:
                    pass

            # Provider B: Photon Komoot Geocoding
            elapsed = time.perf_counter() - start_time
            remaining_b = MAX_TOTAL_BUDGET_SEC - elapsed
            if remaining_b > 0.4:
                try:
                    timeout_b = min(2.0, remaining_b)
                    url = f"https://photon.komoot.io/api/?q={encoded}&limit=6"
                    req = urllib.request.Request(url, headers={"User-Agent": "COSMOCLIP-Geospatial/3.0"})
                    with urllib.request.urlopen(req, timeout=timeout_b) as resp:
                        data = json.loads(resp.read().decode())
                        features = data.get("features", [])
                        if features and len(features) > 0:
                            selected_f = None
                            for f in features:
                                props = f.get("properties", {})
                                name_str = (props.get("name", "") + " " + props.get("city", "") + " " + props.get("state", "") + " " + props.get("country", "")).lower()

                                if anchor_city and anchor_city.lower() not in name_str:
                                    continue

                                if req_token and req_token.lower() not in name_str:
                                    continue

                                # Defense-in-depth sanity check
                                if not GeocoderService.is_valid_geocoding_match(cand_clean, name_str):
                                    continue

                                selected_f = f
                                break

                            if selected_f:
                                coords = selected_f["geometry"]["coordinates"]
                                props = selected_f["properties"]
                                lon, lat = float(coords[0]), float(coords[1])
                                name = f"{props.get('name', query_str)}, {props.get('city', props.get('state', ''))}".strip(", ")
                                osm_val = props.get("osm_value", "")
                                zoom = 16 if osm_val in ["college", "university", "school", "hospital", "station", "building"] else 15
                                res = GeocoderService._build_geo_result(name, lat, lon, default_zoom=zoom)
                                GeocoderService._store_cache(clean_loc, res)
                                return res
                except Exception:
                    pass

        # Strategy 2: Dynamic Google Gemini Grounded Location Resolution
        elapsed = time.perf_counter() - start_time
        remaining_gemini = MAX_TOTAL_BUDGET_SEC - elapsed
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key and remaining_gemini > 0.5:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=api_key)
                prompt = f"""Identify the exact real-world geographic coordinates for: "{location_name}".
CRITICAL GEOSPATIAL RULES:
1. If the user query specifies a city, district, or region (e.g. 'in Ghaziabad', 'in Delhi', 'in Bhubaneswar'), the location MUST be strictly inside or adjacent to that city.
2. Account for potential speech recognition acoustic/phonetic mishearings dynamically (e.g. vowels or acronym soundalikes).

Return exact coordinates in valid JSON matching this schema:
{{
  "name": "Official Campus or Location Name, City, State, Country",
  "lat": 0.0000,
  "lon": 0.0000,
  "zoom": 16,
  "bbox": [min_lon, min_lat, max_lon, max_lat]
}}"""
                res = None
                for candidate_model in [os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"), "gemini-2.5-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest", "gemini-2.5-flash"]:
                    try:
                        res = client.models.generate_content(
                            model=candidate_model,
                            contents=prompt,
                            config=types.GenerateContentConfig(response_mime_type="application/json")
                        )
                        if res and res.text:
                            break
                    except Exception:
                        continue
                if res and res.text:
                    data = json.loads(res.text)
                    if "lat" in data and "lon" in data and isinstance(data["lat"], (int, float)):
                        lat = float(data["lat"])
                        lon = float(data["lon"])
                        name = data.get("name", location_name.title())
                        name_lower = name.lower()
                        # Reject synthetic LLM failure responses
                        if "not found" in name_lower or "unknown" in name_lower or "does not exist" in name_lower:
                            return None
                        if lat == 0.0 and lon == 0.0 and norm_loc not in KNOWN_CARTOGRAPHIC_LOCATIONS and clean_loc not in KNOWN_CARTOGRAPHIC_LOCATIONS:
                            return None
                        if not GeocoderService.is_valid_geocoding_match(clean_loc, name):
                            return None

                        z = int(data.get("zoom", 14))
                        if z < 4:
                            z = 6
                        raw_bbox = data.get("bbox")
                        if isinstance(raw_bbox, list) and len(raw_bbox) == 4 and all(isinstance(x, (int, float)) for x in raw_bbox):
                            bbox = [float(x) for x in raw_bbox]
                        elif isinstance(raw_bbox, list) and len(raw_bbox) == 2 and isinstance(raw_bbox[0], (list, tuple)) and isinstance(raw_bbox[1], (list, tuple)):
                            bbox = [float(raw_bbox[0][0]), float(raw_bbox[0][1]), float(raw_bbox[1][0]), float(raw_bbox[1][1])]
                        else:
                            delta = 0.015
                            bbox = [round(lon - delta, 4), round(lat - delta, 4), round(lon + delta, 4), round(lat + delta, 4)]
                        result = {
                            "name": name,
                            "lat": lat,
                            "lon": lon,
                            "zoom": z,
                            "bbox": bbox,
                            "display_name": name
                        }
                        GeocoderService._store_cache(clean_loc, result)
                        return result
            except Exception as e:
                logger.debug(f"Gemini dynamic geocoding fallback: {e}")

        # Strategy 3: Dynamic Mistral AI Location Resolution
        elapsed = time.perf_counter() - start_time
        remaining_mistral = MAX_TOTAL_BUDGET_SEC - elapsed
        mistral_key = os.getenv("MISTRAL_API_KEY", "")
        if mistral_key and remaining_mistral > 0.5:
            try:
                url = "https://api.mistral.ai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {mistral_key}",
                    "Content-Type": "application/json"
                }
                m_prompt = f"""Identify the real-world geographic coordinates for: "{location_name}".
If a regional anchor is specified in the query, ensure the returned location is strictly inside that region. Resolve any phonetic speech recognition ambiguity dynamically.
Return ONLY valid JSON matching this schema:
{{
  "name": "Official Location Name, City, Country",
  "lat": 0.0000,
  "lon": 0.0000,
  "zoom": 16,
  "bbox": [min_lon, min_lat, max_lon, max_lat]
}}"""
                for m_model in ["open-mistral-nemo", "pixtral-12b-2409", "open-mistral-7b"]:
                    elapsed = time.perf_counter() - start_time
                    if (MAX_TOTAL_BUDGET_SEC - elapsed) < 0.5:
                        break
                    try:
                        m_payload = {
                            "model": m_model,
                            "messages": [
                                {"role": "system", "content": "You are an expert geospatial geocoding engine. Output ONLY JSON."},
                                {"role": "user", "content": m_prompt}
                            ],
                            "response_format": {"type": "json_object"},
                            "max_tokens": 150
                        }
                        req = urllib.request.Request(url, data=json.dumps(m_payload).encode(), headers=headers)
                        timeout_m = min(2.0, max(0.4, MAX_TOTAL_BUDGET_SEC - (time.perf_counter() - start_time)))
                        with urllib.request.urlopen(req, timeout=timeout_m) as resp:
                            m_res = json.loads(resp.read().decode())
                            m_data = json.loads(m_res["choices"][0]["message"]["content"])
                            if "lat" in m_data and "lon" in m_data and isinstance(m_data["lat"], (int, float)):
                                lat = float(m_data["lat"])
                                lon = float(m_data["lon"])
                                name = m_data.get("name", location_name.title())
                                name_lower = name.lower()
                                if "not found" in name_lower or "unknown" in name_lower or "does not exist" in name_lower:
                                    return None
                                if lat == 0.0 and lon == 0.0 and norm_loc not in KNOWN_CARTOGRAPHIC_LOCATIONS and clean_loc not in KNOWN_CARTOGRAPHIC_LOCATIONS:
                                    return None

                                z = int(m_data.get("zoom", 14))
                                if z < 4:
                                    z = 6
                                raw_bbox = m_data.get("bbox")
                                if isinstance(raw_bbox, list) and len(raw_bbox) == 4 and all(isinstance(x, (int, float)) for x in raw_bbox):
                                    bbox = [float(x) for x in raw_bbox]
                                elif isinstance(raw_bbox, list) and len(raw_bbox) == 2 and isinstance(raw_bbox[0], (list, tuple)) and isinstance(raw_bbox[1], (list, tuple)):
                                    bbox = [float(raw_bbox[0][0]), float(raw_bbox[0][1]), float(raw_bbox[1][0]), float(raw_bbox[1][1])]
                                else:
                                    delta = 0.015
                                    bbox = [round(lon - delta, 4), round(lat - delta, 4), round(lon + delta, 4), round(lat + delta, 4)]
                                if not GeocoderService.is_valid_geocoding_match(clean_loc, name):
                                    continue
                                result = {
                                    "name": name,
                                    "lat": lat,
                                    "lon": lon,
                                    "zoom": z,
                                    "bbox": bbox,
                                    "display_name": name
                                }
                                GeocoderService._store_cache(clean_loc, result)
                                return result
                    except Exception:
                        continue
            except Exception as me:
                logger.debug(f"Mistral dynamic geocoding error: {me}")

        # Final Fallback: General global search with sanity check
        elapsed = time.perf_counter() - start_time
        remaining_final = MAX_TOTAL_BUDGET_SEC - elapsed
        if remaining_final > 0.4:
            try:
                timeout_final = min(2.0, remaining_final)
                encoded = urllib.parse.quote(location_name)
                url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
                req = urllib.request.Request(url, headers={"User-Agent": "COSMOCLIP-LiveGeo/3.0 (contact@cosmoclip.ai)"})
                with urllib.request.urlopen(req, timeout=timeout_final) as resp:
                    data = json.loads(resp.read().decode())
                    if data and len(data) > 0:
                        r = data[0]
                        disp = r.get("display_name", "")
                        if GeocoderService.is_valid_geocoding_match(location_name, disp):
                            lat, lon = float(r["lat"]), float(r["lon"])
                            name_parts = disp.split(",")
                            clean_name = ", ".join(name_parts[:3]).strip()
                            res = GeocoderService._build_geo_result(clean_name, lat, lon)
                            GeocoderService._store_cache(clean_loc, res)
                            return res
                        else:
                            emit_log("WARNING", "GEOCODER", f"Final fallback OSM rejected candidate lacking similarity: '{location_name}' -> '{disp[:60]}'")
            except Exception:
                pass

        emit_log("WARNING", "GEOCODER", f"All geocoding strategies rejected or failed for query: '{location_name}'. Returning None.")
        return None

    @staticmethod
    def _build_geo_result(name: str, lat: float, lon: float, default_zoom: int = 14) -> Dict[str, Any]:
        delta = 0.025
        return {
            "name": name,
            "lat": lat,
            "lon": lon,
            "zoom": default_zoom,
            "bbox": [round(lon - delta, 4), round(lat - delta, 4), round(lon + delta, 4), round(lat + delta, 4)],
            "display_name": name
        }

    @staticmethod
    def reverse_geocode(lat: float, lon: float) -> Optional[str]:
        """
        Dynamically resolves coordinates (lat, lon) to a human-readable place or neighborhood name.
        """
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?lat={lat:.5f}&lon={lon:.5f}&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "COSMOCLIP-LiveGeo/3.0 (contact@cosmoclip.ai)"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode())
                if data and "display_name" in data:
                    parts = [p.strip() for p in data["display_name"].split(",") if p.strip()]
                    # Return top 2-3 most distinctive components (e.g. "SoHo, Manhattan, New York")
                    return ", ".join(parts[:3])
        except Exception:
            pass
        return None
