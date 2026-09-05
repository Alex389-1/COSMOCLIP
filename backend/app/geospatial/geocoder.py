import os
import re
import json
import logging
import urllib.request
import urllib.parse
try:
    from app.api.logs import emit_log
except ImportError:
    from backend.app.api.logs import emit_log

logger = logging.getLogger(__name__)

STOPWORDS = {
    "show", "me", "locate", "find", "tell", "about", "view", "details", "where", "is",
    "the", "a", "an", "in", "at", "near", "around", "of", "for", "to", "between", "across",
    "please", "this", "that", "place", "area", "region", "city", "map", "satellite", "imagery",
    "observation", "changes", "change", "what", "are", "there", "i", "askd", "asked"
}

NON_REGIONS = {
    "college", "campus", "university", "institute", "institutions", "group", "school",
    "hospital", "dept", "department", "road", "block", "sector", "zone", "office", "center", "centre"
}

class GeocoderService:
    """
    100% Dynamic Geospatial Resolver:
    Pulls exact coordinates, bounding boxes, and official place names dynamically
    using live global Web Geocoding APIs (OpenStreetMap Nominatim, Photon Komoot)
    and Google Gemini / Mistral AI Geospatial Intelligence.
    ZERO hardcoded coordinates or static city-institution tables.
    """

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

        return ordered_plan


    @staticmethod
    def geocode(location_name: str) -> Dict[str, Any]:
        """
        Dynamically geocodes any query to exact ground-truth coordinates over the internet.
        Priority:
        1. Live High-Precision Authoritative Geospatial Index (OSM Nominatim & Photon Komoot)
           with dynamic regional anchor consistency and acoustic token verification.
        2. Google Gemini & Mistral AI Geospatial Intelligence with spatial constraints.
        """
        anchor_city = GeocoderService.extract_regional_anchor(location_name)
        search_plan = GeocoderService.build_search_plan(location_name)

        # Strategy 1: High-Precision Live Geospatial Search (Nominatim & Photon Komoot)
        for query_str, req_token in search_plan:
            cand_clean = re.sub(r'[,.]', ' ', query_str).strip()
            encoded = urllib.parse.quote(cand_clean)

            # Provider A: OpenStreetMap Nominatim
            try:
                url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=6&addressdetails=1"
                req = urllib.request.Request(url, headers={"User-Agent": "COSMOCLIP-LiveGeo/3.0 (contact@cosmoclip.ai)"})
                with urllib.request.urlopen(req, timeout=3.5) as resp:
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

                            selected = item
                            break

                        if selected:
                            lat, lon = float(selected["lat"]), float(selected["lon"])
                            name_parts = selected.get("display_name", query_str).split(",")
                            clean_name = ", ".join(name_parts[:3]).strip()
                            ptype = selected.get("type", "")
                            zoom = 16 if ptype in ["college", "university", "school", "hospital", "station", "building", "aerodrome", "port"] else 15
                            return GeocoderService._build_geo_result(clean_name, lat, lon, default_zoom=zoom)
            except Exception:
                pass

            # Provider B: Photon Komoot Geocoding
            try:
                url = f"https://photon.komoot.io/api/?q={encoded}&limit=6"
                req = urllib.request.Request(url, headers={"User-Agent": "COSMOCLIP-Geospatial/3.0"})
                with urllib.request.urlopen(req, timeout=3.0) as resp:
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

                            selected_f = f
                            break

                        if selected_f:
                            coords = selected_f["geometry"]["coordinates"]
                            props = selected_f["properties"]
                            lon, lat = float(coords[0]), float(coords[1])
                            name = f"{props.get('name', query_str)}, {props.get('city', props.get('state', ''))}".strip(", ")
                            osm_val = props.get("osm_value", "")
                            zoom = 16 if osm_val in ["college", "university", "school", "hospital", "station", "building"] else 15
                            return GeocoderService._build_geo_result(name, lat, lon, default_zoom=zoom)
            except Exception:
                pass

        # Strategy 2: Dynamic Google Gemini Grounded Location Resolution
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
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
                res = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                if res and res.text:
                    data = json.loads(res.text)
                    if "lat" in data and "lon" in data and isinstance(data["lat"], (int, float)):
                        lat = float(data["lat"])
                        lon = float(data["lon"])
                        name = data.get("name", location_name.title())
                        z = data.get("zoom", 16)
                        delta = 0.015
                        bbox = data.get("bbox") or [round(lon - delta, 4), round(lat - delta, 4), round(lon + delta, 4), round(lat + delta, 4)]
                        return {
                            "name": name,
                            "lat": lat,
                            "lon": lon,
                            "zoom": z,
                            "bbox": bbox,
                            "display_name": name
                        }
            except Exception as e:
                logger.debug(f"Gemini dynamic geocoding fallback: {e}")

        # Strategy 3: Dynamic Mistral AI Location Resolution
        mistral_key = os.getenv("MISTRAL_API_KEY", "")
        if mistral_key:
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
                        with urllib.request.urlopen(req, timeout=4.0) as resp:
                            m_res = json.loads(resp.read().decode())
                            m_data = json.loads(m_res["choices"][0]["message"]["content"])
                            if "lat" in m_data and "lon" in m_data and isinstance(m_data["lat"], (int, float)):
                                lat = float(m_data["lat"])
                                lon = float(m_data["lon"])
                                name = m_data.get("name", location_name.title())
                                z = m_data.get("zoom", 16)
                                delta = 0.015
                                bbox = m_data.get("bbox") or [round(lon - delta, 4), round(lat - delta, 4), round(lon + delta, 4), round(lat + delta, 4)]
                                return {
                                    "name": name,
                                    "lat": lat,
                                    "lon": lon,
                                    "zoom": z,
                                    "bbox": bbox,
                                    "display_name": name
                                }
                    except Exception:
                        continue
            except Exception as me:
                logger.debug(f"Mistral dynamic geocoding error: {me}")

        # Final Fallback: General global search
        try:
            encoded = urllib.parse.quote(location_name)
            url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "COSMOCLIP-LiveGeo/3.0 (contact@cosmoclip.ai)"})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode())
                if data and len(data) > 0:
                    r = data[0]
                    lat, lon = float(r["lat"]), float(r["lon"])
                    name_parts = r.get("display_name", location_name).split(",")
                    clean_name = ", ".join(name_parts[:3]).strip()
                    return GeocoderService._build_geo_result(clean_name, lat, lon)
        except Exception:
            pass

        return GeocoderService._build_geo_result(location_name.title(), 0.0, 0.0, default_zoom=3)

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
