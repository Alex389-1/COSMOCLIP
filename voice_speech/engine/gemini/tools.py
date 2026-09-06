"""Gemini Live Function Calling Tools & Registry.

Provides zero-key real-time news retrieval (Google News RSS + NewsAPI fallback)
and an extensible dispatcher registry for tool calls.
"""

import asyncio
import json
import logging
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Awaitable, Callable, Dict, List, Optional
import re
from google.genai import types


logger = logging.getLogger("riva.tools")


async def fetch_news_summary(query: str) -> str:
    """Fetches a 3-headline news summary using NewsAPI (if key provided) or Google News RSS (zero key)."""
    clean_query = query.strip()
    if not clean_query:
        clean_query = "top world news"

    news_api_key = os.getenv("NEWS_API_KEY", "").strip()
    loop = asyncio.get_running_loop()

    # 1. Optional NewsAPI.org query (if key provided)
    if news_api_key:
        try:
            encoded = urllib.parse.quote(clean_query)
            url = f"https://newsapi.org/v2/everything?q={encoded}&pageSize=3&sortBy=publishedAt&apiKey={news_api_key}"
            req = urllib.request.Request(url, headers={"User-Agent": "RivaVoice/1.0"})

            def _fetch_newsapi():
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    return resp.read()

            raw_json = await loop.run_in_executor(None, _fetch_newsapi)
            data = json.loads(raw_json)
            articles = data.get("articles", [])
            headlines = [a.get("title", "").strip() for a in articles if a.get("title")]
            if headlines:
                summary = " | ".join(headlines[:3])[:320]
                logger.info(f"Live NewsAPI response for '{clean_query}': {summary!r}")
                return summary
        except Exception as e:
            logger.warning(f"NewsAPI error (falling back to Google News RSS): {e}")

    # 2. Universal Zero-Key Fallback: Google News RSS
    try:
        encoded = urllib.parse.quote(clean_query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

        def _fetch_rss():
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                return resp.read()

        xml_data = await loop.run_in_executor(None, _fetch_rss)
        root = ET.fromstring(xml_data)
        items = root.findall(".//item")

        headlines = []
        for item in items[:3]:
            title = item.find("title")
            if title is not None and title.text:
                clean_title = title.text.split(" - ")[0] if " - " in title.text else title.text
                headlines.append(clean_title)

        if headlines:
            summary = " | ".join(headlines)[:320]
            logger.info(f"Live News RSS response for '{clean_query}': {summary!r}")
            return summary

        return f"No recent breaking news found for '{clean_query}'."
    except Exception as e:
        logger.warning(f"News RSS fetch error for '{clean_query}': {e}")
        return f"Could not retrieve recent news for '{clean_query}'."


# Tool Declarations
NEWS_TOOL_DECLARATION = types.FunctionDeclaration(
    name="get_latest_news",
    description=(
        "Fetch a brief summary of current/recent news or facts on a topic. "
        "Only call this when the user explicitly asks about recent events, current data, "
        "or information that requires up-to-date knowledge beyond your training."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={"query": types.Schema(type="STRING", description="Search query")},
        required=["query"],
    ),
)

ANALYZE_SATELLITE_IMAGE_DECLARATION = types.FunctionDeclaration(
    name="analyze_satellite_image",
    description=(
        "Analyze high-resolution satellite imagery or the user's active map viewport to answer questions about "
        "cars, vehicles, buildings, roads, ships, vegetation, land cover, object counts, and fine-grained visual details. "
        "If the user asks whether cars or specific features are visible at their current zoom level or in the scene, "
        "ALWAYS invoke this tool so the vision model can inspect the exact high-resolution sub-meter satellite crop. "
        "If the user mentions a specific new location (e.g. Paris, Dubai, New York), pass it in 'location'."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "question": types.Schema(
                type="STRING",
                description="The exact natural-language question being asked about the satellite image.",
            ),
            "location": types.Schema(
                type="STRING",
                description="Place name or geographical region to fetch satellite imagery for.",
            ),
            "date": types.Schema(
                type="STRING",
                description="Optional acquisition date or time range (e.g. '2026-08-15' or 'recent').",
            ),
        },
        required=["question"],
    ),
)

GET_LOCATION_COORDINATES_DECLARATION = types.FunctionDeclaration(
    name="get_location_coordinates",
    description=(
        "Look up real-time geographical coordinates (latitude and longitude) and satellite overview "
        "for any city, landmark, port, island, or region on Earth dynamically over the internet."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "location": types.Schema(
                type="STRING",
                description="Place name or location to geocode and locate on map.",
            )
        },
        required=["location"],
    ),
)

COMPARE_SATELLITE_IMAGES_DECLARATION = types.FunctionDeclaration(
    name="compare_satellite_images",
    description=(
        "Compare two temporal satellite images (Past vs Present) for any worldwide location "
        "to detect changes in water levels, vegetation, flood boundaries, urban construction, or infrastructure, and display a heatmap. "
        "Invoke this tool whenever the user asks to compare past vs present imagery, check changes, or asks for a heatmap / change difference."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "location": types.Schema(
                type="STRING",
                description="Location or place name anywhere in the world to compare.",
            ),
            "date1": types.Schema(
                type="STRING",
                description="The baseline/historical year (e.g. '2020', '2016', '2018', '2022', '2024').",
            ),
            "date2": types.Schema(
                type="STRING",
                description="The second/present date (e.g. '2026' or 'current').",
            ),
            "question": types.Schema(
                type="STRING",
                description="The comparative question or heatmap request.",
            ),
        },
        required=["location"],
    ),
)

ZOOM_MAP_DECLARATION = types.FunctionDeclaration(
    name="zoom_map",
    description=(
        "Zoom into, zoom out of, or change the zoom level of the satellite map. "
        "Invoke this tool ONLY when the user EXPLICITLY uses zoom/magnify commands such as: "
        "'zoom in', 'zoom out', 'zoom to max', 'zoom to maximum', 'magnify the map', 'get closer', 'zoom level 18'. "
        "Standard Web Mercator zoom levels: 14 for city overview, 16 for neighborhood, 18 for street/building detail, "
        "and 19 for maximum close-up sub-meter resolution. "
        "RULES: "
        "- To 'zoom in' or 'magnify': set action='zoom_in' and/or zoom_level=18 or 19. Do NOT pass numbers below 13 for zoom in. "
        "- To 'zoom to max': set action='zoom_to_max' and zoom_level=19. "
        "- To 'zoom out': set action='zoom_out'. "
        "DO NOT invoke this tool when the user asks questions about buildings, cars, objects, or features visible on screen. "
        "Those are analysis questions \u2014 use `analyze_satellite_image` for those, not `zoom_map`."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "action": types.Schema(
                type="STRING",
                description="Zoom action: 'zoom_in', 'zoom_out', 'zoom_to_max', or 'focus'.",
            ),
            "zoom_level": types.Schema(
                type="NUMBER",
                description="Target map Web Mercator zoom level (use 18 for street/building level, 19 for maximum close-up sub-meter detail, 14 for city). Do not use numbers below 13 for zoom in. Maximum zoom is 19.",
            ),
            "target_area": types.Schema(
                type="STRING",
                description="Target area or feature to focus on (e.g. 'center', 'north', 'south', 'east', 'west').",
            ),
            "pan_direction": types.Schema(
                type="STRING",
                description="Direction to pan (e.g. 'center', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'north', 'south', 'east', 'west').",
            ),
        },
    ),
)

DEFAULT_TOOLS: List[types.Tool] = [
    types.Tool(function_declarations=[
        NEWS_TOOL_DECLARATION,
        ANALYZE_SATELLITE_IMAGE_DECLARATION,
        GET_LOCATION_COORDINATES_DECLARATION,
        COMPARE_SATELLITE_IMAGES_DECLARATION,
        ZOOM_MAP_DECLARATION,
    ])
]


async def _handle_get_latest_news(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    query = str((args or {}).get("query", ""))
    return await fetch_news_summary(query)


async def _handle_zoom_map(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    args = args or {}
    raw_zoom = args.get("zoom_level")
    action = str(args.get("action", "")).lower()
    target_area = str(args.get("target_area", "current observation view")).strip()
    from voice_speech.engine.conversation.state import get_or_create_session_state
    session_id = (context or {}).get("session_id", "default")
    session_state = get_or_create_session_state(session_id)
    cur_zoom = session_state.active_viewport_zoom or (context.get("viewport_zoom") if context else 15) or 15

    # Helper to extract numeric zoom level from various inputs
    def extract_zoom(val: Any) -> Optional[int]:
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            lower = val.lower()
            if any(w in lower for w in ("max", "maximum", "full", "most", "highest", "closest")):
                return 19
            if any(w in lower for w in ("min", "minimum")):
                return 2
            match = re.search(r"\d+", lower)
            if match:
                return int(match.group())
        return None

    is_zoom_out = (
        "out" in action or
        "out" in target_area.lower() or
        (isinstance(raw_zoom, str) and "out" in raw_zoom.lower()) or
        (isinstance(raw_zoom, (int, float)) and raw_zoom < 0)
    )

    if is_zoom_out:
        target_zoom = max(2, cur_zoom - 3)
        return f"Zoomed out to wider satellite view of {target_area} (level {target_zoom})."

    is_max = (
        "max" in action or
        (isinstance(raw_zoom, str) and any(w in raw_zoom.lower() for w in ("max", "maximum", "full", "most", "highest", "closest")))
    )

    if is_max:
        target_zoom = 19
    else:
        parsed_zoom = extract_zoom(raw_zoom)
        if parsed_zoom is not None and parsed_zoom >= 13:
            target_zoom = min(19, parsed_zoom)
        else:
            # Default or small number (1-12) requested for zoom in -> advance by +2 from cur_zoom
            target_zoom = min(19, cur_zoom + 2)

    return f"Zoomed in to satellite resolution level {target_zoom} (maximum resolution is level 19) on {target_area}."



INVALID_LOCATIONS = {
    "previous context", "previous", "context", "none", "null", "undefined",
    "target location", "current location", "current viewport location", "current context",
    "global", "earth", "world"
}

async def _handle_get_location_coordinates(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    from voice_speech.engine.conversation.state import get_or_create_session_state
    from backend.app.agent.controller import CosmoClipAgentController
    from backend.app.geospatial.realtime_imagery import RealtimeImageryService

    location = str((args or {}).get("location", "")).strip()
    session_id = (context or {}).get("session_id", "default")
    if not location or location.lower() in INVALID_LOCATIONS:
        return "Please specify a valid city or landmark name to look up coordinates."

    geo = RealtimeImageryService.geocode(location)
    if not geo or not geo.get("lat") or not geo.get("lon") or (geo.get("lat") == 0.0 and geo.get("lon") == 0.0):
        return f"Could not find geographic coordinates for '{location}'. Please specify a more specific city or landmark."
    lat = geo["lat"]
    lon = geo["lon"]
    display_name = geo["name"]

    # Trigger agent controller to prepare live imagery and intelligence for this location
    controller = CosmoClipAgentController()
    response = await controller.run({
        "question": f"Show satellite imagery and features around {display_name}",
        "location_name": display_name,
        "session_id": session_id,
        "enable_grounding": True,
        "enable_voice_response": True,
    })

    session_state = get_or_create_session_state(session_id)
    session_state.latest_response = response.model_dump()

    return f"Located {display_name} at latitude {lat:.4f}, longitude {lon:.4f}. Satellite scene loaded on map."


async def _handle_compare_satellite_images(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    from voice_speech.engine.conversation.state import get_or_create_session_state
    from backend.app.agent.controller import CosmoClipAgentController

    args = args or {}
    session_id = (context or {}).get("session_id", "default")
    session_state = get_or_create_session_state(session_id)

    raw_location = args.get("location")
    has_explicit_loc = bool(raw_location and str(raw_location).strip() and str(raw_location).strip().lower() not in INVALID_LOCATIONS)
    location = str(raw_location).strip() if has_explicit_loc else None
    if not location and session_state.active_location_name and session_state.active_location_name.lower() not in INVALID_LOCATIONS:
        location = session_state.active_location_name

    date1 = str(args.get("date1", "2020")).strip()
    date2 = str(args.get("date2", "2026")).strip()
    question = str(args.get("question", "Compare surface features and changes over time")).strip()

    if date1 and date1 not in question and date1 not in ["baseline", "past"]:
        full_query = f"{question} around {location}" if location else question
    elif location and location.lower() not in question.lower():
        full_query = f"{question} around {location}"
    else:
        full_query = question or (f"Compare satellite changes around {location}" if location else "Compare satellite changes")

    viewport_bbox = None if has_explicit_loc else ((args or {}).get("viewport_bbox") or (context or {}).get("viewport_bbox") or session_state.active_viewport_bbox)
    viewport_zoom = None if has_explicit_loc else ((args or {}).get("viewport_zoom") or (context or {}).get("viewport_zoom") or session_state.active_viewport_zoom)
    viewport_captured_at = None if has_explicit_loc else session_state.active_viewport_captured_at

    logger.info(
        f"Bi-temporal change voice tool: full_query='{full_query}', location='{location}', "
        f"zoom={viewport_zoom}, bbox={viewport_bbox}, captured_at={viewport_captured_at}"
    )

    controller = CosmoClipAgentController()
    response = await controller.run({
        "question": full_query,
        "location_name": location if location else None,
        "viewport_bbox": viewport_bbox,
        "viewport_zoom": viewport_zoom,
        "viewport_captured_at": viewport_captured_at,
        "session_id": session_id,
        "enable_grounding": True,
        "enable_voice_response": True,
    })

    session_state.latest_response = response.model_dump()
    if response.is_new_location_query and response.location_meta:
        session_state.session_active_entity = response.location_meta.get("name")
    return response.spoken_text or response.answer


async def _handle_analyze_satellite_image(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    from voice_speech.engine.conversation.state import get_or_create_session_state
    from backend.app.agent.controller import CosmoClipAgentController

    args = args or {}
    session_id = (context or {}).get("session_id", "default")
    session_state = get_or_create_session_state(session_id)

    raw_location = args.get("location")
    has_explicit_loc = bool(raw_location and str(raw_location).strip() and str(raw_location).strip().lower() not in INVALID_LOCATIONS)
    location = str(raw_location).strip() if has_explicit_loc else None
    question = str(args.get("question", "")).strip()

    image_base64 = None if has_explicit_loc else ((args or {}).get("image_base64") or (context or {}).get("image_base64") or session_state.active_screenshot_base64)

    # 1. DIRECT CURRENT-VIEW DISPATCH:
    # If user is asking about the current screen/view without naming a new destination to navigate to,
    # and a live canvas screenshot is available: analyze the active screen pixels immediately.
    # Do NOT append "around <visited_location>" and do NOT re-fetch satellite imagery for the visited place!
    if not has_explicit_loc and image_base64:
        logger.info(f"Analyze satellite voice tool: Direct Current-View screenshot analysis for '{question}'")
        controller = CosmoClipAgentController()
        response = await controller.run_current_view(
            question=question or "What can you see on screen?",
            image_base64=image_base64,
            session_id=session_id
        )
        session_state.latest_response = response.model_dump()
        return response.spoken_text or response.answer

    # 2. NAVIGATION / EXPLICIT LOCATION PATH:
    # Only fallback to active location name if user explicitly didn't provide a screenshot
    if not location and session_state.active_location_name and session_state.active_location_name.lower() not in INVALID_LOCATIONS:
        location = session_state.active_location_name

    if location and location.lower() not in question.lower():
        full_query = f"{question} around {location}"
    else:
        full_query = question or (f"Satellite scene analysis of {location}" if location else "What do you see in the satellite imagery?")

    viewport_bbox = None if has_explicit_loc else ((args or {}).get("viewport_bbox") or (context or {}).get("viewport_bbox") or session_state.active_viewport_bbox)
    viewport_zoom = None if has_explicit_loc else ((args or {}).get("viewport_zoom") or (context or {}).get("viewport_zoom") or session_state.active_viewport_zoom)
    viewport_captured_at = None if has_explicit_loc else session_state.active_viewport_captured_at

    logger.info(
        f"Analyze satellite voice tool: full_query='{full_query}', location='{location}', "
        f"has_screenshot={bool(image_base64)}, zoom={viewport_zoom}, bbox={viewport_bbox}"
    )

    controller = CosmoClipAgentController()
    response = await controller.run({
        "question": full_query,
        "location_name": location if location else None,
        "image_base64": image_base64,
        "viewport_bbox": viewport_bbox,
        "viewport_zoom": viewport_zoom,
        "viewport_captured_at": viewport_captured_at,
        "session_id": session_id,
        "enable_grounding": True,
        "enable_voice_response": True,
    })

    session_state.latest_response = response.model_dump()
    # Update session entity on navigation intent so followup queries use correct context
    if response.is_new_location_query and response.location_meta:
        session_state.session_active_entity = response.location_meta.get("name")
    return response.spoken_text or response.answer


# Extensible Tool Handler Registry
TOOL_REGISTRY: Dict[str, Callable[..., Awaitable[str]]] = {
    "get_latest_news": _handle_get_latest_news,
    "get_location_coordinates": _handle_get_location_coordinates,
    "analyze_satellite_image": _handle_analyze_satellite_image,
    "compare_satellite_images": _handle_compare_satellite_images,
    "zoom_map": _handle_zoom_map,
}





async def dispatch_tool_call(
    name: str,
    args: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Dispatches a function call to the registered handler.

    Args:
        name: Name of the function declared in tool schema.
        args: Parsed argument dictionary from the model.
        context: Optional dictionary containing session metadata (session_id, etc.).

    Returns:
        String result to return to the model in FunctionResponse.
    """
    handler = TOOL_REGISTRY.get(name)
    if not handler:
        logger.warning(f"No handler registered for tool call '{name}'")
        return f"Tool '{name}' is not supported."

    logger.info(f"Executing tool call '{name}' with args={args}, context={context}")
    try:
        import inspect
        sig = inspect.signature(handler)
        if "context" in sig.parameters:
            return await handler(args, context=context)
        return await handler(args)
    except Exception as e:
        logger.error(f"Error executing tool '{name}': {e}", exc_info=True)
        return f"Error executing tool '{name}': {e}"

