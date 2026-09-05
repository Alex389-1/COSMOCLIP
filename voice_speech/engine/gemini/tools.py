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
        "Analyze a satellite image to answer questions about land cover, water bodies, "
        "urban structures, ships, vegetation, roads, counting, and spatial terrain features. "
        "If the user mentions any place name or location (e.g. Tokyo, Paris, Dubai, New York, Delhi, etc.), "
        "pass it in 'location' to autonomously acquire and analyze a matching Sentinel-2 satellite scene over the internet. "
        "Always invoke this tool whenever the user asks ANY question about satellite imagery or remote-sensing scenes."
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
        "Zoom into, zoom out of, or focus on a specific sector, building, feature, or area of the satellite map image. "
        "Invoke this tool whenever the user asks to 'zoom in', 'zoom out', 'zoom into the building/lake/harbor', "
        "'focus on this area', or 'magnify the map'."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "zoom_level": types.Schema(
                type="NUMBER",
                description="Zoom magnification factor (e.g. 1.5 for slight zoom, 2.0 for 2x, 2.5 for close-up inspection, 1.0 for normal/reset).",
            ),
            "target_area": types.Schema(
                type="STRING",
                description="Target area or feature to focus on (e.g. 'center', 'north', 'south', 'east', 'west', 'building', 'water', 'dock', 'roads').",
            ),
            "pan_direction": types.Schema(
                type="STRING",
                description="Direction to pan (e.g. 'center', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'north', 'south', 'east', 'west').",
            ),
        },
        required=["zoom_level"],
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
    zoom_level = float(args.get("zoom_level", 2.0))
    target_area = str(args.get("target_area", "selected sector")).strip()
    pan_direction = str(args.get("pan_direction", "center")).strip()

    if zoom_level <= 1.0:
        return "Reset satellite map zoom to full scene view."
    return f"Zoomed in to {zoom_level}x magnification focusing on the {target_area}."


async def _handle_get_location_coordinates(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    from voice_speech.engine.conversation.state import get_or_create_session_state
    from backend.app.agent.controller import CosmoClipAgentController
    from backend.app.geospatial.realtime_imagery import RealtimeImageryService

    location = str((args or {}).get("location", "")).strip()
    session_id = (context or {}).get("session_id", "default")
    if not location:
        return "Please specify a location name to look up coordinates."

    geo = RealtimeImageryService.geocode(location)
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
    location = str(args.get("location", "")).strip()
    date1 = str(args.get("date1", "2020")).strip()
    date2 = str(args.get("date2", "2026")).strip()
    question = str(args.get("question", "Compare surface features and changes over time")).strip()
    session_id = (context or {}).get("session_id", "default")

    if date1 and date1 not in question and date1 not in ["baseline", "past"]:
        full_query = f"{question} around {location} between {date1} and {date2 or '2026'}"
    elif location and location.lower() not in question.lower():
        full_query = f"{question} around {location}"
    else:
        full_query = question or (f"Compare satellite changes around {location}" if location else "Compare satellite changes")

    logger.info(f"Bi-temporal change voice tool: full_query='{full_query}', location='{location}', date1='{date1}'")

    controller = CosmoClipAgentController()
    response = await controller.run({
        "question": full_query,
        "location_name": location if location else None,
        "session_id": session_id,
        "enable_grounding": True,
        "enable_voice_response": True,
    })

    session_state = get_or_create_session_state(session_id)
    session_state.latest_response = response.model_dump()

    return response.spoken_text or response.answer


async def _handle_analyze_satellite_image(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    from voice_speech.engine.conversation.state import get_or_create_session_state
    from backend.app.agent.controller import CosmoClipAgentController

    args = args or {}
    question = str(args.get("question", "")).strip()
    location = str(args.get("location", "")).strip() if args.get("location") else None
    session_id = (context or {}).get("session_id", "default")

    if location and location.lower() not in question.lower():
        full_query = f"{question} around {location}"
    else:
        full_query = question or (f"Satellite scene analysis of {location}" if location else "What do you see in the satellite imagery?")

    logger.info(f"Analyze satellite voice tool: full_query='{full_query}', location='{location}'")

    controller = CosmoClipAgentController()
    response = await controller.run({
        "question": full_query,
        "location_name": location if location else None,
        "session_id": session_id,
        "enable_grounding": True,
        "enable_voice_response": True,
    })

    session_state = get_or_create_session_state(session_id)
    session_state.latest_response = response.model_dump()

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

