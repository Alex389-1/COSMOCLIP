import os
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_LIVE_MODEL = "gemini-3.1-flash-live-preview"

class VoiceSessionManager:
    """
    Manages real-time audio interaction sessions.
    Supports Gemini 3.1 Flash Live Preview WebSocket audio dialogue
    with graceful fallback to browser Web Speech API & SpeechSynthesis.
    """

    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.model_name = GEMINI_LIVE_MODEL

    def get_capabilities(self) -> Dict[str, Any]:
        has_gemini_key = bool(self.api_key)
        return {
            "gemini_live_available": has_gemini_key,
            "gemini_live_model": self.model_name,
            "local_web_speech_supported": True,
            "browser_tts_supported": True,
            "default_mode": "gemini_live" if has_gemini_key else "web_speech_local"
        }

    def format_gemini_live_tools(self) -> List[Dict[str, Any]]:
        """
        Tool schema definitions for Gemini 3.1 Flash Live Preview function calling.
        """
        return [
            {
                "name": "search_satellite_image",
                "description": "Searches Copernicus Sentinel-2 optical scenes by location, bounding box, date, and cloud cover.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "location_name": {"type": "STRING", "description": "City, landmark, or region name"},
                        "max_cloud_cover": {"type": "NUMBER", "description": "Max allowable cloud percentage (0-100)"}
                    },
                    "required": ["location_name"]
                }
            },
            {
                "name": "satellite_vqa",
                "description": "Asks a natural language question about an earth observation satellite scene.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "scene_id": {"type": "STRING", "description": "The Sentinel-2 scene identifier"},
                        "question": {"type": "STRING", "description": "The natural language query"}
                    },
                    "required": ["question"]
                }
            }
        ]
