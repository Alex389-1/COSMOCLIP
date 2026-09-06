"""Conversation State Management for active WebSocket sessions.

Encapsulates per-client audio state, barge-in epoch tracking, session resumption handle,
and thread-safe WebSocket transmission helpers.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional
from fastapi import WebSocket


@dataclass
class ConversationState:
    """Encapsulates all mutable state for an individual active conversation session."""
    session_id: str = "default"
    session_active: bool = True
    current_epoch: int = 0
    resumption_handle: Optional[str] = None
    active_image_path: Optional[str] = None
    active_image_id: Optional[str] = None
    active_viewport_bbox: Optional[list] = None
    active_viewport_zoom: Optional[int] = None
    active_viewport_captured_at: Optional[float] = None   # ms epoch for staleness check
    active_location_name: Optional[str] = None
    active_center_lat: Optional[float] = None
    active_center_lon: Optional[float] = None
    active_screenshot_base64: Optional[str] = None
    # Navigation-intent session memory — only updated when intent == 'navigation'
    session_active_entity: Optional[str] = None
    latest_response: Optional[dict] = None
    is_playing: bool = False
    mic_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=30))
    ws_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


    def advance_epoch(self) -> int:
        """Increments the epoch counter on barge-in to invalidate obsolete playback buffers."""
        self.current_epoch += 1
        self.is_playing = False
        return self.current_epoch

    def terminate(self) -> None:
        """Marks the session as inactive."""
        self.session_active = False

    async def safe_send_json(self, websocket: WebSocket, data: dict) -> None:
        """Thread-safe WebSocket JSON message delivery."""
        if not self.session_active:
            return
        async with self.ws_lock:
            try:
                await websocket.send_json(data)
            except Exception:
                pass

    async def safe_send_bytes(self, websocket: WebSocket, data: bytes) -> None:
        """Thread-safe WebSocket binary payload delivery."""
        if not self.session_active:
            return
        async with self.ws_lock:
            try:
                await websocket.send_bytes(data)
            except Exception:
                pass


# Global session state registry for multi-turn and cross-endpoint binding
ACTIVE_SESSIONS: dict[str, ConversationState] = {}


def get_or_create_session_state(session_id: str) -> ConversationState:
    """Retrieves or initializes a ConversationState for a session_id."""
    if session_id not in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[session_id] = ConversationState(session_id=session_id)
    return ACTIVE_SESSIONS[session_id]


def remove_session_state(session_id: str) -> None:
    """Cleans up session state on disconnect."""
    ACTIVE_SESSIONS.pop(session_id, None)

