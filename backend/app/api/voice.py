import json
import logging
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional
from backend.app.voice.live_session import VoiceSessionManager
from backend.app.voice.vad import ServerSideVAD
from backend.app.agent.controller import SatQueryAgentController

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["Voice Interaction"])
voice_manager = VoiceSessionManager()
controller = SatQueryAgentController()

@router.get("/config")
async def get_voice_configuration():
    """
    Returns voice capabilities (Gemini Live WebSocket vs Server VAD WebSocket).
    """
    return {
        "websocket_endpoint": "/api/voice/ws",
        "server_vad_supported": True,
        "gemini_live_available": bool(voice_manager.api_key),
        "gemini_live_model": voice_manager.model_name,
        "default_mode": "websocket_live_vad"
    }

@router.get("/tools")
async def get_voice_tools():
    """
    Returns Gemini Live tool schema declarations for voice function calling.
    """
    return {
        "model": voice_manager.model_name,
        "tools": voice_manager.format_gemini_live_tools()
    }

@router.websocket("/ws")
async def voice_realtime_websocket(websocket: WebSocket):
    """
    Real-Time Continuous Voice WebSocket with Server-Side VAD.
    Maintains a persistent bi-directional connection for uninterrupted speech dialogue.
    """
    await websocket.accept()
    vad = ServerSideVAD(energy_threshold=0.015, silence_timeout_ms=900)
    session_id = f"ws_session_{id(websocket)}"

    # Send initial session greeting
    await websocket.send_json({
        "type": "session_ready",
        "session_id": session_id,
        "status": "connected",
        "message": "COSMOCLIP Real-Time Voice Channel Active (Server VAD enabled)."
    })

    try:
        while True:
            message = await websocket.receive()
            
            # 1. Binary message: Raw PCM audio frame
            if "bytes" in message and message["bytes"]:
                pcm_data = message["bytes"]
                is_speaking, speech_ended = vad.process_pcm_frame(pcm_data)

                if is_speaking:
                    await websocket.send_json({"type": "vad_status", "state": "user_speaking"})

                if speech_ended:
                    await websocket.send_json({"type": "vad_status", "state": "speech_ended_processing"})
                    # Trigger analysis on buffered audio / default query
                    vad.reset()

            # 2. Text/JSON message: Direct streaming voice query / speech transcript / ping
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    payload = {"type": "query", "text": message["text"]}

                msg_type = payload.get("type", "query")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg_type in ["query", "speech_input"]:
                    query_text = payload.get("text", "").strip()
                    if query_text:
                        # Notify client that processing has begun
                        await websocket.send_json({
                            "type": "status",
                            "state": "processing",
                            "query": query_text,
                            "message": f"Analyzing remote sensing imagery for '{query_text}'..."
                        })

                        # Execute stateful LangGraph agent
                        response = await controller.run({
                            "question": query_text,
                            "session_id": session_id,
                            "enable_grounding": payload.get("enable_grounding", True),
                            "enable_voice_response": True
                        })

                        # Stream back full multi-modal response
                        await websocket.send_json({
                            "type": "response",
                            "data": response.model_dump()
                        })

                elif msg_type == "reset":
                    vad.reset()
                    await websocket.send_json({"type": "status", "state": "reset_ready"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket session {session_id} disconnected normally.")
    except Exception as e:
        logger.error(f"WebSocket error in {session_id}: {str(e)}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
