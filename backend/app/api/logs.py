import asyncio
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Set, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/logs", tags=["Real-time Logs"])

MAX_LOG_HISTORY = 400
LOG_HISTORY: List[Dict[str, Any]] = []
SSE_SUBSCRIBERS: Set[asyncio.Queue] = set()
WS_SUBSCRIBERS: Set[WebSocket] = set()
_MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None

def emit_log(level: str, category: str, message: str, meta: Dict[str, Any] = None):
    """
    Publish a real-time log entry to in-memory buffer, all active WebSockets,
    and all SSE queues with immediate broadcast.
    """
    entry = {
        "timestamp": datetime.utcnow().strftime("%H:%M:%S.%f")[:-3],
        "level": level.upper(),
        "category": category.upper(),
        "message": message,
        "meta": meta or {}
    }

    LOG_HISTORY.append(entry)
    if len(LOG_HISTORY) > MAX_LOG_HISTORY:
        LOG_HISTORY.pop(0)

    # 1. Queue to SSE subscribers
    dead_queues = set()
    for q in SSE_SUBSCRIBERS:
        try:
            q.put_nowait(entry)
        except Exception:
            dead_queues.add(q)
    for dq in dead_queues:
        SSE_SUBSCRIBERS.discard(dq)

    # 2. Broadcast to WebSockets
    msg_json = json.dumps({"type": "log", "data": entry})
    dead_ws = set()
    for ws in list(WS_SUBSCRIBERS):
        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(ws.send_text(msg_json))
            except RuntimeError:
                if _MAIN_LOOP and _MAIN_LOOP.is_running():
                    asyncio.run_coroutine_threadsafe(ws.send_text(msg_json), _MAIN_LOOP)
        except Exception:
            dead_ws.add(ws)
    for dws in dead_ws:
        WS_SUBSCRIBERS.discard(dws)

class LiveStreamingLogHandler(logging.Handler):
    """
    Standard logging handler intercepting system logs across all modules.
    """
    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            if not msg or "GET /api/logs" in msg or "GET /api/voice/ws" in msg or "/api/logs/ws" in msg:
                return  # Skip polling log spam

            level = record.levelname
            name = record.name.lower()

            category = "SYSTEM"
            if "geocoder" in name or "nominatim" in msg.lower() or "photon" in msg.lower():
                category = "GEOCODER"
            elif "sentinel" in name or "s2" in msg.lower() or "stac" in msg.lower():
                category = "SENTINEL-2"
            elif "sar" in name or "radar" in msg.lower() or "sigma0" in msg.lower():
                category = "SAR-RADAR"
            elif "mistral" in name or "pixtral" in msg.lower() or "vlm" in msg.lower():
                category = "MISTRAL-AI"
            elif "gemini" in name or "voice" in msg.lower() or "live" in msg.lower():
                category = "GEMINI-LIVE"
            elif "agent" in name or "controller" in name or "langgraph" in msg.lower():
                category = "LANGGRAPH"
            elif "uvicorn" in name or "fastapi" in name:
                category = "SERVER"

            emit_log(level=level, category=category, message=msg)
        except Exception:
            self.handleError(record)

# Attach handler to root logger
handler = LiveStreamingLogHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(handler)

# Seed initial system status events
emit_log("INFO", "SYSTEM", "COSMOCLIP AI Agentic Core Engine initialized & online.")
emit_log("INFO", "LANGGRAPH", "LangGraph Controller v3.0 loaded with multi-modal RS-VQA workflow.")
emit_log("INFO", "SAR-RADAR", "Sentinel-1 C-Band GRD processor standby (Radiometric Terrain Correction).")
emit_log("INFO", "SENTINEL-2", "Sentinel-2 MSI Level-2A STAC Client active at 10m GSD.")
emit_log("INFO", "GEMINI-LIVE", "Gemini Live WebVoice streaming bridge ready on WebSockets.")
emit_log("INFO", "MISTRAL-AI", "Mistral AI Vision-Language reasoning active (pixtral-12b-2409).")

@router.get("/recent")
async def get_recent_logs(since: Optional[str] = None):
    """Retrieve recent log history, optionally filtering by timestamp."""
    if since:
        filtered = [l for l in LOG_HISTORY if l["timestamp"] > since]
        return {"logs": filtered, "server_time": datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]}
    return {"logs": LOG_HISTORY, "server_time": datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]}

@router.websocket("/ws")
async def websocket_logs_endpoint(websocket: WebSocket):
    """
    Direct Real-Time WebSocket endpoint streaming backend logs with zero buffering.
    Immediately sends historical backfill on connection.
    """
    global _MAIN_LOOP
    _MAIN_LOOP = asyncio.get_event_loop()
    await websocket.accept()
    WS_SUBSCRIBERS.add(websocket)

    # Immediately push existing history as initial batch
    initial_payload = json.dumps({
        "type": "history",
        "data": LOG_HISTORY[-60:]
    })
    await websocket.send_text(initial_payload)

    # Keepalive loop
    try:
        while True:
            # Wait for client ping or text messages
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "time": datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        WS_SUBSCRIBERS.discard(websocket)

@router.get("/stream")
async def stream_logs():
    """SSE endpoint streaming live background logs in real-time."""
    queue = asyncio.Queue(maxsize=150)
    SSE_SUBSCRIBERS.add(queue)

    async def event_generator():
        try:
            # Yield historical batch first
            for old_log in LOG_HISTORY[-40:]:
                yield f"data: {json.dumps(old_log)}\n\n"

            while True:
                entry = await queue.get()
                yield f"data: {json.dumps(entry)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            SSE_SUBSCRIBERS.discard(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
