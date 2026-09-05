import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from backend.app.api.health import router as health_router
from backend.app.api.imagery import router as imagery_router
from backend.app.api.query import router as query_router
from backend.app.api.logs import router as logs_router

# Import Voice Speech Engine
from voice_speech.engine.config.settings import Settings
from voice_speech.engine.conversation.session_manager import SessionManager
from voice_speech.engine.conversation.state import (
    ConversationState,
    get_or_create_session_state,
    remove_session_state,
)
from voice_speech.engine.gemini.session import create_gemini_client
from voice_speech.engine.gemini.streaming import run_live_bridge
from voice_speech.engine.gemini.tools import dispatch_tool_call

logger = logging.getLogger("cosmoclip.main")

# Load unified settings & Gemini client
settings = Settings()
api_key = settings.gemini.api_key or os.getenv("GEMINI_API_KEY", "")
gemini_client = create_gemini_client(api_key=api_key) if api_key else None
session_manager = SessionManager()

app = FastAPI(
    title="COSMOCLIP AI Backend",
    description="Interactive Agentic Vision-Language Assistant with Gemini Live Voice Engine (ISRO PS 26167)",
    version="0.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST Routers
app.include_router(health_router)
app.include_router(imagery_router)
app.include_router(query_router)
app.include_router(logs_router)

# Storage directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SESSIONS_STORAGE_DIR = BASE_DIR / "voice_speech" / "data" / "sessions"
SESSIONS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# ------------------------------------------------------------------------------
# Voice Speech Engine Session & Imagery Endpoints
# ------------------------------------------------------------------------------

@app.get("/session/{session_id}/image")
async def get_session_image(session_id: str = "default"):
    """Serves the currently active image for the session."""
    state = get_or_create_session_state(session_id)
    if state.active_image_path and os.path.exists(state.active_image_path):
        return FileResponse(state.active_image_path)

    session_dir = SESSIONS_STORAGE_DIR / session_id
    for ext in ALLOWED_EXTENSIONS:
        candidate = session_dir / f"active_image{ext}"
        if candidate.exists():
            state.active_image_path = str(candidate)
            return FileResponse(str(candidate))

    # Fallback default image
    default_candidate = BASE_DIR / "data" / "sample_scenes" / "lake_pichola_s2.png"
    if default_candidate.exists():
        return FileResponse(str(default_candidate))

    raise HTTPException(status_code=404, detail="No active satellite image found.")

@app.get("/session/{session_id}/change/{target}")
async def get_change_image(session_id: str = "default", target: str = "diff"):
    """Serves bi-temporal images: 't1' (past), 't2' (present), or 'diff' (marked change overlay)."""
    session_dir = SESSIONS_STORAGE_DIR / session_id
    mapping = {
        "t1": "change_t1.png",
        "t2": "change_t2.png",
        "diff": "diff_overlay.png",
    }
    fname = mapping.get(target, "diff_overlay.png")
    file_path = session_dir / fname

    if not file_path.exists():
        default_candidate = session_dir / "active_image.png"
        if default_candidate.exists():
            return FileResponse(str(default_candidate), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        
        fallback_scene = BASE_DIR / "data" / "cache" / f"rt_{'before' if target == 't1' else 'after'}_mumbai_port___eastern_waterfront.png"
        if fallback_scene.exists():
            return FileResponse(str(fallback_scene), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        
        raise HTTPException(status_code=404, detail=f"Change image for '{target}' not found.")

    return FileResponse(str(file_path), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

class CompareRequest(BaseModel):
    location: str
    date1: Optional[str] = "2021-11-15"
    date2: Optional[str] = "2026-08-28"
    question: Optional[str] = "Compare surface features and changes"
    session_id: Optional[str] = "default"

@app.post("/api/compare")
async def api_compare_satellite_images_post(payload: CompareRequest):
    """Executes bi-temporal satellite change detection between two dates."""
    answer = await dispatch_tool_call(
        "compare_satellite_images",
        {"location": payload.location, "date1": payload.date1, "date2": payload.date2, "question": payload.question},
        context={"session_id": payload.session_id},
    )

    return JSONResponse({
        "status": "success",
        "session_id": payload.session_id,
        "location": payload.location,
        "date1": payload.date1,
        "date2": payload.date2,
        "summary": answer,
        "t1_url": f"/session/{payload.session_id}/change/t1",
        "t2_url": f"/session/{payload.session_id}/change/t2",
        "diff_url": f"/session/{payload.session_id}/change/diff",
    })

@app.get("/api/compare")
async def api_compare_satellite_images_get(
    location: str,
    date1: str = "2021-11-15",
    date2: str = "2026-08-28",
    question: str = "Compare surface features and changes",
    session_id: str = "default",
):
    """Executes bi-temporal satellite change detection via query params."""
    answer = await dispatch_tool_call(
        "compare_satellite_images",
        {"location": location, "date1": date1, "date2": date2, "question": question},
        context={"session_id": session_id},
    )

    return JSONResponse({
        "status": "success",
        "session_id": session_id,
        "location": location,
        "date1": date1,
        "date2": date2,
        "summary": answer,
        "t1_url": f"/session/{session_id}/change/t1",
        "t2_url": f"/session/{session_id}/change/t2",
        "diff_url": f"/session/{session_id}/change/diff",
    })


@app.get("/api/voice/config")
async def get_voice_config():
    return {
        "websocket_endpoint": "/ws",
        "gemini_live_available": bool(api_key and gemini_client),
        "gemini_live_model": settings.gemini.model,
        "voice_name": settings.gemini.voice_name,
        "server_vad_supported": True,
        "default_mode": "gemini_live" if api_key else "web_speech_local"
    }

# ------------------------------------------------------------------------------
# Real-Time Bidirectional Voice WebSocket (Gemini Live Engine + Server VAD)
# ------------------------------------------------------------------------------

@app.websocket("/ws")
@app.websocket("/api/voice/ws")
async def audio_websocket_endpoint(websocket: WebSocket):
    session_id = websocket.query_params.get("session_id") or "default"
    voice = websocket.query_params.get("voice") or settings.gemini.voice_name
    language = websocket.query_params.get("language") or "auto"

    acquired, error_msg = await session_manager.try_acquire()
    if not acquired:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": error_msg})
        await websocket.close(code=4029)
        return

    try:
        await websocket.accept()
    except Exception:
        await session_manager.release()
        return

    logger.info(f"Client connected to Gemini Live /ws (session={session_id}, voice={voice})")
    state = get_or_create_session_state(session_id)
    state.session_active = True

    try:
        if gemini_client and api_key:
            # Full Gemini Live Bidirectional Streaming Bridge with Server-Side VAD
            await run_live_bridge(
                client=gemini_client,
                websocket=websocket,
                settings=settings,
                state=state,
                session_mgr=session_manager,
                voice=voice,
                language=language,
            )
        else:
            # Fallback simulated VAD loop if no Gemini key
            await websocket.send_json({
                "type": "state",
                "state": "LISTENING",
                "message": "Connected in local audio mode (Gemini Live API key not set)."
            })
            while state.session_active:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
    except WebSocketDisconnect:
        logger.info(f"Client '{session_id}' disconnected cleanly.")
    except Exception as e:
        logger.error(f"WebSocket bridge error for '{session_id}': {e}")
    finally:
        state.terminate()
        await session_manager.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
