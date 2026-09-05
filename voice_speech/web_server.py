import logging
import os
import shutil
import sys
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from voice_speech.engine.config.settings import Settings
from voice_speech.engine.conversation.session_manager import SessionManager
from voice_speech.engine.conversation.state import (
    ConversationState,
    get_or_create_session_state,
    remove_session_state,
)
from voice_speech.engine.gemini.session import create_gemini_client
from voice_speech.engine.gemini.streaming import run_live_bridge

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("riva.web_server")

# Load unified settings
settings = Settings()

# Shared Gemini Live Client and Concurrency / Rate Limiting Session Manager
api_key = settings.gemini.api_key or os.getenv("GEMINI_API_KEY", "dummy-api-key-for-tests")
if not settings.gemini.api_key:
    logger.warning("GEMINI_API_KEY is not set. Using test fallback key for local endpoints.")

gemini_client = create_gemini_client(api_key=api_key)
session_manager = SessionManager()

app = FastAPI(title="COSMOCLIP AI Voice Gateway")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
SESSIONS_STORAGE_DIR = Path(BASE_DIR) / "data" / "sessions"
SESSIONS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# Static Web UI Routes
@app.get("/")
async def get_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/app.js")
async def get_app_js():
    return FileResponse(os.path.join(WEB_DIR, "app.js"))


@app.get("/worklet.js")
async def get_worklet_js():
    return FileResponse(os.path.join(WEB_DIR, "worklet.js"))


@app.get("/favicon.ico")
async def get_favicon():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="45" fill="#4dbc1b"/></svg>'
    return Response(content=svg, media_type="image/svg+xml")


# Satellite Image Upload & Session Management API (PRD F1, §10)
@app.post("/session/{session_id}/image")
@app.post("/api/upload")
async def upload_satellite_image(
    session_id: str = "default",
    file: UploadFile = File(...),
):
    """Uploads, validates, and binds a satellite image tile to an active session."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Only PNG and JPEG satellite images are supported in V1.",
        )

    # Prepare session storage
    session_dir = SESSIONS_STORAGE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    saved_filename = f"active_image{ext}"
    dest_path = session_dir / saved_filename

    # Save to disk
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File exceeds limit of {MAX_FILE_SIZE // (1024*1024)}MB.")

    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    # Bind to session state
    state = get_or_create_session_state(session_id)
    state.active_image_path = str(dest_path)
    state.active_image_id = file.filename

    logger.info(f"Uploaded satellite image for session '{session_id}': {file.filename} -> {dest_path}")

    return JSONResponse(
        {
            "status": "success",
            "session_id": session_id,
            "image_id": file.filename,
            "image_path": str(dest_path),
            "image_url": f"/session/{session_id}/image",
            "message": "Satellite image loaded successfully.",
        }
    )


@app.get("/api/fetch")
@app.post("/api/fetch")
async def api_fetch_satellite_image(
    location: str,
    date: Optional[str] = None,
    session_id: str = "default",
):
    """Acquires a live Sentinel-2 satellite image for a given place name (PRD §8.1 / §10.3)."""
    from voice_speech.engine.vqa.fetch import fetch_satellite_image

    fetch_res = await fetch_satellite_image(location=location, date=date, session_id=session_id)
    if fetch_res.status != "success":
        raise HTTPException(status_code=400, detail=fetch_res.error_message or "Satellite image acquisition failed.")

    state = get_or_create_session_state(session_id)
    state.active_image_path = fetch_res.image_path
    state.active_image_id = f"sentinel2_{location.lower().replace(' ', '_')}"

    return JSONResponse(
        {
            "status": "success",
            "session_id": session_id,
            "location": location,
            "latitude": fetch_res.latitude,
            "longitude": fetch_res.longitude,
            "scene_date": fetch_res.scene_date,
            "cloud_cover_pct": fetch_res.cloud_cover_pct,
            "image_url": f"/session/{session_id}/image",
            "latency_ms": fetch_res.latency_ms,
        }
    )


@app.get("/session/{session_id}/image")
async def get_session_image(session_id: str = "default"):
    """Serves the currently active image for the session."""
    state = get_or_create_session_state(session_id)
    if state.active_image_path and os.path.exists(state.active_image_path):
        return FileResponse(state.active_image_path)

    # Check disk fallback
    session_dir = SESSIONS_STORAGE_DIR / session_id
    for ext in ALLOWED_EXTENSIONS:
        candidate = session_dir / f"active_image{ext}"
        if candidate.exists():
            state.active_image_path = str(candidate)
            return FileResponse(str(candidate))

    raise HTTPException(status_code=404, detail="No active satellite image found for this session.")


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
        # Fallback to active image or generate placeholder
        default_candidate = session_dir / "active_image.png"
        if default_candidate.exists():
            return FileResponse(str(default_candidate), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        raise HTTPException(status_code=404, detail=f"Change image for '{target}' not found.")

    return FileResponse(str(file_path), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})



@app.get("/api/compare")
@app.post("/api/compare")
async def api_compare_satellite_images(
    location: str,
    date1: str = "2026-08-31",
    date2: str = "2026-09-02",
    question: str = "Compare surface features and changes",
    session_id: str = "default",
):
    """Executes bi-temporal satellite change detection between two dates."""
    from voice_speech.engine.gemini.tools import dispatch_tool_call

    answer = await dispatch_tool_call(
        "compare_satellite_images",
        {"location": location, "date1": date1, "date2": date2, "question": question},
        context={"session_id": session_id},
    )

    return JSONResponse(
        {
            "status": "success",
            "session_id": session_id,
            "location": location,
            "date1": date1,
            "date2": date2,
            "summary": answer,
            "t1_url": f"/session/{session_id}/change/t1",
            "t2_url": f"/session/{session_id}/change/t2",
            "diff_url": f"/session/{session_id}/change/diff",
        }
    )




# Real-Time Bidirectional Voice WebSocket Endpoint
@app.websocket("/ws")
async def audio_websocket_endpoint(websocket: WebSocket):
    # 1. Origin validation
    origin = websocket.headers.get("origin", "")
    allowed_origins = {"http://localhost:8000", "http://localhost", "http://127.0.0.1:8000", "https://localhost:8000"}
    if origin and origin not in allowed_origins:
        logger.warning(f"Rejected WebSocket from unauthorized origin: {origin}")
        await websocket.close(code=4003, reason="Origin not allowed")
        return

    # 2. Concurrency & Circuit Breaker Admission
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

    session_id = websocket.query_params.get("session_id") or "default"
    voice = websocket.query_params.get("voice") or "Aoede"
    language = websocket.query_params.get("language") or "auto"
    logger.info(
        f"Browser client connected to /ws (session={session_id}, voice={voice}, language={language}) "
        f"[{session_manager.active_sessions}/{session_manager.max_concurrent_sessions} sessions]"
    )

    state = get_or_create_session_state(session_id)
    state.session_active = True

    try:
        await run_live_bridge(
            client=gemini_client,
            websocket=websocket,
            settings=settings,
            state=state,
            session_mgr=session_manager,
            voice=voice,
            language=language,
        )
    except WebSocketDisconnect:
        logger.info(f"Browser client '{session_id}' disconnected cleanly.")
    except Exception as e:
        logger.error(f"WebSocket bridge exception for '{session_id}': {e}", exc_info=True)
    finally:
        state.terminate()
        await session_manager.release()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("voice_speech.web_server:app", host="0.0.0.0", port=port, log_level="info")

