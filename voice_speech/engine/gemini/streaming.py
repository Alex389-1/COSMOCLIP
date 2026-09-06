"""Gemini Live Audio Streaming & Bidirectional Pipeline.

Handles low-latency PCM16 audio bridging between browser Web Audio and Gemini Live,
barge-in interruption tracking, tool call execution, and automated session resumption.
"""

import asyncio
import logging
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from google.genai import types

from voice_speech.engine.config.settings import Settings
from voice_speech.engine.conversation.state import ConversationState
from voice_speech.engine.conversation.session_manager import SessionManager
from voice_speech.engine.gemini.session import build_connect_config
from voice_speech.engine.gemini.tools import dispatch_tool_call

logger = logging.getLogger("riva.streaming")


async def ws_reader(websocket: WebSocket, state: ConversationState) -> None:
    """Continuously drains binary PCM16 audio chunks or JSON control messages from the client WebSocket into the mic queue."""
    try:
        while state.session_active:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            # Handle JSON text messages (e.g. live map viewport bounds and zoom updates)
            text_data = msg.get("text")
            if text_data:
                try:
                    import json
                    payload = json.loads(text_data)
                    if isinstance(payload, dict) and payload.get("type") in ("viewport_update", "screenshot_update"):
                        if payload.get("image_base64"):
                            state.active_screenshot_base64 = payload["image_base64"]
                        if "viewport_bbox" in payload:
                            state.active_viewport_bbox = payload.get("viewport_bbox")
                        if "zoom" in payload and payload.get("zoom") is not None:
                            state.active_viewport_zoom = int(payload.get("zoom"))
                        if payload.get("location_name"):
                            state.active_location_name = payload.get("location_name")
                        if payload.get("center_lat") is not None:
                            state.active_center_lat = float(payload.get("center_lat"))
                        if payload.get("center_lng") is not None:
                            state.active_center_lon = float(payload.get("center_lng"))
                        # captured_at for staleness check in tools.py
                        if payload.get("captured_at") is not None:
                            state.active_viewport_captured_at = float(payload["captured_at"])
                        logger.info(
                            f"Live voice context updated: zoom={state.active_viewport_zoom}, "
                            f"has_screenshot={bool(state.active_screenshot_base64)}, loc='{state.active_location_name}'"
                        )
                except Exception as e:
                    logger.debug(f"Non-JSON or invalid text ws message: {e}")
                continue

            pcm_bytes = msg.get("bytes")
            if not pcm_bytes:
                continue
            if state.mic_queue.full():
                try:
                    state.mic_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            state.mic_queue.put_nowait(pcm_bytes)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as e:
        logger.debug(f"ws_reader ended: {e}")
    finally:
        state.terminate()


async def mic_to_gemini(
    session,
    state: ConversationState,
    session_mgr: SessionManager,
    websocket: WebSocket,
) -> None:
    """Transmits microphone audio chunks from the mic queue to Gemini Live in real time."""
    mime_type = "audio/pcm;rate=16000"
    while state.session_active:
        try:
            try:
                pcm_bytes = await asyncio.wait_for(state.mic_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            # Batch-drain all pending chunks for minimal latency
            chunks = [pcm_bytes]
            while not state.mic_queue.empty():
                try:
                    chunks.append(state.mic_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            combined = b"".join(chunks)
            blob = types.Blob(data=combined, mime_type=mime_type)
            await session.send_realtime_input(audio=blob)
            for _ in chunks:
                state.mic_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            if state.session_active:
                logger.error(f"Gemini mic transmit error: {e}")
                if "exhausted" in str(e).lower() or "1011" in str(e):
                    session_mgr.trip_circuit_breaker()
                    state.terminate()
                    await state.safe_send_json(
                        websocket,
                        {"type": "error", "message": f"Gemini API Quota Exceeded. Please wait ~{int(session_mgr.cooldown_seconds)}s before retrying."}
                    )
            break


async def gemini_to_browser(
    session,
    state: ConversationState,
    session_mgr: SessionManager,
    websocket: WebSocket,
) -> None:
    """Receives model audio chunks, barge-in interruptions, and tool calls from Gemini Live."""
    while state.session_active:
        try:
            async for response in session.receive():
                if not state.session_active:
                    break

                # 0. Session Resumption Handle & Go-Away Signals
                resump = getattr(response, "session_resumption", None)
                if resump and getattr(resump, "handle", None):
                    state.resumption_handle = resump.handle
                    logger.debug(f"Saved session resumption handle: {state.resumption_handle[:16]}...")

                go_away = getattr(response, "go_away", None)
                if go_away:
                    time_left = getattr(go_away, "time_left", "N/A")
                    logger.warning(f"Received go_away from Gemini server (time left: {time_left}). Resuming session...")
                    return  # Exit cleanly so outer loop reconnects with resumption_handle

                # 1. Tool Calling Dispatch via Extensible Registry
                tool_call = getattr(response, "tool_call", None)
                if tool_call and getattr(tool_call, "function_calls", None):
                    function_responses = []
                    for fc in tool_call.function_calls:
                        target_loc = str((fc.args or {}).get("location") or (fc.args or {}).get("query") or "").strip()
                        target_q = str((fc.args or {}).get("question") or "").strip()
                        
                        # Immediately notify client to display the progress bar and HUD
                        await state.safe_send_json(
                            websocket,
                            {
                                "type": "tool_start",
                                "name": fc.name,
                                "args": fc.args,
                                "question": target_q or (f"Analyzing satellite scene for {target_loc}" if target_loc else "Analyzing remote sensing scene..."),
                                "location": target_loc,
                            }
                        )

                        state.latest_response = None
                        context = {"session_id": state.session_id}
                        result_str = await dispatch_tool_call(fc.name, fc.args or {}, context=context)
                        
                        # Retrieve structured agent response payload if generated by tool
                        latest_payload = getattr(state, "latest_response", None)
                        if not latest_payload:
                            from voice_speech.engine.conversation.state import get_or_create_session_state
                            conv_state = get_or_create_session_state(state.session_id)
                            latest_payload = getattr(conv_state, "latest_response", None)

                        function_responses.append(
                            types.FunctionResponse(id=fc.id, name=fc.name, response={"result": result_str})
                        )
                        # Notify browser client of execution trace event with full structured payload
                        await state.safe_send_json(
                            websocket,
                            {
                                "type": "tool_call",
                                "name": fc.name,
                                "args": fc.args,
                                "result": result_str,
                                "payload": latest_payload,
                            }
                        )
                    if function_responses:
                        try:
                            await session.send_tool_response(function_responses=function_responses)
                            logger.info(f"Delivered {len(function_responses)} tool response(s) to Gemini.")
                        except Exception as tool_err:
                            logger.error(f"Error delivering tool response: {tool_err}", exc_info=True)

                server_content = response.server_content
                if server_content is None:
                    continue

                # 2. Server-Side Barge-In Interruption
                if getattr(server_content, "interrupted", False):
                    state.is_playing = False
                    new_epoch = state.advance_epoch()
                    logger.info(f"Barge-in triggered by Gemini! Epoch advanced to {new_epoch}.")
                    await state.safe_send_json(
                        websocket,
                        {"type": "barge_in", "epoch": new_epoch, "state": "LISTENING"}
                    )
                    continue

                # 3. Incoming Model Audio Chunks
                model_turn = server_content.model_turn
                if model_turn:
                    for part in model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            if not state.is_playing:
                                state.is_playing = True
                                await state.safe_send_json(websocket, {"type": "state", "state": "PLAYING"})
                            epoch_header = state.current_epoch.to_bytes(4, byteorder="big")
                            await state.safe_send_bytes(websocket, epoch_header + part.inline_data.data)

                # 4. Turn Complete
                if getattr(server_content, "turn_complete", False):
                    logger.info("Gemini turn completed.")
                    state.is_playing = False
                    await state.safe_send_json(websocket, {"type": "state", "state": "LISTENING"})

        except asyncio.CancelledError:
            break
        except Exception as e:
            err_str = str(e).lower()
            if not state.session_active:
                break
            # Soft-recover from empty model turn errors caused by context window compression.
            # These are transient and the session can continue normally after them.
            if "model output" in err_str and ("output text" in err_str or "tool calls" in err_str):
                logger.warning(f"Gemini emitted empty model turn (likely context compression artefact). Continuing session. Detail: {e}")
                state.is_playing = False
                await state.safe_send_json(websocket, {"type": "state", "state": "LISTENING"})
                continue
            logger.error(f"Gemini receive error: {e}")
            if "exhausted" in err_str or "1011" in str(e):
                session_mgr.trip_circuit_breaker()
                state.terminate()
                await state.safe_send_json(
                    websocket,
                    {"type": "error", "message": f"Gemini API Quota Exceeded. Please wait ~{int(session_mgr.cooldown_seconds)}s before retrying."}
                )
            break


async def run_live_bridge(
    client,
    websocket: WebSocket,
    settings: Settings,
    state: ConversationState,
    session_mgr: SessionManager,
    voice: str = "Aoede",
    language: str = "auto",
) -> None:
    """Runs the persistent bidirectional bridge with automated session resumption reconnects."""
    model_name = settings.gemini.model
    reader_task = asyncio.create_task(ws_reader(websocket, state))
    # Max reconnect attempts per connection loss event (not cumulative across the session)
    max_attempts = 8

    try:
        while state.session_active:
            # Reset attempt counter at the start of each outer loop cycle.
            # This ensures a previous failure run doesn't block future reconnects
            # after a successful go_away → resumption cycle.
            connect_attempts = 0

            # Check circuit breaker before each reconnect attempt
            is_open, remaining = session_mgr.is_circuit_open()
            if is_open:
                logger.warning(f"Active quota cooldown in progress ({remaining}s). Aborting session bridge.")
                await state.safe_send_json(
                    websocket,
                    {"type": "error", "message": f"Gemini API rate limit active. Please wait {remaining}s."}
                )
                state.terminate()
                break

            connect_config = build_connect_config(
                settings=settings,
                voice=voice,
                language=language,
                resumption_handle=state.resumption_handle,
            )
            is_resumed = bool(state.resumption_handle)
            logger.info(f"Connecting to Gemini Live (model={model_name}, voice={voice}, resumed={is_resumed})...")

            connected_ok = False
            while state.session_active and connect_attempts < max_attempts:
                try:
                    await state.safe_send_json(
                        websocket,
                        {"type": "state", "state": "CONNECTING", "message": f"Connecting to Gemini Live ({model_name})..."}
                    )

                    async with client.aio.live.connect(model=model_name, config=connect_config) as session:
                        connect_attempts = 0
                        connected_ok = True
                        logger.info("Connected to Gemini Live session successfully!")
                        await state.safe_send_json(
                            websocket,
                            {"type": "state", "state": "LISTENING", "message": "Gemini Live Voice Active! Speak freely anytime..."}
                        )

                        mic_task = asyncio.create_task(mic_to_gemini(session, state, session_mgr, websocket))
                        gemini_task = asyncio.create_task(gemini_to_browser(session, state, session_mgr, websocket))

                        done, pending = await asyncio.wait(
                            [mic_task, gemini_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for t in pending:
                            t.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)

                    # Clean session exit (go_away handled, or task completed normally)
                    break

                except Exception as conn_err:
                    if not state.session_active:
                        break

                    err_str = str(conn_err)
                    logger.warning(f"Gemini connection error (attempt {connect_attempts + 1}/{max_attempts}): {err_str[:120]}")

                    # Fatal: quota exhaustion — no point retrying
                    if "exhausted" in err_str.lower() or "1011" in err_str:
                        session_mgr.trip_circuit_breaker()
                        await state.safe_send_json(
                            websocket,
                            {"type": "error", "message": f"Gemini API Quota Exceeded. Please wait ~{int(session_mgr.cooldown_seconds)}s before retrying."}
                        )
                        state.terminate()
                        return

                    connect_attempts += 1
                    if connect_attempts < max_attempts and state.session_active:
                        # Exponential backoff: 1s, 2s, 4s, 8s … capped at 10s
                        backoff = min(10.0, 2 ** (connect_attempts - 1))
                        logger.info(f"Retrying Gemini connection in {backoff:.0f}s ({connect_attempts}/{max_attempts})...")
                        await state.safe_send_json(
                            websocket,
                            {"type": "state", "state": "CONNECTING", "message": f"Reconnecting to Gemini Live ({connect_attempts}/{max_attempts})..."}
                        )
                        await asyncio.sleep(backoff)
                    else:
                        # All attempts exhausted
                        if state.resumption_handle and state.session_active:
                            # We have a resumption handle — try once more in the outer loop
                            logger.warning("All retry attempts exhausted but resumption handle present. Outer loop will retry.")
                        else:
                            await state.safe_send_json(
                                websocket,
                                {"type": "error", "message": f"Could not connect to Gemini Live after {max_attempts} attempts."}
                            )
                            state.terminate()
                        break

            if not state.session_active:
                break

            # Session ended cleanly (go_away or normal turn end) — resume seamlessly
            if state.resumption_handle and state.session_active:
                logger.info("Gemini session ended. Seamless resumption in 0.3s...")
                await asyncio.sleep(0.3)
                # Rebuild config with updated resumption handle for next connect
                connect_config = build_connect_config(
                    settings=settings,
                    voice=voice,
                    language=language,
                    resumption_handle=state.resumption_handle,
                )
            elif not connected_ok:
                # Never connected successfully and no resumption handle — give up
                break

    finally:
        state.terminate()
        reader_task.cancel()
        await asyncio.gather(reader_task, return_exceptions=True)
