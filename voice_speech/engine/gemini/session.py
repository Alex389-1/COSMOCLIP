"""Gemini Live API Session & Connection Configuration Builder.

Builds LiveConnectConfig instances with server-side VAD, prebuilt voice configurations,
context compression, tools, and session resumption handles.
"""

from typing import List, Optional
from google import genai
from google.genai import types

from voice_speech.engine.config.settings import Settings, VADConfig
from voice_speech.engine.config.prompts import get_system_instruction
from voice_speech.engine.gemini.tools import DEFAULT_TOOLS

VALID_VOICES = {"Aoede", "Kore", "Puck", "Charon", "Fenrir"}


def create_gemini_client(api_key: str) -> genai.Client:
    """Creates an asynchronous Gemini Live client.
    
    Pins api_version to 'v1alpha', which is required by Google GenAI SDK for Gemini Live
    bidirectional streaming WebSockets, context compression, and session resumption handles.
    """
    return genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})


def build_vad_config(vad_settings: VADConfig) -> types.AutomaticActivityDetection:
    """Builds server-side Voice Activity Detection configuration."""
    return types.AutomaticActivityDetection(
        disabled=vad_settings.disabled,
        start_of_speech_sensitivity=getattr(
            types.StartSensitivity,
            vad_settings.start_sensitivity,
            types.StartSensitivity.START_SENSITIVITY_LOW,
        ),
        end_of_speech_sensitivity=getattr(
            types.EndSensitivity,
            vad_settings.end_sensitivity,
            types.EndSensitivity.END_SENSITIVITY_HIGH,
        ),
        prefix_padding_ms=vad_settings.prefix_padding_ms,
        silence_duration_ms=vad_settings.silence_duration_ms,
    )


def build_speech_config(voice_name: str) -> types.SpeechConfig:
    """Builds speech output configuration for the chosen prebuilt voice."""
    return types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
        )
    )


def build_thinking_config(thinking_level_str: str) -> Optional[types.ThinkingConfig]:
    """Builds thinking configuration for the Gemini Live model.

    MINIMAL/LOW both map to ThinkingLevel.LOW for lowest latency voice responses.
    None is returned only when thinking is explicitly disabled.
    """
    level = thinking_level_str.upper() if thinking_level_str else "LOW"
    # Map MINIMAL -> LOW to avoid None ThinkingConfig which can cause empty model turns
    if level in ("MINIMAL", "NONE", "OFF", "DISABLED"):
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    gemini_level = getattr(types.ThinkingLevel, level, types.ThinkingLevel.LOW)
    return types.ThinkingConfig(thinking_level=gemini_level)


def build_connect_config(
    settings: Settings,
    voice: str = "Aoede",
    language: str = "auto",
    resumption_handle: Optional[str] = None,
    tools: Optional[List[types.Tool]] = None,
) -> types.LiveConnectConfig:
    """Builds a complete, immutable LiveConnectConfig for a Gemini Live session.

    Args:
        settings: Application settings container.
        voice: Requested prebuilt voice name.
        language: Spoken language code ('auto', 'hindi', 'english', 'hinglish').
        resumption_handle: Optional opaque resumption handle from prior turns.
        tools: Optional list of tools to provide to the model (defaults to DEFAULT_TOOLS).

    Returns:
        Fully configured types.LiveConnectConfig ready for client.aio.live.connect().
    """
    selected_voice = voice if voice in VALID_VOICES else settings.gemini.voice_name
    instruction = get_system_instruction(language)
    vad_config = build_vad_config(settings.vad)
    speech_config = build_speech_config(selected_voice)
    thinking_config = build_thinking_config(settings.gemini.thinking_level)
    active_tools = tools if tools is not None else DEFAULT_TOOLS

    return types.LiveConnectConfig(
        response_modalities=settings.gemini.response_modalities,
        speech_config=speech_config,
        thinking_config=thinking_config,
        tools=active_tools,
        system_instruction=types.Content(parts=[types.Part.from_text(text=instruction)]),
        realtime_input_config=types.RealtimeInputConfig(automatic_activity_detection=vad_config),
        # Context window compression: use a high trigger threshold (32k tokens) to avoid
        # aggressive compression that causes Gemini to emit empty model turns (no text/tools),
        # which results in "model output must contain either output text or tool calls" errors.
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=32000,
            sliding_window=types.SlidingWindow(target_tokens=24000),
        ),
        session_resumption=types.SessionResumptionConfig(handle=resumption_handle),
    )
