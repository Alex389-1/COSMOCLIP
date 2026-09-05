import time
import numpy as np
from typing import Optional, Tuple

class ServerSideVAD:
    """
    Server-Side Voice Activity Detection (VAD) engine.
    Analyzes streaming audio frames, measures energy/RMS levels,
    tracks speech onset and silence hangover duration to detect speech completion.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 0.018,
        min_speech_duration_ms: float = 300,
        silence_timeout_ms: float = 900
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.silence_timeout_ms = silence_timeout_ms

        self.is_speech_active = False
        self.speech_start_time = 0.0
        self.last_speech_time = 0.0
        self.audio_buffer = bytearray()

    def reset(self):
        self.is_speech_active = False
        self.speech_start_time = 0.0
        self.last_speech_time = 0.0
        self.audio_buffer.clear()

    def process_pcm_frame(self, pcm_bytes: bytes) -> Tuple[bool, bool]:
        """
        Processes a raw 16-bit PCM mono audio chunk.
        Returns: (is_currently_speaking: bool, has_speech_just_ended: bool)
        """
        if not pcm_bytes:
            return False, False

        self.audio_buffer.extend(pcm_bytes)
        now = time.perf_counter() * 1000.0

        # Convert to numpy int16 array and calculate RMS energy
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(samples) == 0:
                return self.is_speech_active, False
            rms = float(np.sqrt(np.mean(samples ** 2)))
        except Exception:
            rms = 0.0

        is_frame_speech = rms > self.energy_threshold

        if is_frame_speech:
            if not self.is_speech_active:
                self.is_speech_active = True
                self.speech_start_time = now
            self.last_speech_time = now
            return True, False

        # Silence frame
        if self.is_speech_active:
            silence_duration = now - self.last_speech_time
            total_speech_duration = self.last_speech_time - self.speech_start_time

            if silence_duration >= self.silence_timeout_ms:
                if total_speech_duration >= self.min_speech_duration_ms:
                    # Speech segment completed successfully
                    self.is_speech_active = False
                    return False, True
                else:
                    # Too short, probably a transient click or noise
                    self.is_speech_active = False
                    return False, False

        return self.is_speech_active, False
