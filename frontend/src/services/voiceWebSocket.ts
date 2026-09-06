import { captureCurrentView } from '../utils/captureView';

export type VoiceState = 'disconnected' | 'connecting' | 'connected' | 'listening' | 'speaking' | 'processing';

export interface ToolStartEvent {
  name: string;
  args: Record<string, any>;
  question?: string;
  location?: string;
}

export interface ToolCallEvent {
  name: string;
  args: Record<string, any>;
  result: string;
  payload?: any;
}


export class VoiceWebSocketClient {
  private ws: WebSocket | null = null;
  private isSessionActive: boolean = false;
  private mediaStream: MediaStream | null = null;

  // Input audio context: browser native sample rate (for mic worklet resampling to 16kHz)
  private inputAudioCtx: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private micDummyDest: MediaStreamAudioDestinationNode | null = null;

  // Output audio context: locked to 24kHz to exactly match Gemini Live PCM output.
  // A dedicated 24kHz context eliminates all browser-side resampling artifacts (buzzing/beeping).
  private outputAudioCtx: AudioContext | null = null;
  private masterGainNode: GainNode | null = null;

  // Audio playback scheduling
  private nextPlayTime = 0;
  private currentEpoch = 0;
  private activeSourceNodes: AudioBufferSourceNode[] = [];

  private onStateChange: (state: VoiceState, message?: string) => void;
  private onToolCall: (event: ToolCallEvent) => void;
  private onToolStart?: (event: ToolStartEvent) => void;

  private currentClientState: VoiceState = 'disconnected';
  private currentMessage?: string;

  private updateState(state: VoiceState, message?: string) {
    if (this.currentClientState === state && (!message || this.currentMessage === message)) {
      return;
    }
    this.currentClientState = state;
    this.currentMessage = message;
    this.onStateChange(state, message);
  }

  constructor(
    onStateChange: (state: VoiceState, message?: string) => void,
    onToolCall: (event: ToolCallEvent) => void,
    onToolStart?: (event: ToolStartEvent) => void
  ) {
    this.onStateChange = onStateChange;
    this.onToolCall = onToolCall;
    this.onToolStart = onToolStart;
  }

  private viewportDebounceTimer: any = null;

  public isActive(): boolean {
    return this.isSessionActive;
  }

  public sendViewportContext(
    viewportBbox?: [number, number, number, number],
    zoom?: number,
    locationName?: string,
    centerLat?: number,
    centerLng?: number,
    immediate: boolean = false
  ) {
    if (this.viewportDebounceTimer) {
      clearTimeout(this.viewportDebounceTimer);
      this.viewportDebounceTimer = null;
    }

    const transmit = () => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN && this.isSessionActive) {
        this.ws.send(
          JSON.stringify({
            type: 'viewport_update',
            viewport_bbox: viewportBbox,
            zoom: zoom,
            location_name: locationName,
            center_lat: centerLat,
            center_lng: centerLng,
            captured_at: Date.now(),
          })
        );
      }
    };

    if (immediate) {
      transmit();
    } else {
      this.viewportDebounceTimer = setTimeout(transmit, 150);
    }
  }

  public async startSession() {
    if (this.isSessionActive) return;
    this.isSessionActive = true;
    this.updateState('connecting', 'Connecting to Gemini Live Voice Channel...');

    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;

      // ── 1. Microphone Input: native rate AudioContext ──────────────────────
      // This context runs at the browser's native sample rate (usually 48000 Hz).
      // The AudioWorklet resamples mic audio down to 16kHz PCM16 for Gemini.
      this.inputAudioCtx = new AudioCtx();

      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      try {
        await this.inputAudioCtx.audioWorklet.addModule('/worklet.js');
        const micSource = this.inputAudioCtx.createMediaStreamSource(this.mediaStream);
        this.workletNode = new AudioWorkletNode(this.inputAudioCtx, 'mic-processor');

        this.workletNode.port.onmessage = (e) => {
          if (this.ws && this.ws.readyState === WebSocket.OPEN && this.isSessionActive) {
            this.ws.send(e.data); // Raw 16kHz PCM16 ArrayBuffer → Gemini Live
          }
        };

        micSource.connect(this.workletNode);
        // Route through virtual dest (not speakers) to keep graph alive
        this.micDummyDest = this.inputAudioCtx.createMediaStreamDestination();
        this.workletNode.connect(this.micDummyDest);
      } catch (workletErr) {
        console.warn('[Voice] AudioWorklet load failed:', workletErr);
      }

      // ── 2. Audio Output: 24kHz context exactly matching Gemini Live PCM ───
      // Locking the output AudioContext to 24kHz means every AudioBufferSourceNode
      // plays at its native rate with ZERO browser resampling. This eliminates the
      // buzzing/chirping artifacts caused by sample-rate mismatch conversion.
      this.outputAudioCtx = new AudioCtx({ sampleRate: 24000 });
      this.masterGainNode = this.outputAudioCtx.createGain();
      // Set gain once — NEVER touch it again. Any gain.cancelScheduledValues()
      // or gain ramp causes an audible click/beep at the point of discontinuity.
      this.masterGainNode.gain.value = 1.0;
      this.masterGainNode.connect(this.outputAudioCtx.destination);

      await Promise.all([
        this.inputAudioCtx.resume(),
        this.outputAudioCtx.resume(),
      ]);

      this.nextPlayTime = this.outputAudioCtx.currentTime;
      this.currentEpoch = 0;

      // ── 3. Connect WebSocket ───────────────────────────────────────────────
      this.connectWebSocket();
    } catch (err: any) {
      console.error('[Voice] Failed to start voice session:', err);
      this.stopSession();
      this.updateState('disconnected', `Microphone error: ${err.message || err}`);
    }
  }

  public stopSession() {
    this.isSessionActive = false;
    this.clearAudioQueue();

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }
    if (this.workletNode) {
      try { this.workletNode.disconnect(); } catch (e) {}
      this.workletNode = null;
    }
    if (this.micDummyDest) {
      try { this.micDummyDest.disconnect(); } catch (e) {}
      this.micDummyDest = null;
    }
    if (this.masterGainNode) {
      try { this.masterGainNode.disconnect(); } catch (e) {}
      this.masterGainNode = null;
    }
    if (this.inputAudioCtx) {
      this.inputAudioCtx.close().catch(() => {});
      this.inputAudioCtx = null;
    }
    if (this.outputAudioCtx) {
      this.outputAudioCtx.close().catch(() => {});
      this.outputAudioCtx = null;
    }
    if (this.ws) {
      try { this.ws.close(); } catch (e) {}
      this.ws = null;
    }

    this.updateState('disconnected', 'Voice session ended');
  }

  public async sendScreenshotUpdate() {
    try {
      const screenshot = await captureCurrentView();
      if (this.ws && this.ws.readyState === WebSocket.OPEN && this.isSessionActive) {
        this.ws.send(JSON.stringify({
          type: 'screenshot_update',
          image_base64: screenshot,
          captured_at: Date.now(),
        }));
      }
    } catch (e) {
      console.debug('[Voice] No map available for screenshot update:', e);
    }
  }

  private reconnectAttempts = 0;
  private maxReconnectAttempts = 8;

  private connectWebSocket() {
    if (!this.isSessionActive) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/voice/ws?session_id=default`;

    try {
      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.updateState('connecting', 'Connecting to Gemini Live Voice Channel...');
      };

      this.ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          this.handleIncomingAudio(event.data);
          return;
        }
        try {
          const msg = JSON.parse(event.data as string);
          this.handleServerMessage(msg);
        } catch (e) {
          console.error('[Voice] Error parsing WS message:', e);
        }
      };

      this.ws.onerror = (err) => {
        console.warn('[Voice] WS error:', err);
      };

      this.ws.onclose = () => {
        if (this.isSessionActive) {
          this.reconnectAttempts++;
          if (this.reconnectAttempts <= this.maxReconnectAttempts) {
            // Exponential backoff: 1.2s, 2.4s, 4.8s... capped at 10s
            const delay = Math.min(10000, 1200 * Math.pow(2, this.reconnectAttempts - 1));
            this.updateState('connecting', `Reconnecting to voice session (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => this.connectWebSocket(), delay);
          } else {
            this.isSessionActive = false;
            this.updateState('disconnected', 'Voice connection lost. Click to reconnect.');
          }
        } else {
          this.updateState('disconnected');
        }
      };
    } catch (err) {
      console.error('[Voice] WebSocket connection failure:', err);
      this.isSessionActive = false;
      this.updateState('disconnected', 'Failed to connect to voice engine.');
    }
  }

  private handleServerMessage(msg: any) {
    switch (msg.type) {
      case 'state':
        if (msg.state === 'PLAYING') {
          this.updateState('speaking', msg.message || 'Gemini is speaking...');
        } else if (msg.state === 'LISTENING') {
          // Let any buffered audio finish before marking as listening
          const remainingMs = this.outputAudioCtx
            ? Math.max(0, (this.nextPlayTime - this.outputAudioCtx.currentTime) * 1000)
            : 0;
          if (remainingMs > 60) {
            setTimeout(() => {
              if (this.isSessionActive) {
                this.updateState('listening', msg.message || 'Gemini Live Voice Active! Speak freely anytime...');
              }
            }, remainingMs);
          } else {
            this.updateState('listening', msg.message || 'Gemini Live Voice Active! Speak freely anytime...');
          }
        } else if (msg.state === 'CONNECTING') {
          this.updateState('connecting', msg.message || 'Connecting to Gemini Live...');
        }
        break;

      case 'barge_in':
        // User interrupted. Advance epoch so all subsequent packets from the old
        // turn are discarded. Then do a SOFT drain — do NOT call node.stop() on
        // any currently-playing node. Stopping mid-waveform creates an abrupt
        // amplitude discontinuity (a click/beep). The in-flight nodes are already
        // scheduled ≤80ms ahead; let them complete silently on their own.
        this.currentEpoch = msg.epoch ?? this.currentEpoch + 1;
        this.drainAudioQueue();
        this.updateState('listening', 'Interruption detected! Listening to you...');
        break;

      case 'tool_start':
        this.updateState('processing', `Processing satellite query: ${msg.location || msg.args?.location || msg.name}...`);
        // Only capture screen for visual analysis tools
        if (msg.name === 'analyze_satellite_image' || msg.name === 'compare_satellite_images') {
          this.sendScreenshotUpdate();
        }
        if (this.onToolStart) {
          this.onToolStart({
            name: msg.name,
            args: msg.args || {},
            question: msg.question,
            location: msg.location,
          });
        }
        break;

      case 'tool_call':
        if (msg.name) {
          this.onToolCall({
            name: msg.name,
            args: msg.args || {},
            result: msg.result || '',
            payload: msg.payload,
          });
        }
        break;

      case 'error':
        this.updateState('listening', msg.message || 'Voice error occurred.');
        break;
    }
  }

  private handleIncomingAudio(arrayBuffer: ArrayBuffer) {
    if (!this.outputAudioCtx || !this.masterGainNode || arrayBuffer.byteLength <= 4) return;

    const dataView = new DataView(arrayBuffer);
    const packetEpoch = dataView.getUint32(0, false); // big-endian epoch header

    // Drop stale packets from interrupted epochs
    if (packetEpoch < this.currentEpoch) return;

    const usableBytes = Math.floor((arrayBuffer.byteLength - 4) / 2) * 2;
    if (usableBytes <= 0) return;

    const int16Array = new Int16Array(arrayBuffer, 4, usableBytes / 2);
    if (int16Array.length === 0) return;

    // Int16 → Float32 [-1.0, 1.0]
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32Array[i] = int16Array[i] / 32768.0;
    }

    const ctx = this.outputAudioCtx;
    const now = ctx.currentTime;
    const ctxRate = ctx.sampleRate; // Use actual context rate (browser may override 24kHz request)

    if (this.nextPlayTime < now) {
      // Buffer underrun or fresh start: add 100ms jitter cushion ahead of now.
      // Apply a short linear fade-in to eliminate DC-step click when restarting from silence.
      this.nextPlayTime = now + 0.1;
      const fadeSamples = Math.min(64, float32Array.length);
      for (let i = 0; i < fadeSamples; i++) {
        float32Array[i] *= i / fadeSamples;
      }
    }

    // Use ctxRate (not hardcoded 24000) so the AudioContext interprets samples correctly
    // even if the browser overrode the requested sample rate.
    const audioBuffer = ctx.createBuffer(1, float32Array.length, ctxRate);
    audioBuffer.copyToChannel(float32Array, 0);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    // masterGainNode stays at 1.0 always — never touch it to prevent beeps
    source.connect(this.masterGainNode);

    source.start(this.nextPlayTime);
    this.nextPlayTime += audioBuffer.duration;

    this.activeSourceNodes.push(source);
    source.onended = () => {
      const idx = this.activeSourceNodes.indexOf(source);
      if (idx !== -1) this.activeSourceNodes.splice(idx, 1);
    };

    if (this.currentClientState !== 'speaking') {
      this.updateState('speaking', 'Gemini speaking live audio...');
    }
  }

  /**
   * SOFT drain — used on barge-in (user interruption).
   * Advances the scheduling cursor and unregisters nodes WITHOUT calling node.stop().
   * In-flight nodes (≤80ms of scheduled audio) drain silently to their natural end.
   * Calling node.stop() mid-sample creates a waveform truncation = audible click/beep.
   */
  private drainAudioQueue() {
    // Abandon ownership of the current nodes so no new audio is appended.
    // The nodes will fire their own onended and GC themselves.
    this.activeSourceNodes = [];
    // Reset to 0 — next new-epoch packet will start fresh with a 100ms jitter buffer.
    // Do NOT set to future time: that would overlap with draining old nodes.
    this.nextPlayTime = 0;
  }

  /**
   * HARD stop — used only on full session teardown (stopSession).
   * Safe to call node.stop() here because the AudioContext is about to be closed
   * anyway, so any residual click is inaudible.
   */
  private clearAudioQueue() {
    const nodesToStop = [...this.activeSourceNodes];
    this.activeSourceNodes = [];
    this.nextPlayTime = 0;

    for (const node of nodesToStop) {
      try {
        node.stop();
        node.disconnect();
      } catch (e) {
        // Already stopped or context closing
      }
    }
  }
}
