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
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private masterGainNode: GainNode | null = null;
  private dummyDestNode: MediaStreamAudioDestinationNode | null = null;

  // Audio Playback Pipeline (24kHz PCM from Gemini Live)
  private outputSampleRate = 24000;
  private nextPlayTime = 0;
  private currentEpoch = 0;
  private activeSourceNodes: AudioBufferSourceNode[] = [];

  private onStateChange: (state: VoiceState, message?: string) => void;
  private onToolCall: (event: ToolCallEvent) => void;
  private onToolStart?: (event: ToolStartEvent) => void;

  constructor(
    onStateChange: (state: VoiceState, message?: string) => void,
    onToolCall: (event: ToolCallEvent) => void,
    onToolStart?: (event: ToolStartEvent) => void
  ) {
    this.onStateChange = onStateChange;
    this.onToolCall = onToolCall;
    this.onToolStart = onToolStart;
  }

  public isActive(): boolean {
    return this.isSessionActive;
  }

  public async startSession() {
    if (this.isSessionActive) return;
    this.isSessionActive = true;
    this.onStateChange('connecting', 'Connecting to Gemini Live Voice Channel...');

    try {
      // 1. Initialize Web Audio Context
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      this.audioContext = new AudioCtx();
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      // Master output gain node for smooth audio playback envelopes (prevents DC pops)
      this.masterGainNode = this.audioContext.createGain();
      this.masterGainNode.gain.setValueAtTime(1.0, this.audioContext.currentTime);
      this.masterGainNode.connect(this.audioContext.destination);

      // 2. Request Microphone Access
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // 3. Load AudioWorklet Processor for 16kHz PCM16 resampling
      try {
        await this.audioContext.audioWorklet.addModule('/worklet.js');
        const sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
        this.workletNode = new AudioWorkletNode(this.audioContext, 'mic-processor');

        this.workletNode.port.onmessage = (e) => {
          if (this.ws && this.ws.readyState === WebSocket.OPEN && this.isSessionActive) {
            this.ws.send(e.data); // Stream raw 16kHz PCM16 ArrayBuffer
          }
        };

        sourceNode.connect(this.workletNode);
        // Connect to a virtual MediaStreamAudioDestinationNode instead of real speakers
        // to keep the AudioWorklet graph active without routing mic audio or DC-steps to DAC
        this.dummyDestNode = this.audioContext.createMediaStreamDestination();
        this.workletNode.connect(this.dummyDestNode);
      } catch (workletErr) {
        console.warn('AudioWorklet load fallback:', workletErr);
      }

      // 4. Connect WebSocket to backend Gemini Live Gateway
      this.connectWebSocket();
    } catch (err: any) {
      console.error('Failed to start voice session:', err);
      this.stopSession();
      this.onStateChange('disconnected', `Microphone error: ${err.message || err}`);
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
      try {
        this.workletNode.disconnect();
      } catch (e) {}
      this.workletNode = null;
    }
    if (this.dummyDestNode) {
      try {
        this.dummyDestNode.disconnect();
      } catch (e) {}
      this.dummyDestNode = null;
    }
    if (this.masterGainNode) {
      try {
        this.masterGainNode.disconnect();
      } catch (e) {}
      this.masterGainNode = null;
    }
    if (this.audioContext) {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
    if (this.ws) {
      try {
        this.ws.close();
      } catch (e) {}
      this.ws = null;
    }

    this.onStateChange('disconnected', 'Voice session ended');
  }

  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;

  private connectWebSocket() {
    if (!this.isSessionActive) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Use window.location.host so Vite proxy handles WebSocket forwarding seamlessly
    const wsUrl = `${protocol}//${window.location.host}/api/voice/ws?session_id=default`;

    try {
      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.onStateChange('connecting', 'Connecting to Gemini Live Voice Channel...');
      };

      this.ws.onmessage = (event) => {
        // Binary Data: 24kHz PCM Audio Chunks from Gemini Live
        if (event.data instanceof ArrayBuffer) {
          this.handleIncomingAudio(event.data);
          return;
        }

        // JSON Data: Events, Tool Calls, State Changes, Barge-In
        try {
          const msg = JSON.parse(event.data);
          this.handleServerMessage(msg);
        } catch (e) {
          console.error('Error parsing WS message:', e);
        }
      };

      this.ws.onerror = (err) => {
        console.warn('Gemini Live WS warning:', err);
      };

      this.ws.onclose = () => {
        if (this.isSessionActive) {
          this.reconnectAttempts++;
          if (this.reconnectAttempts <= this.maxReconnectAttempts) {
            this.onStateChange('connecting', `Connecting to voice session (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => this.connectWebSocket(), 1200);
          } else {
            this.isSessionActive = false;
            this.onStateChange('disconnected', 'Voice connection lost. Click to reconnect.');
          }
        } else {
          this.onStateChange('disconnected');
        }
      };
    } catch (err) {
      console.error('WebSocket connection failure:', err);
      this.isSessionActive = false;
      this.onStateChange('disconnected', 'Failed to connect to voice engine.');
    }
  }

  private handleServerMessage(msg: any) {
    switch (msg.type) {
      case 'state':
        if (msg.state === 'PLAYING') {
          this.onStateChange('speaking', msg.message || 'Gemini is speaking...');
        } else if (msg.state === 'LISTENING') {
          // If speech audio chunks are still scheduled, let playback finish smoothly
          const remainingMs = this.audioContext ? Math.max(0, (this.nextPlayTime - this.audioContext.currentTime) * 1000) : 0;
          if (remainingMs > 60) {
            setTimeout(() => {
              if (this.isSessionActive && (!this.audioContext || this.audioContext.currentTime >= this.nextPlayTime - 0.05)) {
                this.onStateChange('listening', msg.message || 'Gemini Live Voice Active! Speak freely anytime...');
              }
            }, remainingMs);
          } else {
            this.onStateChange('listening', msg.message || 'Gemini Live Voice Active! Speak freely anytime...');
          }
        } else if (msg.state === 'CONNECTING') {
          this.onStateChange('connecting', msg.message || 'Connecting to Gemini Live...');
        }
        break;

      case 'barge_in':
        // User interrupted model speaking: smoothly fade out pending audio buffers
        this.currentEpoch = msg.epoch ?? this.currentEpoch + 1;
        this.clearAudioQueue();
        this.onStateChange('listening', 'Interruption detected! Listening to you...');
        break;

      case 'tool_start':
        this.onStateChange('processing', `Processing satellite query: ${msg.location || msg.args?.location || msg.name}...`);
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
        this.onStateChange('listening', msg.message || 'Voice error occurred.');
        break;
    }
  }

  private handleIncomingAudio(arrayBuffer: ArrayBuffer) {
    if (!this.audioContext || !this.masterGainNode || arrayBuffer.byteLength <= 4) return;

    const dataView = new DataView(arrayBuffer);
    const packetEpoch = dataView.getUint32(0, false); // 4-byte big endian epoch

    // Discard stale packets from previous speech epochs before barge-in
    if (packetEpoch < this.currentEpoch) {
      return;
    }

    const usableBytes = Math.floor((arrayBuffer.byteLength - 4) / 2) * 2;
    if (usableBytes <= 0) return;

    const pcmData = new Int16Array(arrayBuffer, 4, usableBytes / 2);
    if (pcmData.length === 0) return;

    // Convert Int16 PCM to Float32 [-1.0, 1.0]
    const float32Data = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      float32Data[i] = pcmData[i] / 32768.0;
    }

    const audioBuffer = this.audioContext.createBuffer(1, float32Data.length, this.outputSampleRate);
    audioBuffer.getChannelData(0).set(float32Data);

    const sourceNode = this.audioContext.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.connect(this.masterGainNode);

    const now = this.audioContext.currentTime;
    if (this.nextPlayTime < now) {
      this.nextPlayTime = now + 0.02; // Small 20ms jitter buffer
      // Smooth fade-in from silence
      try {
        this.masterGainNode.gain.cancelScheduledValues(now);
        this.masterGainNode.gain.setValueAtTime(0.001, now);
        this.masterGainNode.gain.exponentialRampToValueAtTime(1.0, now + 0.015);
      } catch (e) {}
    }

    sourceNode.start(this.nextPlayTime);
    this.nextPlayTime += audioBuffer.duration;

    this.activeSourceNodes.push(sourceNode);
    sourceNode.onended = () => {
      const idx = this.activeSourceNodes.indexOf(sourceNode);
      if (idx !== -1) this.activeSourceNodes.splice(idx, 1);
    };

    this.onStateChange('speaking', 'Gemini speaking live audio...');
  }

  private clearAudioQueue() {
    if (this.audioContext && this.masterGainNode) {
      const now = this.audioContext.currentTime;
      try {
        this.masterGainNode.gain.cancelScheduledValues(now);
        this.masterGainNode.gain.setTargetAtTime(0.0001, now, 0.005);
      } catch (e) {}
    }

    const nodesToStop = [...this.activeSourceNodes];
    this.activeSourceNodes = [];

    // Allow 20ms for gain ramp-down to finish before stopping buffer nodes (zero click/pop)
    setTimeout(() => {
      for (const node of nodesToStop) {
        try {
          node.stop();
          node.disconnect();
        } catch (e) {}
      }
      if (this.audioContext && this.masterGainNode) {
        try {
          this.masterGainNode.gain.cancelScheduledValues(this.audioContext.currentTime);
          this.masterGainNode.gain.setValueAtTime(1.0, this.audioContext.currentTime);
        } catch (e) {}
      }
    }, 25);

    if (this.audioContext) {
      this.nextPlayTime = this.audioContext.currentTime;
    }
  }
}
