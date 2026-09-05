import React, { useState, useRef } from 'react';
import { Send, Sparkles, BoxSelect, Radio, Volume2, Mic, MicOff } from 'lucide-react';
import { VoiceState } from '../services/voiceWebSocket';

interface ChatInterfaceProps {
  onSubmitQuery: (question: string, enableGrounding: boolean) => void;
  isLoading: boolean;
  voiceState: VoiceState;
  voiceStatusMessage: string;
  onToggleVoiceSession: () => void;
  activeQuestion: string;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  onSubmitQuery,
  isLoading,
  voiceState,
  voiceStatusMessage,
  onToggleVoiceSession,
  activeQuestion,
}) => {
  const [inputQuery, setInputQuery] = useState('');
  const [enableGrounding, setEnableGrounding] = useState(true);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const recognitionRef = useRef<any>(null);

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || isLoading) return;
    onSubmitQuery(inputQuery.trim(), enableGrounding);
  };

  const toggleMicListening = () => {
    // Connect directly to Gemini Live Voice session (clean bidirectional audio with zero browser earcon beeps)
    onToggleVoiceSession();
  };

  const isVoiceActive = voiceState !== 'disconnected';

  return (
    <div className="glass-panel rounded-2xl p-4 flex flex-col gap-3">
      {/* Live Voice Status Indicator Ribbon */}
      {isVoiceActive ? (
        <div className="flex items-center justify-between bg-cyan-950/80 border border-cyan-500/40 rounded-xl px-3 py-2 text-xs">
          <div className="flex items-center gap-2">
            {voiceState === 'listening' ? (
              <div className="flex items-center gap-0.5 px-1">
                <span className="wave-bar"></span>
                <span className="wave-bar"></span>
                <span className="wave-bar"></span>
              </div>
            ) : voiceState === 'speaking' ? (
              <Volume2 className="w-4 h-4 text-cyan-400 animate-bounce" />
            ) : (
              <Radio className="w-4 h-4 text-emerald-400 animate-pulse" />
            )}
            <span className="text-cyan-200 font-medium font-mono text-[11px]">
              {voiceStatusMessage || 'Real-Time Voice Channel Active (Speak Anytime)'}
            </span>
          </div>

          <button
            type="button"
            onClick={onToggleVoiceSession}
            className="text-[10px] font-mono text-rose-300 hover:text-rose-100 bg-rose-950/80 px-2 py-0.5 rounded border border-rose-800/60"
          >
            Disconnect Voice
          </button>
        </div>
      ) : (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Natural Language & Real-Time Voice
            </span>
          </div>

          <label className="flex items-center gap-1.5 cursor-pointer select-none text-xs text-slate-400 hover:text-slate-200">
            <input
              type="checkbox"
              checked={enableGrounding}
              onChange={(e) => setEnableGrounding(e.target.checked)}
              className="w-3.5 h-3.5 rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0 cursor-pointer accent-cyan-500"
            />
            <BoxSelect className="w-3.5 h-3.5 text-cyan-400" />
            <span>Spatial Grounding & Change Overlays</span>
          </label>
        </div>
      )}

      {/* Single Unified Input Bar with Direct Mic Button */}
      <form onSubmit={handleFormSubmit} className="relative flex items-center gap-2">
        <div className="relative flex-1">
          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            placeholder={
              isRecognizing
                ? '🎙️ Listening to your voice right now... Speak clearly...'
                : isVoiceActive
                ? '🎙️ Gemini Live voice active: Speak freely or type query here...'
                : 'Ask any question (e.g. "Building expansion around red fort delhi india") or click mic...'
            }
            disabled={isLoading}
            className={`w-full bg-slate-900/90 border rounded-xl pl-4 pr-16 py-3 text-sm text-slate-100 placeholder-slate-400 focus:outline-none focus:border-cyan-400 shadow-inner ${
              isRecognizing
                ? 'border-rose-500 ring-2 ring-rose-500/40'
                : isVoiceActive
                ? 'border-cyan-500/60 ring-1 ring-cyan-500/30'
                : 'border-slate-700/80'
            }`}
          />
          
          <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
            {inputQuery && (
              <button
                type="button"
                onClick={() => setInputQuery('')}
                className="text-slate-500 hover:text-slate-300 text-xs px-1"
              >
                ×
              </button>
            )}

            {/* Quick Mic Speech-to-Text Button */}
            <button
              type="button"
              onClick={toggleMicListening}
              title={isRecognizing ? 'Listening... Click to stop' : 'Click to speak query'}
              className={`p-1.5 rounded-lg transition-all ${
                isRecognizing
                  ? 'bg-rose-600 text-white animate-pulse shadow-md shadow-rose-600/40'
                  : 'text-slate-400 hover:text-cyan-300 hover:bg-slate-800'
              }`}
            >
              {isRecognizing ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Submit / Ask Button */}
        <button
          type="submit"
          disabled={!inputQuery.trim() || isLoading}
          className="flex items-center gap-2 px-5 py-3 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium text-sm transition-all shadow-lg shadow-cyan-600/25 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
        >
          {isLoading ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <span>Ask</span>
              <Send className="w-4 h-4" />
            </>
          )}
        </button>
      </form>
    </div>
  );
};
