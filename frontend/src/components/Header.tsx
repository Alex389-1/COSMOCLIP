import React, { useState } from 'react';
import { Satellite, Cpu, Radio, Sparkles, Database, Search, MapPin, Mic, MicOff, Volume2, Activity } from 'lucide-react';
import { VoiceState } from '../services/voiceWebSocket';

interface HeaderProps {
  onSearchLocation: (query: string) => void;
  onOpenRegistry: () => void;
  voiceState: VoiceState;
  voiceStatusMessage: string;
  activeLocationName: string;
  onToggleVoiceSession: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  onSearchLocation,
  onOpenRegistry,
  voiceState,
  voiceStatusMessage,
  activeLocationName,
  onToggleVoiceSession,
}) => {
  const [searchInput, setSearchInput] = useState('');

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchInput.trim()) return;
    onSearchLocation(searchInput.trim());
    setSearchInput('');
  };

  const isSessionActive = voiceState !== 'disconnected';

  return (
    <header className="glass-panel border-b border-slate-800/80 px-6 py-3.5 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-40">
      {/* Brand & ISRO Badge */}
      <div className="flex items-center gap-3.5">
        <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-600 shadow-lg shadow-cyan-500/20 ring-1 ring-cyan-400/40">
          <Satellite className="w-5 h-5 text-white" />
          <span className="absolute -bottom-0.5 -right-0.5 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
          </span>
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-cyan-300 bg-clip-text text-transparent">
              COSMOCLIP
            </h1>
            <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase rounded-md bg-cyan-950/80 border border-cyan-500/30 text-cyan-300">
              ISRO PS 26167
            </span>
            <span className="hidden sm:inline-flex px-2 py-0.5 text-[10px] font-mono rounded-md bg-emerald-950/80 border border-emerald-500/30 text-emerald-300">
              Real-Time WebSocket
            </span>
          </div>
          <p className="text-xs text-slate-400 font-normal">
            Real-Time Voice-Driven Remote Sensing & Bi-Temporal Change Assistant
          </p>
        </div>
      </div>

      {/* Center Dynamic Location Geocoder Search Bar */}
      <form onSubmit={handleSearchSubmit} className="flex items-center gap-2 max-w-sm w-full">
        <div className="relative w-full">
          <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-cyan-400" />
          <input
            type="text"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            placeholder={`Location: ${activeLocationName || 'e.g. Mumbai Port, Chennai, Dubai...'}`}
            className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl pl-9 pr-3 py-1.5 text-xs text-slate-100 placeholder-slate-400 focus:outline-none focus:border-cyan-400"
          />
        </div>
        <button
          type="submit"
          className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 border border-slate-700 hover:border-cyan-500/40 shrink-0"
        >
          Go
        </button>
      </form>

      {/* Right Controls & Single Live Voice Button */}
      <div className="flex items-center flex-wrap gap-2.5">
        {/* Single Unified Real-Time Live Voice Chat Button */}
        <button
          onClick={onToggleVoiceSession}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium text-xs transition-all shadow-lg ${
            isSessionActive
              ? voiceState === 'listening'
                ? 'bg-rose-600 border border-rose-400 text-white shadow-rose-600/30 animate-pulse'
                : voiceState === 'speaking'
                ? 'bg-cyan-600 border border-cyan-400 text-white shadow-cyan-600/30'
                : 'bg-emerald-600 border border-emerald-400 text-white shadow-emerald-600/30'
              : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-cyan-600/25 border border-cyan-400/40'
          }`}
        >
          {isSessionActive ? (
            <>
              {voiceState === 'listening' ? (
                <div className="flex items-center gap-0.5">
                  <span className="wave-bar"></span>
                  <span className="wave-bar"></span>
                  <span className="wave-bar"></span>
                </div>
              ) : voiceState === 'speaking' ? (
                <Volume2 className="w-4 h-4 animate-bounce" />
              ) : (
                <Activity className="w-4 h-4 animate-spin" />
              )}
              <span>{voiceState === 'listening' ? 'Live Voice Active (Listening)' : voiceState === 'speaking' ? 'Speaking Answer...' : 'Connected (WS)'}</span>
              <span className="ml-1 text-[10px] bg-black/30 px-1.5 py-0.2 rounded font-mono">End</span>
            </>
          ) : (
            <>
              <Mic className="w-4 h-4 text-white" />
              <span>Start Live Voice Chat</span>
            </>
          )}
        </button>

        {/* Model Spec Badge */}
        <div className="hidden xl:flex items-center gap-1.5 bg-slate-900/60 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-slate-400">Model:</span>
          <span className="font-mono text-emerald-300 font-medium">RS-LLaVA-7B</span>
        </div>

        {/* Tool Registry Button */}
        <button
          onClick={onOpenRegistry}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-xs font-medium text-slate-200 transition-all hover:border-cyan-500/40"
        >
          <Database className="w-3.5 h-3.5 text-cyan-400" />
          <span>Tools</span>
        </button>
      </div>
    </header>
  );
};
