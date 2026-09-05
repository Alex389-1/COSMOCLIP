import React from 'react';
import { Volume2, VolumeX, ShieldCheck, GitCompare, Info, CheckCircle2, AlertTriangle, Sparkles, BoxSelect, Activity } from 'lucide-react';
import { QueryResponse } from '../types';

interface AnswerCardProps {
  response: QueryResponse | null;
  isLoading: boolean;
  isSpeaking: boolean;
  onToggleSpeak: () => void;
}

export const AnswerCard: React.FC<AnswerCardProps> = ({
  response,
  isLoading,
  isSpeaking,
  onToggleSpeak,
}) => {
  if (isLoading) {
    return (
      <div className="glass-panel-glow rounded-2xl p-6 flex flex-col gap-4 animate-pulse">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-cyan-400 animate-ping" />
            <span className="text-xs font-mono text-cyan-300">RS-LLaVA VLM Specialist & Change Engine Analyzing...</span>
          </div>
          <div className="h-4 w-24 bg-slate-800 rounded"></div>
        </div>
        <div className="h-6 w-3/4 bg-slate-800 rounded"></div>
        <div className="h-16 w-full bg-slate-800/60 rounded-xl"></div>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="glass-panel rounded-2xl p-6 flex flex-col items-center justify-center text-center text-slate-400 min-h-[160px] border border-dashed border-slate-800">
        <Sparkles className="w-8 h-8 text-cyan-400/60 mb-2 animate-pulse" />
        <p className="text-sm font-semibold text-slate-200">Voice-Driven Satellite Intelligence Ready</p>
        <p className="text-xs text-slate-400 mt-1 max-w-md">
          Press the microphone button or ask: <em>"Tell me the changes around Mumbai Port"</em> to view real-time bi-temporal satellite analysis and listen to the spoken answer.
        </p>
      </div>
    );
  }

  const confidencePct = Math.round(response.confidence.score * 100);
  const isHighConfidence = response.confidence.category === 'High';

  return (
    <div className={`glass-panel rounded-2xl p-5 flex flex-col gap-4 shadow-xl border-l-4 ${
      response.is_comparison ? 'border-l-emerald-500' : 'border-l-cyan-500'
    }`}>
      {/* Header Bar with Task & Confidence */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase border ${
            response.is_comparison
              ? 'bg-emerald-950/90 text-emerald-300 border-emerald-700/60'
              : 'bg-cyan-950/90 text-cyan-300 border-cyan-700/60'
          }`}>
            {response.is_comparison ? 'TASK: BI-TEMPORAL CHANGE' : `TASK: ${response.task.toUpperCase()}`}
          </span>
          {response.target_entity && (
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300 border border-slate-700">
              Target: {response.target_entity}
            </span>
          )}
        </div>

        {/* Confidence Gauge Badge */}
        <div className="flex items-center gap-2 bg-slate-900/90 border border-slate-700/80 px-3 py-1 rounded-xl">
          {isHighConfidence ? (
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          )}
          <div className="flex flex-col">
            <span className="text-[10px] text-slate-400 leading-none">Model Confidence</span>
            <span className="text-xs font-mono font-bold text-slate-100">
              {confidencePct}% ({response.confidence.category})
            </span>
          </div>
        </div>
      </div>

      {/* Primary Answer Output */}
      <div className="text-slate-100 text-sm leading-relaxed font-normal bg-slate-900/70 rounded-xl p-4 border border-slate-800/80 shadow-inner">
        {response.answer}
      </div>

      {/* Spoken Audio Banner & Replay Button */}
      <div className="flex items-center justify-between flex-wrap gap-3 pt-1 border-t border-slate-800/80 text-xs">
        <div className="flex items-center gap-2 text-slate-300">
          <div className={`p-1.5 rounded-lg border ${
            isSpeaking ? 'bg-cyan-950 border-cyan-500 text-cyan-400 animate-pulse' : 'bg-slate-800 border-slate-700 text-slate-400'
          }`}>
            <Volume2 className="w-3.5 h-3.5" />
          </div>
          <span className="text-[11px] text-slate-300 italic">
            {isSpeaking ? 'Speaking response aloud...' : 'Voice narration ready'}
          </span>
        </div>

        {/* Spoken Output Replay Button */}
        <button
          onClick={onToggleSpeak}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${
            isSpeaking
              ? 'bg-cyan-600 text-white border-cyan-400 shadow-md shadow-cyan-500/20'
              : 'bg-slate-800/80 hover:bg-slate-700 text-slate-200 border-slate-700'
          }`}
        >
          {isSpeaking ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5 text-cyan-400" />}
          <span>{isSpeaking ? 'Mute Voice' : 'Replay Voice'}</span>
        </button>
      </div>

      {/* Real-World Ground Truth & Live News Intelligence Feed */}
      {response.ground_truth_context && (
        <div className="bg-slate-900/90 border border-blue-500/30 rounded-xl p-3.5 flex flex-col gap-2 shadow-md">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-blue-300">
              <Sparkles className="w-3.5 h-3.5 text-blue-400" />
              <span>Real-World Ground Truth & Google News Intelligence</span>
            </div>
            {response.ground_truth_context.sources && response.ground_truth_context.sources.length > 0 && (
              <div className="flex items-center gap-1">
                {response.ground_truth_context.sources.map((src, idx) => (
                  <span
                    key={idx}
                    className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-blue-950/80 border border-blue-800/60 text-blue-200"
                  >
                    {src}
                  </span>
                ))}
              </div>
            )}
          </div>

          <p className="text-xs text-slate-300 leading-relaxed font-normal">
            {response.ground_truth_context.summary}
          </p>

          {response.ground_truth_context.headlines && response.ground_truth_context.headlines.length > 0 && (
            <div className="flex flex-col gap-1 pt-1 border-t border-slate-800">
              <span className="text-[10px] uppercase font-mono text-slate-400">Live News & Project Context:</span>
              <div className="flex flex-wrap gap-1.5">
                {response.ground_truth_context.headlines.map((headline, idx) => (
                  <span
                    key={idx}
                    className="text-[10px] text-slate-300 bg-slate-800/80 border border-slate-700/80 px-2 py-0.5 rounded-md flex items-center gap-1"
                  >
                    <Info className="w-3 h-3 text-cyan-400 shrink-0" />
                    <span>{headline}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Grounded Evidence Regions with Category Color Themes */}
      {response.evidence.length > 0 && (
        <div className="flex items-center gap-2 flex-wrap text-xs pt-1">
          <span className="text-slate-400 text-[11px] font-medium flex items-center gap-1">
            <BoxSelect className="w-3.5 h-3.5 text-cyan-400" />
            Detected Regions:
          </span>
          {response.evidence.map((ev, i) => {
            const isRoad = ev.change_type === 'road' || ev.label.toLowerCase().includes('road') || ev.label.toLowerCase().includes('expressway');
            const isBuilding = ev.change_type === 'building' || ev.label.toLowerCase().includes('building') || ev.label.toLowerCase().includes('terminal') || ev.label.toLowerCase().includes('warehouse');
            const isForest = ev.change_type === 'forest' || ev.label.toLowerCase().includes('forest') || ev.label.toLowerCase().includes('plant') || ev.label.toLowerCase().includes('canopy');
            const isWater = ev.change_type === 'water' || ev.label.toLowerCase().includes('water') || ev.label.toLowerCase().includes('berth') || ev.label.toLowerCase().includes('vessel');

            let chipStyle = 'bg-cyan-950/80 text-cyan-300 border-cyan-700/60';
            if (isRoad) chipStyle = 'bg-amber-950/80 text-amber-300 border-amber-500/60';
            else if (isBuilding) chipStyle = 'bg-orange-950/80 text-orange-300 border-orange-500/60';
            else if (isForest) chipStyle = 'bg-emerald-950/80 text-emerald-300 border-emerald-500/60';
            else if (isWater) chipStyle = 'bg-cyan-950/80 text-cyan-300 border-cyan-500/60';

            return (
              <span
                key={i}
                className={`px-2 py-0.5 rounded-md border text-[11px] font-mono flex items-center gap-1 shadow-sm ${chipStyle}`}
              >
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                {ev.label}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
};
