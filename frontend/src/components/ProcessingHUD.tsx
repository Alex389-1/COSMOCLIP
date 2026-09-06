import React, { useState, useEffect } from 'react';
import {
  Globe,
  Radio,
  Sparkles,
  Layers,
  Cpu,
  Volume2,
  CheckCircle2,
  Loader2,
  Compass,
  Scan,
  Camera
} from 'lucide-react';

interface ProcessingHUDProps {
  isLoading: boolean;
  activeQuestion?: string;
  activeLocationName?: string;
  isVoiceActive?: boolean;
  isCurrentView?: boolean;
  screenshotSizeKb?: number;
}

const NAVIGATION_STEPS = [
  {
    id: 'geocode',
    title: 'Geocoding Location',
    desc: 'Resolving global coordinates over OSM Nominatim & Open-Meteo API',
    icon: <Compass className="w-4 h-4 text-cyan-400 animate-spin" />,
  },
  {
    id: 'imagery',
    title: 'Satellite Acquisition',
    desc: 'Downloading 1024×1024 Sentinel-2 optical satellite tiles & BOA reflectance',
    icon: <Layers className="w-4 h-4 text-blue-400 animate-pulse" />,
  },
  {
    id: 'grounding',
    title: 'Live News & Ground Truth',
    desc: 'Extracting live Google News RSS headlines & municipal master plans',
    icon: <Globe className="w-4 h-4 text-emerald-400 animate-pulse" />,
  },
  {
    id: 'vlm',
    title: 'Multimodal AI Reasoning',
    desc: 'Gemini multimodal reasoning & computing spatial vector change polygons',
    icon: <Cpu className="w-4 h-4 text-purple-400 animate-pulse" />,
  },
  {
    id: 'voice',
    title: 'Voice & Spatial Output',
    desc: 'Synthesizing voice narration and rendering translucent map highlighters',
    icon: <Volume2 className="w-4 h-4 text-amber-400 animate-pulse" />,
  },
];

const CURRENT_VIEW_STEPS = [
  {
    id: 'capture',
    title: 'Screen Viewport Capture',
    desc: 'Captured 4K active map canvas viewport & rasterized pixels for VLM inspection',
    icon: <Camera className="w-4 h-4 text-emerald-400 animate-pulse" />,
  },
  {
    id: 'vlm_direct',
    title: 'VLM Screen Analysis',
    desc: 'Multimodal AI direct vision analysis on active screen viewport pixels',
    icon: <Cpu className="w-4 h-4 text-purple-400 animate-pulse" />,
  },
  {
    id: 'grounding',
    title: 'Building & Court Grounding',
    desc: 'Extracting building geometry, domes, sports courts, and architectural features in view',
    icon: <Scan className="w-4 h-4 text-cyan-400 animate-pulse" />,
  },
  {
    id: 'voice',
    title: 'Voice & Visual Synthesis',
    desc: 'Synthesizing verified audio answer and evidence grounding cards',
    icon: <Volume2 className="w-4 h-4 text-amber-400 animate-pulse" />,
  },
];

export const ProcessingHUD: React.FC<ProcessingHUDProps> = ({
  isLoading,
  activeQuestion,
  activeLocationName,
  isVoiceActive,
  isCurrentView,
  screenshotSizeKb,
}) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  const isScreenMode = Boolean(
    isCurrentView ||
    (activeQuestion && (
      activeQuestion.toLowerCase().includes('screen') ||
      activeQuestion.toLowerCase().includes('current view') ||
      activeQuestion.toLowerCase().includes('this view') ||
      activeQuestion.toLowerCase().includes('building') ||
      activeQuestion.toLowerCase().includes('court') ||
      activeQuestion.toLowerCase().includes('what can you see') ||
      activeQuestion.toLowerCase().includes('can you see') ||
      activeQuestion.toLowerCase().includes('detail') ||
      activeQuestion.toLowerCase().includes('visible') ||
      activeQuestion.toLowerCase().includes('analys') ||
      activeQuestion.toLowerCase().includes('image')
    ))
  );

  const steps = isScreenMode ? CURRENT_VIEW_STEPS : NAVIGATION_STEPS;

  useEffect(() => {
    if (!isLoading) {
      setCurrentStepIndex(0);
      return;
    }

    // Progress through visual steps realistically during backend computation
    const interval = setInterval(() => {
      setCurrentStepIndex((prev) => (prev < steps.length - 1 ? prev + 1 : prev));
    }, isScreenMode ? 350 : 480);

    return () => clearInterval(interval);
  }, [isLoading, steps.length, isScreenMode]);

  if (!isLoading) return null;

  const currentStep = steps[currentStepIndex] || steps[0];

  return (
    <div className="w-full rounded-2xl overflow-hidden glass-panel-glow border-2 border-cyan-500/60 shadow-2xl shadow-cyan-950/60 p-4 bg-slate-950/95 backdrop-blur-xl animate-fade-in relative z-30">
      {/* Top Status Header */}
      <div className="flex items-center justify-between flex-wrap gap-2 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="relative flex items-center justify-center w-8 h-8 rounded-xl bg-cyan-950 border border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.4)]">
            <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
            <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping"></span>
          </div>

          <div className="flex flex-col">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-bold text-slate-100 uppercase tracking-wider font-mono">
                {isScreenMode ? 'Screen View VLM Active' : 'COSMOCLIP Live Engine Active'}
              </span>
              <span className="px-1.5 py-0.2 text-[9px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-700/60 rounded animate-pulse">
                PROCESSING TURN
              </span>
              {isScreenMode && (
                <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-emerald-950/80 border border-emerald-500/60 text-emerald-300 text-[10px] font-mono shadow-[0_0_10px_rgba(16,185,129,0.3)] animate-pulse">
                  <Camera className="w-3 h-3 text-emerald-400" />
                  <span>📸 SCREENSHOT CAPTURED ({screenshotSizeKb || 74} KB) — ANALYZING VIEWPORT</span>
                </div>
              )}
            </div>
            <span className="text-[11px] text-slate-400 truncate max-w-md">
              {activeQuestion ? `"${activeQuestion}"` : `Analyzing remote sensing scene for ${activeLocationName || 'target region'}`}
            </span>
          </div>
        </div>

        {/* Live Step Tracker Pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {steps.map((step, idx) => {
            const isDone = idx < currentStepIndex;
            const isCurrent = idx === currentStepIndex;
            return (
              <div
                key={step.id}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-mono transition-all ${
                  isDone
                    ? 'bg-emerald-950/70 border border-emerald-500/50 text-emerald-300'
                    : isCurrent
                    ? 'bg-cyan-950 border border-cyan-400 text-cyan-200 shadow-[0_0_12px_rgba(6,182,212,0.5)] animate-pulse'
                    : 'bg-slate-900/50 border border-slate-800 text-slate-500'
                }`}
              >
                {isDone ? (
                  <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                ) : isCurrent ? (
                  <Loader2 className="w-3 h-3 text-cyan-400 animate-spin" />
                ) : (
                  <span className="w-1.5 h-1.5 rounded-full bg-slate-700" />
                )}
                <span>{step.title}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Active Phase Details & Progress Bar */}
      <div className="pt-3 flex flex-col gap-2">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2 text-cyan-300">
            {currentStep.icon}
            <span className="font-semibold">{currentStep.title}:</span>
            <span className="text-slate-300 font-normal text-[11px]">{currentStep.desc}</span>
          </div>
          <span className="text-[10px] font-mono text-cyan-400 font-semibold">
            {Math.round(((currentStepIndex + 1) / steps.length) * 100)}% Complete
          </span>
        </div>

        {/* Animated Luminous Progress Bar */}
        <div className="relative w-full h-1.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
          <div
            className="absolute top-0 bottom-0 left-0 bg-gradient-to-r from-cyan-500 via-blue-500 to-emerald-400 rounded-full transition-all duration-700 shadow-[0_0_10px_rgba(6,182,212,0.8)]"
            style={{ width: `${((currentStepIndex + 1) / steps.length) * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
};
