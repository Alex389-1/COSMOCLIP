import React, { useState } from 'react';
import { GitBranch, Clock, CheckCircle2, AlertTriangle, XCircle, ChevronDown, ChevronUp, Activity } from 'lucide-react';
import { TraceStep } from '../types';

interface ExecutionTraceProps {
  trace: TraceStep[];
  runId?: string;
}

export const ExecutionTrace: React.FC<ExecutionTraceProps> = ({ trace, runId }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!trace || trace.length === 0) return null;

  const totalLatency = trace.reduce((acc, step) => acc + (step.latency_ms || 0), 0);

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'ok':
        return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />;
      case 'warning':
        return <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />;
      case 'error':
        return <XCircle className="w-3.5 h-3.5 text-rose-400" />;
      default:
        return <Activity className="w-3.5 h-3.5 text-cyan-400" />;
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-4 flex flex-col gap-3 transition-all">
      {/* Header with expand toggle */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between cursor-pointer select-none"
      >
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            LangGraph Execution Trace
          </span>
          {runId && (
            <span className="text-[10px] font-mono text-slate-500 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
              {runId}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 text-[11px] font-mono text-cyan-300 bg-cyan-950/60 border border-cyan-800/40 px-2 py-0.5 rounded">
            <Clock className="w-3 h-3 text-cyan-400" />
            <span>{totalLatency.toFixed(1)} ms total</span>
          </div>
          <button className="text-slate-400 hover:text-slate-200">
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Expanded Timeline Steps */}
      {isExpanded && (
        <div className="flex flex-col gap-2 pt-2 border-t border-slate-800/80">
          {trace.map((step, idx) => (
            <div
              key={idx}
              className="flex items-start gap-3 bg-slate-900/60 rounded-xl p-2.5 border border-slate-800/80 text-xs"
            >
              <div className="mt-0.5 shrink-0">{getStepIcon(step.status)}</div>
              <div className="flex-1 flex flex-col gap-0.5">
                <div className="flex items-center justify-between">
                  <span className="font-mono font-semibold text-slate-200">
                    {step.step}
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">
                    +{step.latency_ms.toFixed(1)} ms
                  </span>
                </div>
                {step.detail && (
                  <p className="text-[11px] text-slate-400 font-normal leading-normal">
                    {step.detail}
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
