import React, { useEffect, useState } from 'react';
import { X, Database, Cpu, CheckCircle, ShieldCheck, Sparkles, Terminal } from 'lucide-react';
import { ToolCapability } from '../types';
import { fetchModelRegistry } from '../services/api';

interface RegistryModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const RegistryModal: React.FC<RegistryModalProps> = ({ isOpen, onClose }) => {
  const [tools, setTools] = useState<ToolCapability[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!isOpen) return;
    setIsLoading(true);
    fetchModelRegistry()
      .then((data) => setTools(data.tools))
      .catch((err) => console.error(err))
      .finally(() => setIsLoading(false));
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="glass-panel w-full max-w-2xl rounded-2xl p-6 flex flex-col gap-5 border border-cyan-500/30 shadow-2xl shadow-cyan-950/50 max-h-[85vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-cyan-950/80 border border-cyan-500/40 text-cyan-400">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100">
                COSMOCLIP Model & Tool Registry
              </h2>
              <p className="text-xs text-slate-400">
                Typed capability contracts & specialist remote sensing interfaces
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tools List */}
        {isLoading ? (
          <div className="py-12 text-center text-slate-400 text-xs font-mono animate-pulse">
            Loading tool schemas...
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {tools.map((t) => (
              <div
                key={t.tool_id}
                className="bg-slate-900/80 rounded-xl p-4 border border-slate-800 flex flex-col gap-2.5 hover:border-slate-700 transition-all"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-cyan-400" />
                    <span className="font-semibold text-sm text-slate-200">{t.name}</span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                      v{t.version}
                    </span>
                  </div>
                  <span className="flex items-center gap-1 text-[11px] font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 rounded">
                    <CheckCircle className="w-3 h-3" />
                    {t.runtime}
                  </span>
                </div>

                <p className="text-xs text-slate-400 leading-relaxed font-normal">
                  {t.description}
                </p>

                <div className="flex items-center gap-4 text-[11px] font-mono text-slate-400 pt-1 border-t border-slate-800/60">
                  <div>
                    <span className="text-slate-500">Tasks: </span>
                    <span className="text-cyan-300">{t.task_types.join(', ')}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Modalities: </span>
                    <span className="text-slate-300">{t.modalities.join(', ')}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Outputs: </span>
                    <span className="text-emerald-300">{t.outputs.join(', ')}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Footer */}
        <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-xs text-slate-500">
          <span>Contract: Typed Model Registry v0.1.0</span>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium transition-all"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
