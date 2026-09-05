import React, { useState, useEffect, useRef } from 'react';
import {
  Terminal,
  Trash2,
  Copy,
  Check,
  Search,
  ArrowDown,
  Play,
  Pause,
  Wifi,
  WifiOff
} from 'lucide-react';

export interface LogEntry {
  timestamp: string;
  level: 'INFO' | 'SUCCESS' | 'WARNING' | 'ERROR' | 'DEBUG';
  category: string;
  message: string;
  meta?: Record<string, any>;
}

export const LiveLogsViewer: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [filterCategory, setFilterCategory] = useState<string>('ALL');
  const [filterLevel, setFilterLevel] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [autoScroll, setAutoScroll] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);

  const logsEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const lastTimestampRef = useRef<string>('');

  // Real-Time WebSocket Connection with Auto-Reconnect & Polling Fallback
  useEffect(() => {
    let isUnmounted = false;
    let reconnectTimeout: any = null;
    let pingInterval: any = null;
    let pollInterval: any = null;

    const connectWebSocket = () => {
      if (isUnmounted) return;

      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/logs/ws`;
        
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (isUnmounted) return;
          setIsConnected(true);
          // Ping keepalive every 10 seconds
          pingInterval = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send('ping');
            }
          }, 10000);
        };

        ws.onmessage = (event) => {
          if (isUnmounted) return;
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'history' && Array.isArray(msg.data)) {
              setLogs(msg.data);
              if (msg.data.length > 0) {
                lastTimestampRef.current = msg.data[msg.data.length - 1].timestamp;
              }
            } else if (msg.type === 'log' && msg.data) {
              const entry: LogEntry = msg.data;
              lastTimestampRef.current = entry.timestamp;
              setLogs((prev) => {
                const updated = [...prev, entry];
                return updated.length > 400 ? updated.slice(-400) : updated;
              });
            } else if (msg.timestamp && msg.message) {
              // Direct log entry
              const entry: LogEntry = msg;
              lastTimestampRef.current = entry.timestamp;
              setLogs((prev) => {
                const updated = [...prev, entry];
                return updated.length > 400 ? updated.slice(-400) : updated;
              });
            }
          } catch (e) {
            console.error('Error parsing WS log event:', e);
          }
        };

        ws.onerror = () => {
          if (isUnmounted) return;
          setIsConnected(false);
        };

        ws.onclose = () => {
          if (isUnmounted) return;
          setIsConnected(false);
          if (pingInterval) clearInterval(pingInterval);
          // Auto reconnect after 2 seconds
          reconnectTimeout = setTimeout(connectWebSocket, 2000);
        };
      } catch (err) {
        setIsConnected(false);
        reconnectTimeout = setTimeout(connectWebSocket, 3000);
      }
    };

    connectWebSocket();

    // Secondary Polling Fallback (if websocket is disconnected or blocked by proxy)
    pollInterval = setInterval(async () => {
      if (isUnmounted) return;
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        try {
          const url = lastTimestampRef.current 
            ? `/api/logs/recent?since=${encodeURIComponent(lastTimestampRef.current)}`
            : '/api/logs/recent';
          const res = await fetch(url);
          if (res.ok) {
            const data = await res.json();
            if (data.logs && data.logs.length > 0) {
              lastTimestampRef.current = data.logs[data.logs.length - 1].timestamp;
              setLogs((prev) => {
                const existingKeys = new Set(prev.map(p => `${p.timestamp}-${p.message}`));
                const newEntries = data.logs.filter((l: LogEntry) => !existingKeys.has(`${l.timestamp}-${l.message}`));
                if (newEntries.length === 0) return prev;
                const updated = [...prev, ...newEntries];
                return updated.length > 400 ? updated.slice(-400) : updated;
              });
              setIsConnected(true);
            }
          }
        } catch (e) {
          // ignore polling errors
        }
      }
    }, 1000);

    return () => {
      isUnmounted = true;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (pingInterval) clearInterval(pingInterval);
      if (pollInterval) clearInterval(pollInterval);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Auto-scroll to bottom on new log entries
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  // Handle user manual scroll
  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const atBottom = scrollHeight - scrollTop - clientHeight < 40;
    setAutoScroll(atBottom);
  };

  const handleClear = () => {
    setLogs([]);
  };

  const handleCopy = () => {
    const text = filteredLogs
      .map((l) => `[${l.timestamp}] [${l.level}] [${l.category}] ${l.message}`)
      .join('\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Filter logs based on category, level, and search text
  const filteredLogs = logs.filter((log) => {
    if (filterCategory !== 'ALL' && log.category !== filterCategory) return false;
    if (filterLevel !== 'ALL' && log.level !== filterLevel) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return (
        log.message.toLowerCase().includes(q) ||
        log.category.toLowerCase().includes(q) ||
        log.timestamp.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const getCategoryColor = (cat: string) => {
    switch (cat) {
      case 'LANGGRAPH':
        return 'bg-cyan-950/80 text-cyan-300 border-cyan-700/60';
      case 'GEOCODER':
        return 'bg-emerald-950/80 text-emerald-300 border-emerald-700/60';
      case 'SENTINEL-2':
        return 'bg-sky-950/80 text-sky-300 border-sky-700/60';
      case 'SAR-RADAR':
        return 'bg-purple-950/80 text-purple-300 border-purple-700/60';
      case 'MISTRAL-AI':
        return 'bg-amber-950/80 text-amber-300 border-amber-700/60';
      case 'GEMINI-LIVE':
        return 'bg-indigo-950/80 text-indigo-300 border-indigo-700/60';
      case 'WEB-INTEL':
        return 'bg-teal-950/80 text-teal-300 border-teal-700/60';
      case 'STAC-API':
        return 'bg-blue-950/80 text-blue-300 border-blue-700/60';
      default:
        return 'bg-slate-900 text-slate-400 border-slate-700';
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'SUCCESS':
        return 'text-emerald-400 font-bold';
      case 'WARNING':
        return 'text-amber-400 font-bold';
      case 'ERROR':
        return 'text-rose-400 font-bold';
      case 'DEBUG':
        return 'text-slate-500';
      default:
        return 'text-cyan-400';
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-3.5 flex flex-col gap-2.5 transition-all bg-slate-950/90 border border-slate-800 shadow-2xl">
      {/* Top Controls Bar */}
      <div className="flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-slate-900 border border-slate-800 text-[11px] font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'
              }`}
            />
            <span className="text-slate-200 font-semibold tracking-wide">
              {isConnected ? 'LIVE WEBSOCKET' : 'CONNECTING...'}
            </span>
          </div>
          <span className="text-[10px] font-mono text-slate-400">
            {filteredLogs.length} events
          </span>
        </div>

        {/* Quick Action Buttons */}
        <div className="flex items-center gap-1.5">
          {/* Search Input */}
          <div className="relative flex items-center">
            <Search className="w-3 h-3 text-slate-500 absolute left-2 pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter logs..."
              className="pl-6 pr-2 py-0.5 text-[11px] font-mono bg-slate-900 border border-slate-800 rounded-lg text-slate-200 focus:outline-none focus:border-cyan-500/50 w-28 md:w-36 transition-all"
            />
          </div>

          {/* Auto Scroll Toggle */}
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`p-1 rounded-lg border text-xs font-mono transition-all flex items-center gap-1 ${
              autoScroll
                ? 'bg-cyan-950/80 text-cyan-300 border-cyan-800/60'
                : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
            title={autoScroll ? 'Auto-scroll enabled' : 'Auto-scroll paused'}
          >
            {autoScroll ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />}
          </button>

          {/* Copy Logs */}
          <button
            onClick={handleCopy}
            className="p-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 transition-all"
            title="Copy logs to clipboard"
          >
            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
          </button>

          {/* Clear Logs */}
          <button
            onClick={handleClear}
            className="p-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-rose-400 transition-all"
            title="Clear current log view"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Category Pills Bar (Flex wrap without ugly scrollbar) */}
      <div className="flex flex-wrap items-center gap-1.5 text-[10px] font-mono py-0.5">
        {['ALL', 'LANGGRAPH', 'GEOCODER', 'SENTINEL-2', 'SAR-RADAR', 'MISTRAL-AI', 'GEMINI-LIVE', 'WEB-INTEL'].map(
          (cat) => (
            <button
              key={cat}
              onClick={() => setFilterCategory(cat)}
              className={`px-2 py-0.5 rounded-md border transition-all ${
                filterCategory === cat
                  ? 'bg-cyan-600 text-white font-bold border-cyan-400 shadow-sm'
                  : 'bg-slate-900/90 text-slate-400 border-slate-800 hover:text-slate-200 hover:border-slate-700'
              }`}
            >
              {cat}
            </button>
          )
        )}
      </div>

      {/* Terminal Viewport */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="w-full h-[260px] max-h-[300px] overflow-y-auto bg-slate-950/95 rounded-xl p-2.5 border border-slate-900 font-mono text-[11px] leading-relaxed flex flex-col gap-1 select-text scrollbar-thin scrollbar-thumb-slate-800"
      >
        {filteredLogs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-slate-500 gap-2 select-none">
            <Terminal className="w-6 h-6 animate-pulse text-cyan-500/50" />
            <span className="text-xs">Awaiting live backend agent events & telemetry...</span>
          </div>
        ) : (
          filteredLogs.map((log, index) => (
            <div
              key={index}
              className="flex items-start gap-2 hover:bg-slate-900/60 px-1 py-0.5 rounded transition-colors group"
            >
              {/* Timestamp */}
              <span className="text-slate-500 text-[10px] shrink-0 font-mono">
                {log.timestamp}
              </span>

              {/* Category Badge */}
              <span
                className={`text-[9px] px-1.5 py-0.2 rounded border uppercase font-bold shrink-0 ${getCategoryColor(
                  log.category
                )}`}
              >
                {log.category}
              </span>

              {/* Log Level */}
              <span className={`text-[10px] shrink-0 uppercase font-mono ${getLevelColor(log.level)}`}>
                [{log.level}]
              </span>

              {/* Log Message */}
              <span className="text-slate-300 break-all flex-1">{log.message}</span>
            </div>
          ))
        )}
        <div ref={logsEndRef} />
      </div>

      {/* Bottom Status */}
      <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 pt-1 border-t border-slate-900">
        <div className="flex items-center gap-2">
          <span>Buffer: {logs.length}/400 events</span>
          <span>•</span>
          <span>Protocol: WebSocket (/api/logs/ws)</span>
        </div>
        {!autoScroll && (
          <button
            onClick={() => {
              setAutoScroll(true);
              if (logsEndRef.current) logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
            }}
            className="text-cyan-400 hover:text-cyan-200 flex items-center gap-1 font-semibold"
          >
            <span>Scroll to bottom</span>
            <ArrowDown className="w-3 h-3 animate-bounce" />
          </button>
        )}
      </div>
    </div>
  );
};

