import React, { useEffect, useState, useRef } from 'react';
import { Header } from './components/Header';
import { MapViewer } from './components/MapViewer';
import { ImageStage } from './components/ImageStage';
import { ComparisonStage } from './components/ComparisonStage';
import { GlobalEarthStage } from './components/GlobalEarthStage';
import { ChatInterface } from './components/ChatInterface';
import { AnswerCard } from './components/AnswerCard';
import { ExecutionTrace } from './components/ExecutionTrace';
import { LiveLogsViewer } from './components/LiveLogsViewer';
import { RegistryModal } from './components/RegistryModal';
import { ProcessingHUD } from './components/ProcessingHUD';
import { QueryResponse, BoundingBox } from './types';
import { submitVQAQuery, geocodeLocation } from './services/api';
import { VoiceWebSocketClient, VoiceState, ToolCallEvent, ToolStartEvent } from './services/voiceWebSocket';
import { Map as MapIcon, Image as ImageIcon, Globe2, Radio, Sparkles, GitCompare, Layers, AlertCircle, RefreshCw, Terminal, GitBranch } from 'lucide-react';

export const App: React.FC = () => {
  const [queryResponse, setQueryResponse] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [activeQuestion, setActiveQuestion] = useState<string>('');
  const [isRegistryOpen, setIsRegistryOpen] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'global' | 'optical' | 'sar' | 'fused' | 'compare' | 'map'>('global');
  const [rightPanelTab, setRightPanelTab] = useState<'trace' | 'logs'>('trace');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);

  // Real-Time Gemini Live Voice State
  const [voiceState, setVoiceState] = useState<VoiceState>('disconnected');
  const [voiceStatusMessage, setVoiceStatusMessage] = useState<string>('');

  // Active Location & Map Coordinates State (Default: Global Earth Grid)
  const [currentLocationName, setCurrentLocationName] = useState<string>('Global Earth Observation Grid');
  const [centerCoords, setCenterCoords] = useState<{ lat: number; lng: number; zoom: number }>({
    lat: 20.0,
    lng: 30.0,
    zoom: 3,
  });
  const [currentBbox, setCurrentBbox] = useState<[number, number, number, number]>([-180, -60, 180, 60]);
  const [activeImageUrl, setActiveImageUrl] = useState<string>('');

  const voiceClientRef = useRef<VoiceWebSocketClient | null>(null);

  // Initialize Voice Client
  useEffect(() => {
    // Instantiate persistent Gemini Live WebSocket client
    voiceClientRef.current = new VoiceWebSocketClient(
      (state, message) => {
        setVoiceState(state);
        if (message) setVoiceStatusMessage(message);
        if (state === 'processing') setIsLoading(true);
        if (state === 'speaking' || state === 'listening') setIsLoading(false);
      },
      (toolEvent: ToolCallEvent) => {
        handleToolCallEvent(toolEvent);
      },
      (startEvent: ToolStartEvent) => {
        setIsLoading(true);
        const loc = startEvent.location || startEvent.args?.location || '';
        const q = startEvent.question || startEvent.args?.question || '';
        if (loc) {
          setCurrentLocationName(loc);
        }
        if (q || loc) {
          setActiveQuestion(q || `Show satellite imagery for ${loc}`);
        }
      }
    );

    return () => {
      if (voiceClientRef.current) {
        voiceClientRef.current.stopSession();
      }
    };
  }, []);

  // Interactive Zoom & Pan State from Voice Tool Commands
  const [stageZoom, setStageZoom] = useState<number>(1);
  const [stagePan, setStagePan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Handle Gemini Live Real-Time Tool Calls
  const handleToolCallEvent = async (event: ToolCallEvent) => {
    // 1. Handle Voice Map Zoom & Pan Tool Calls
    if (event.name === 'zoom_map') {
      const z = typeof event.args?.zoom_level === 'number' ? event.args.zoom_level : 2.2;
      const target = (event.args?.target_area || '').toLowerCase();
      const panDir = (event.args?.pan_direction || '').toLowerCase();

      let px = 0, py = 0;
      if (panDir.includes('left') || target.includes('west') || target.includes('left')) px = 80;
      if (panDir.includes('right') || target.includes('east') || target.includes('right')) px = -80;
      if (panDir.includes('top') || panDir.includes('north') || target.includes('top') || target.includes('north')) py = 80;
      if (panDir.includes('bottom') || panDir.includes('south') || target.includes('bottom') || target.includes('south')) py = -80;

      setStageZoom(z);
      setStagePan({ x: px, y: py });

      if (z > 1.0) {
        setCenterCoords((prev) => ({ ...prev, zoom: Math.min(18, prev.zoom + 2) }));
      } else {
        setCenterCoords((prev) => ({ ...prev, zoom: 14 }));
      }
      return;
    }

    const loc = event.args?.location || event.args?.query || '';
    const q = event.args?.question || '';

    // If location is detected in voice arguments, update location state immediately
    if (loc && typeof loc === 'string' && loc.length > 2 && !event.payload) {
      geocodeLocation(loc).then((geo) => {
        setCurrentLocationName(geo.name);
        setCenterCoords({ lat: geo.lat, lng: geo.lon, zoom: geo.zoom || 14 });
        setCurrentBbox(geo.bbox);
        if (geo.image_url) setActiveImageUrl(geo.image_url);
        if (event.name === 'compare_satellite_images') {
          setActiveTab('compare');
        }
      }).catch(() => {});
    }

    // If backend provided full structured payload from CosmoClipAgentController
    if (event.payload) {
      const p = event.payload;
      setQueryResponse(p);
      if (q || loc) {
        setActiveQuestion(q || (event.name === 'compare_satellite_images' ? `Compare satellite changes around ${loc}` : `Satellite analysis for ${loc}`));
      }
      if (p.location_meta) {
        setCurrentLocationName(p.location_meta.name);
        setCenterCoords({
          lat: p.location_meta.lat,
          lng: p.location_meta.lon,
          zoom: p.location_meta.zoom || 14,
        });
        setCurrentBbox(p.location_meta.bbox);
      }
      if (p.image_url) {
        setActiveImageUrl(p.image_url);
      }
      if (p.is_comparison || event.name === 'compare_satellite_images') {
        setActiveTab('compare');
      } else {
        setActiveTab('optical');
      }
      return;
    }

    // Fallback if payload is raw tool call: query agent controller directly
    const isCompTool = event.name === 'compare_satellite_images';
    const targetQuery = q || (isCompTool
      ? (loc ? `Compare satellite changes around ${loc} between 2020 and 2026` : 'Compare satellite changes')
      : (loc ? `What are the satellite features around ${loc}?` : 'What are the satellite features?'));
    setActiveQuestion(targetQuery);
    try {
      const res = await submitVQAQuery({
        question: targetQuery,
        location_name: loc || undefined,
        bbox: currentBbox,
        enable_grounding: true,
        enable_voice_response: true,
      });
      setQueryResponse(res);
      if (res.location_meta) {
        setCurrentLocationName(res.location_meta.name);
        setCenterCoords({
          lat: res.location_meta.lat,
          lng: res.location_meta.lon,
          zoom: res.location_meta.zoom || 14,
        });
        setCurrentBbox(res.location_meta.bbox);
      }
      if (res.image_url) {
        setActiveImageUrl(res.image_url);
      }
      if (res.is_comparison || isCompTool) {
        setActiveTab('compare');
      } else {
        setActiveTab('optical');
      }
    } catch (e) {
      console.error('Error handling voice tool call:', e);
    }
  };

  // Toggle Live Persistent Voice Session
  const handleToggleVoiceSession = () => {
    if (!voiceClientRef.current) return;
    if (voiceClientRef.current.isActive()) {
      voiceClientRef.current.stopSession();
    } else {
      voiceClientRef.current.startSession();
    }
  };

  // Handle preset benchmark target selection from Global Start Screen
  const handleSelectPresetLocation = async (promptText: string, name: string, lat: number, lon: number) => {
    setIsSearching(true);
    setErrorMessage(null);
    try {
      const geo = await geocodeLocation(name);
      setCurrentLocationName(geo.name);
      setCenterCoords({ lat: geo.lat, lng: geo.lon, zoom: geo.zoom });
      setCurrentBbox(geo.bbox);
      if (geo.image_url) setActiveImageUrl(geo.image_url);
    } catch {
      setCurrentLocationName(name);
      setCenterCoords({ lat, lng: lon, zoom: 14 });
    } finally {
      setIsSearching(false);
    }
    setActiveTab('optical');
    handleSubmitQuery(promptText, true);
  };

  // Location search bar
  const handleSearchLocation = async (locQuery: string) => {
    setIsSearching(true);
    setErrorMessage(null);
    try {
      const geo = await geocodeLocation(locQuery);
      setCurrentLocationName(geo.name);
      setCenterCoords({ lat: geo.lat, lng: geo.lon, zoom: geo.zoom });
      setCurrentBbox(geo.bbox);
      setActiveImageUrl(geo.image_url);
      setActiveTab('optical');
    } catch (err: any) {
      setErrorMessage(err.message || 'Location not found');
    } finally {
      setIsSearching(false);
    }
  };

  // AOI search from map view
  const handleSearchAOI = async (bbox: BoundingBox) => {
    setIsSearching(true);
    setErrorMessage(null);
    try {
      const lat = (bbox.min_lat + bbox.max_lat) / 2;
      const lon = (bbox.min_lon + bbox.max_lon) / 2;
      const geo = await geocodeLocation(`${lat.toFixed(4)}, ${lon.toFixed(4)}`);
      setCurrentLocationName(geo.name);
      setCenterCoords({ lat: geo.lat, lng: geo.lon, zoom: geo.zoom });
      setCurrentBbox(geo.bbox);
      setActiveImageUrl(geo.image_url);
      setActiveTab('optical');
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to search AOI');
    } finally {
      setIsSearching(false);
    }
  };

  // Map point click
  const handleMapClick = async (lat: number, lng: number) => {
    handleSubmitQuery(`Tell me about the satellite features around this location (${lat.toFixed(4)}N, ${lng.toFixed(4)}E)`, true);
  };

  // Local Speech Synthesis for Voice Output
  const [isLocalSpeaking, setIsLocalSpeaking] = useState<boolean>(false);

  const speakText = (text: string) => {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    utterance.onstart = () => setIsLocalSpeaking(true);
    utterance.onend = () => setIsLocalSpeaking(false);
    utterance.onerror = () => setIsLocalSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };

  const handleToggleSpeak = () => {
    if (!('speechSynthesis' in window)) return;
    if (isLocalSpeaking) {
      window.speechSynthesis.cancel();
      setIsLocalSpeaking(false);
    } else if (queryResponse) {
      const textToSpeak = queryResponse.spoken_text || queryResponse.answer;
      speakText(textToSpeak);
    }
  };

  // Fallback REST / Text Query Submission
  const handleSubmitQuery = async (question: string, enableGrounding: boolean) => {
    setIsLoading(true);
    setActiveQuestion(question);
    setErrorMessage(null);

    try {
      const res = await submitVQAQuery({
        question,
        image_data_url: uploadedImageUrl || undefined,
        bbox: currentBbox,
        enable_grounding: enableGrounding,
        enable_voice_response: true,
      });

      setQueryResponse(res);

      if (res.location_meta) {
        setCurrentLocationName(res.location_meta.name);
        setCenterCoords({
          lat: res.location_meta.lat,
          lng: res.location_meta.lon,
          zoom: res.location_meta.zoom || 13,
        });
        setCurrentBbox(res.location_meta.bbox);
      }

      if (res.image_url) {
        setActiveImageUrl(res.image_url);
      }

      if (res.is_comparison) {
        setActiveTab('compare');
      } else {
        setActiveTab('optical');
      }

      // Auto-narrate synthesized spoken text response
      if (res.spoken_text || res.answer) {
        speakText(res.spoken_text || res.answer);
      }
    } catch (err: any) {
      setErrorMessage(err.message || 'Error processing satellite query');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation & Single Voice Button */}
      <Header
        onSearchLocation={handleSearchLocation}
        onOpenRegistry={() => setIsRegistryOpen(true)}
        voiceState={voiceState}
        voiceStatusMessage={voiceStatusMessage}
        activeLocationName={currentLocationName}
        onToggleVoiceSession={handleToggleVoiceSession}
      />

      {/* Error Toast */}
      {errorMessage && (
        <div className="mx-6 mt-4 p-3 rounded-xl bg-rose-950/80 border border-rose-600/50 text-rose-200 text-xs flex items-center justify-between shadow-lg">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400" />
            <span>{errorMessage}</span>
          </div>
          <button onClick={() => setErrorMessage(null)} className="text-rose-400 hover:text-rose-200 font-bold">
            ×
          </button>
        </div>
      )}

      {/* Main Cockpit Layout: Top Map Stage + Bottom Intelligence & Execution Trace */}
      <main className="flex-1 p-6 flex flex-col gap-6 max-w-[1700px] w-full mx-auto">
        {/* Real-Time Processing HUD (Shows Geocoding, Satellite Fetch, Web Intelligence & AI Reasoning) */}
        <ProcessingHUD
          isLoading={isLoading || voiceState === 'processing'}
          activeQuestion={activeQuestion}
          activeLocationName={currentLocationName}
          isVoiceActive={voiceState !== 'disconnected'}
        />

        {/* TOP SECTION: Dynamic Multimodal Satellite & Geospatial Stage */}
        <section className="w-full flex flex-col gap-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                {activeTab === 'compare'
                  ? `Bi-Temporal Change Comparison: ${currentLocationName} (${queryResponse?.baseline_year || '2020'} vs 2026)`
                  : activeTab === 'sar'
                  ? `Sentinel-1 SAR Radar Observation: ${currentLocationName}`
                  : activeTab === 'fused'
                  ? `Optical + SAR Cross-Modal Fusion: ${currentLocationName}`
                  : activeTab === 'global'
                  ? 'Global Earth Observation Grid'
                  : `Optical Observation Stage (Sentinel-2): ${currentLocationName}`}
              </span>
            </div>

            {/* First-Class Multi-Modal Sensor Switcher */}
            <div className="flex items-center bg-slate-900 border border-slate-800 rounded-xl p-0.5 text-xs flex-wrap gap-0.5">
              <button
                onClick={() => setActiveTab('global')}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg font-medium transition-all ${
                  activeTab === 'global' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Globe2 className="w-3.5 h-3.5" />
                <span>Global Earth</span>
              </button>
              <button
                onClick={() => setActiveTab('optical')}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg font-medium transition-all ${
                  activeTab === 'optical' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <ImageIcon className="w-3.5 h-3.5" />
                <span>Optical (S2)</span>
              </button>
              <button
                onClick={() => setActiveTab('sar')}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg font-medium transition-all ${
                  activeTab === 'sar' ? 'bg-purple-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Radio className="w-3.5 h-3.5 text-purple-300" />
                <span>SAR (S1 GRD)</span>
              </button>
              <button
                onClick={() => setActiveTab('fused')}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg font-medium transition-all ${
                  activeTab === 'fused' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Sparkles className="w-3.5 h-3.5 text-indigo-300" />
                <span>Fused (Opt+SAR)</span>
              </button>
              <button
                onClick={() => setActiveTab('compare')}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg font-medium transition-all ${
                  activeTab === 'compare' ? 'bg-amber-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <GitCompare className="w-3.5 h-3.5 text-amber-300" />
                <span>Change Compare</span>
              </button>
              <button
                onClick={() => setActiveTab('map')}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg font-medium transition-all ${
                  activeTab === 'map' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <MapIcon className="w-3.5 h-3.5" />
                <span>Interactive Map</span>
              </button>
            </div>
          </div>

          {/* Dynamic Top Viewport */}
          <div className="w-full min-h-[480px]">
            {activeTab === 'compare' ? (
              /* DYNAMIC STATE 1: COMPARISON VIEW (Only in Change Compare tab) */
              <ComparisonStage
                beforeImageUrl={queryResponse?.before_image_url || queryResponse?.optical_url || ''}
                afterImageUrl={queryResponse?.after_image_url || queryResponse?.optical_url || ''}
                diffImageUrl={queryResponse?.sar_diff_url || queryResponse?.image_url}
                heatmapUrl={queryResponse?.heatmap_url}
                sarHeatmapUrl={queryResponse?.sar_heatmap_url}
                heatmapBounds={queryResponse?.heatmap_bounds}
                cvaMetrics={queryResponse?.cva_metrics}
                locationName={currentLocationName}
                centerLat={centerCoords.lat}
                centerLng={centerCoords.lng}
                zoom={centerCoords.zoom}
                bbox={currentBbox}
                evidence={queryResponse?.evidence || []}
                changeSummary={queryResponse?.change_summary}
                externalZoom={stageZoom}
                externalPan={stagePan}
                activeQuestion={activeQuestion}
                baselineYear={queryResponse?.baseline_year || '2020'}
                baselinePeriod={queryResponse?.baseline_period}
                baselineTileUrl={queryResponse?.baseline_tile_url}
                onSelectBaselineYear={(year: string) => {
                  handleSubmitQuery(`Compare change at ${currentLocationName} between ${year} and 2026`, true);
                }}
              />
            ) : activeTab === 'global' ? (
              /* DYNAMIC STATE 2: GLOBAL EARTH START SCREEN */
              <GlobalEarthStage
                onSelectPresetLocation={handleSelectPresetLocation}
                onSearchAOI={handleSearchAOI}
              />
            ) : activeTab === 'map' ? (
              /* DYNAMIC STATE 3: SINGLE INTERACTIVE MAP AOI */
              <div style={{ height: '480px', minHeight: '480px', width: '100%' }}>
                <MapViewer
                  centerLat={centerCoords.lat}
                  centerLng={centerCoords.lng}
                  zoom={centerCoords.zoom}
                  bbox={currentBbox}
                  locationName={currentLocationName}
                  onSearchAOI={handleSearchAOI}
                  onMapClickLocation={handleMapClick}
                  isSearching={isSearching}
                />
              </div>
            ) : (
              /* DYNAMIC STATE 4: MULTI-MODAL SENSOR OBSERVATION (Optical / SAR / Fused) */
              <ImageStage
                imageUrl={activeImageUrl || queryResponse?.image_url || '/api/imagery/preview/lake_pichola_s2'}
                opticalUrl={queryResponse?.optical_url}
                sarUrl={queryResponse?.sar_url}
                sarVvUrl={queryResponse?.sar_vv_url}
                sarVhUrl={queryResponse?.sar_vh_url}
                fusedUrl={queryResponse?.fused_url}
                activeLayer={activeTab === 'sar' ? 'sar' : activeTab === 'fused' ? 'fused' : 'optical'}
                sarMetrics={queryResponse?.sar_metrics}
                opticalMetrics={queryResponse?.optical_metrics}
                centerLat={centerCoords.lat}
                centerLng={centerCoords.lng}
                zoom={centerCoords.zoom}
                bbox={currentBbox}
                locationName={currentLocationName}
                uploadedImageUrl={uploadedImageUrl}
                onUploadImage={(url) => setUploadedImageUrl(url)}
                evidence={queryResponse?.evidence || []}
              />
            )}
          </div>
        </section>

        {/* BOTTOM SECTION: Intelligence, Natural Language Q&A, Voice Bar & LangGraph Execution Trace */}
        <section className="w-full grid grid-cols-1 lg:grid-cols-12 gap-6 pt-2 border-t border-slate-800/80">
          {/* Left Sub-Column: Voice / Query Interface & Spoken Answer (7 Cols) */}
          <div className="lg:col-span-7 flex flex-col gap-4">
            <ChatInterface
              onSubmitQuery={handleSubmitQuery}
              isLoading={isLoading}
              voiceState={voiceState}
              voiceStatusMessage={voiceStatusMessage}
              onToggleVoiceSession={handleToggleVoiceSession}
              activeQuestion={activeQuestion}
            />

            {/* Answer Card */}
            <AnswerCard
              response={queryResponse}
              isLoading={isLoading}
              isSpeaking={isLocalSpeaking || voiceState === 'speaking'}
              onToggleSpeak={handleToggleSpeak}
            />
          </div>

          {/* Right Sub-Column: LangGraph Stateful Execution Trace & Real-Time Background Logs (5 Cols) */}
          <div className="lg:col-span-5 flex flex-col gap-3">
            {/* Tab Switcher: LangGraph Trace vs Live Background Logs */}
            <div className="flex items-center justify-between bg-slate-900/90 border border-slate-800 rounded-xl p-1 text-xs">
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setRightPanelTab('trace')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
                    rightPanelTab === 'trace'
                      ? 'bg-cyan-600 text-white shadow-sm font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <GitBranch className="w-3.5 h-3.5" />
                  <span>LangGraph Trace</span>
                  {queryResponse?.trace && queryResponse.trace.length > 0 && (
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-cyan-950 text-cyan-300 font-mono border border-cyan-800/50">
                      {queryResponse.trace.length}
                    </span>
                  )}
                </button>

                <button
                  onClick={() => setRightPanelTab('logs')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
                    rightPanelTab === 'logs'
                      ? 'bg-emerald-600 text-white shadow-sm font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Terminal className="w-3.5 h-3.5" />
                  <span>Live Background Logs</span>
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                </button>
              </div>

              <span className="text-[10px] font-mono text-slate-500 pr-2 hidden sm:inline">
                {rightPanelTab === 'trace' ? 'DAG Trace' : 'Real-Time Telemetry'}
              </span>
            </div>

            {/* Active Sub-Panel Content */}
            {rightPanelTab === 'logs' ? (
              <LiveLogsViewer />
            ) : queryResponse ? (
              <ExecutionTrace trace={queryResponse.trace} runId={queryResponse.run_id} />
            ) : (
              <div className="glass-panel rounded-2xl p-5 border border-slate-800 flex flex-col items-center justify-center text-center text-slate-400 gap-3 min-h-[220px]">
                <Layers className="w-8 h-8 text-slate-600 animate-pulse" />
                <div className="text-xs font-medium text-slate-300">
                  LangGraph Agentic Controller Ready
                </div>
                <p className="text-[11px] text-slate-500 max-w-sm">
                  Speak into the microphone or submit a query to visualize real-time agentic reasoning, STAC retrieval, and RS-VQA execution trace.
                </p>
              </div>
            )}
          </div>
        </section>
      </main>

      {/* Model & Tool Registry Modal */}
      <RegistryModal isOpen={isRegistryOpen} onClose={() => setIsRegistryOpen(false)} />
    </div>
  );
};

export default App;
