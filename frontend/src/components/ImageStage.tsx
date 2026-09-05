import React, { useState, useRef, useEffect } from 'react';
import L from 'leaflet';
import {
  Upload,
  Eye,
  Radio,
  Sparkles,
  BoxSelect,
  Cpu,
  Layers,
  Activity,
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCcw,
  MapPin,
  Map as MapIcon,
  Image as ImageIcon
} from 'lucide-react';
import { EvidenceRegion } from '../types';

interface ImageStageProps {
  imageUrl: string;
  opticalUrl?: string;
  sarUrl?: string;
  sarVvUrl?: string;
  sarVhUrl?: string;
  fusedUrl?: string;
  activeLayer?: 'optical' | 'sar' | 'fused';
  sarMetrics?: {
    mean_vv_db?: number;
    mean_vh_db?: number;
    vv_vh_ratio?: number;
    temporal_delta_vv_db?: number;
    temporal_delta_vh_db?: number;
    speckle_filter?: string;
    calibration?: string;
  };
  opticalMetrics?: {
    sensor?: string;
    cloud_coverage_pct?: number;
    resolution_m?: number;
    acquisition_date?: string;
  };
  centerLat?: number;
  centerLng?: number;
  zoom?: number;
  bbox?: [number, number, number, number];
  locationName: string;
  uploadedImageUrl: string | null;
  onUploadImage: (dataUrl: string) => void;
  evidence: EvidenceRegion[];
}

export const ImageStage: React.FC<ImageStageProps> = ({
  imageUrl,
  opticalUrl,
  sarUrl,
  sarVvUrl,
  sarVhUrl,
  fusedUrl,
  activeLayer = 'optical',
  sarMetrics,
  opticalMetrics,
  centerLat = 35.6528,
  centerLng = 139.8394,
  zoom = 14,
  bbox,
  locationName,
  uploadedImageUrl,
  onUploadImage,
  evidence,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const polygonLayerRef = useRef<L.LayerGroup | null>(null);
  const markerRef = useRef<L.Marker | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [showEvidenceOverlay, setShowEvidenceOverlay] = useState(true);
  const [sarSubLayer, setSarSubLayer] = useState<'dual' | 'vv' | 'vh'>('dual');
  const [stageMode, setStageMode] = useState<'interactive_map' | 'raster_tile'>('interactive_map');

  // Initialize Interactive High-Resolution Satellite Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [centerLat, centerLng],
      zoom: zoom,
      minZoom: 2,
      maxZoom: 19,
      zoomControl: false,
      attributionControl: false,
      scrollWheelZoom: true,
      worldCopyJump: true,
    });

    // High-Definition ArcGIS World Imagery Satellite Layer (Resolution down to building level)
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 19 }
    ).addTo(map);

    // High-Resolution Reference Places & Boundaries
    L.tileLayer(
      'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 19, opacity: 0.75 }
    ).addTo(map);

    const layerGroup = L.layerGroup().addTo(map);
    polygonLayerRef.current = layerGroup;

    mapInstanceRef.current = map;

    const timer = setTimeout(() => {
      map.invalidateSize();
    }, 200);

    return () => {
      clearTimeout(timer);
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update map center & fly when location coordinates change
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    map.invalidateSize();
    map.flyTo([centerLat, centerLng], Math.max(13, zoom), { duration: 1.2 });

    // Marker
    if (markerRef.current) {
      markerRef.current.remove();
    }
    const marker = L.marker([centerLat, centerLng]).addTo(map);
    marker.bindPopup(`<strong>${locationName}</strong><br/>Sentinel-2 / SAR Observation Target`).openPopup();
    markerRef.current = marker;

    // Render Evidence Polygons & Bounding Boxes
    if (polygonLayerRef.current) {
      polygonLayerRef.current.clearLayers();

      if (showEvidenceOverlay) {
        // Draw Bounding Box AOI if provided
        if (bbox && bbox.length === 4) {
          const [minLon, minLat, maxLon, maxLat] = bbox;
          const rect = L.rectangle([[minLat, minLon], [maxLat, maxLon]], {
            color: '#06b6d4',
            weight: 2,
            fillColor: '#06b6d4',
            fillOpacity: 0.12,
            dashArray: '3, 3'
          });
          polygonLayerRef.current.addLayer(rect);
        }

        // Draw Grounded Evidence Regions
        evidence.forEach((ev) => {
          const strokeColor = ev.color === 'cyan' ? '#06b6d4' : ev.color === 'emerald' ? '#10b981' : '#f59e0b';
          const fillColor = ev.color === 'cyan' ? 'rgba(6, 182, 212, 0.25)' : ev.color === 'emerald' ? 'rgba(16, 185, 129, 0.25)' : 'rgba(245, 158, 11, 0.28)';

          if (bbox && bbox.length === 4 && ev.polygons && ev.polygons.length > 0) {
            const [minLon, minLat, maxLon, maxLat] = bbox;
            const latSpan = maxLat - minLat;
            const lonSpan = maxLon - minLon;

            ev.polygons.forEach((poly) => {
              const geoPoints: [number, number][] = poly.map(([nx, ny]) => [
                minLat + (1.0 - ny) * latSpan,
                minLon + nx * lonSpan
              ]);
              const p = L.polygon(geoPoints, {
                color: strokeColor,
                weight: 2,
                fillColor: fillColor,
                fillOpacity: 0.35,
                dashArray: '4, 2'
              });
              p.bindTooltip(`<strong>${ev.label}</strong> (${Math.round(ev.confidence * 100)}%)`, { permanent: true, direction: 'top' });
              polygonLayerRef.current?.addLayer(p);
            });
          }
        });
      }
    }
  }, [centerLat, centerLng, zoom, bbox, locationName, evidence, showEvidenceOverlay]);

  const handleZoomIn = () => {
    mapInstanceRef.current?.zoomIn();
  };

  const handleZoomOut = () => {
    mapInstanceRef.current?.zoomOut();
  };

  const handleResetView = () => {
    mapInstanceRef.current?.flyTo([centerLat, centerLng], 14);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        onUploadImage(reader.result);
      }
    };
    reader.readAsDataURL(file);
  };

  // Determine active visual source for raster tile mode
  let activeSrc = imageUrl;
  if (uploadedImageUrl) {
    activeSrc = uploadedImageUrl;
  } else if (activeLayer === 'optical') {
    activeSrc = opticalUrl || imageUrl || '/api/imagery/preview/lake_pichola_s2';
  } else if (activeLayer === 'sar') {
    if (sarSubLayer === 'vv' && sarVvUrl) activeSrc = sarVvUrl;
    else if (sarSubLayer === 'vh' && sarVhUrl) activeSrc = sarVhUrl;
    else activeSrc = sarUrl || sarVvUrl || imageUrl;
  } else if (activeLayer === 'fused') {
    activeSrc = fusedUrl || opticalUrl || imageUrl;
  }

  return (
    <div className="relative w-full h-full min-h-[500px] rounded-2xl overflow-hidden glass-panel flex flex-col border border-slate-800 shadow-2xl">
      {/* Top Controls Bar */}
      <div className="p-3 bg-slate-900/95 border-b border-slate-800 flex items-center justify-between flex-wrap gap-2 z-20 backdrop-blur-md">
        <div className="flex items-center gap-2.5">
          <div className={`p-1.5 rounded-lg border ${
            activeLayer === 'sar'
              ? 'bg-purple-950 border-purple-500/40 text-purple-300'
              : activeLayer === 'fused'
              ? 'bg-indigo-950 border-indigo-500/40 text-indigo-300'
              : 'bg-cyan-950 border-cyan-500/40 text-cyan-400'
          }`}>
            {activeLayer === 'sar' ? (
              <Radio className="w-4 h-4 animate-pulse" />
            ) : (
              <Layers className="w-4 h-4" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-bold text-slate-100">
                {uploadedImageUrl
                  ? 'Custom Uploaded Raster'
                  : activeLayer === 'sar'
                  ? `Sentinel-1 SAR Observation: ${locationName}`
                  : activeLayer === 'fused'
                  ? `Optical + SAR Fusion: ${locationName}`
                  : `Optical Observation: ${locationName}`}
              </span>
              <span className={`px-1.5 py-0.2 text-[9px] font-mono uppercase rounded border ${
                activeLayer === 'sar'
                  ? 'bg-purple-950 text-purple-300 border-purple-700/60'
                  : activeLayer === 'fused'
                  ? 'bg-indigo-950 text-indigo-300 border-indigo-700/60'
                  : 'bg-cyan-950 text-cyan-300 border-cyan-700/60'
              }`}>
                {activeLayer === 'sar'
                  ? 'Sentinel-1 C-Band GRD'
                  : activeLayer === 'fused'
                  ? 'Sentinel-2 MSI + Sentinel-1 SAR'
                  : 'Sentinel-2 L2A (10m BOA)'}
              </span>
            </div>
            <span className="text-[11px] text-slate-400 font-mono">
              Coordinates: {centerLat.toFixed(4)}°N, {centerLng.toFixed(4)}°E | Interactive 4K Slippy Canvas
            </span>
          </div>
        </div>

        {/* Right Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Viewport Mode Switcher: Interactive Map vs Raster Tile */}
          <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-0.5 text-xs">
            <button
              onClick={() => setStageMode('interactive_map')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-all ${
                stageMode === 'interactive_map'
                  ? 'bg-cyan-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <MapIcon className="w-3.5 h-3.5" />
              <span>Interactive Map</span>
            </button>
            <button
              onClick={() => setStageMode('raster_tile')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-all ${
                stageMode === 'raster_tile'
                  ? 'bg-cyan-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <ImageIcon className="w-3.5 h-3.5" />
              <span>Raster Scene</span>
            </button>
          </div>

          {/* SAR Sub-channel Toggles */}
          {activeLayer === 'sar' && (
            <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-0.5 text-[10px]">
              <button
                onClick={() => setSarSubLayer('dual')}
                className={`px-2 py-0.5 rounded font-mono transition-all ${
                  sarSubLayer === 'dual' ? 'bg-purple-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Dual-Pol
              </button>
              <button
                onClick={() => setSarSubLayer('vv')}
                className={`px-2 py-0.5 rounded font-mono transition-all ${
                  sarSubLayer === 'vv' ? 'bg-purple-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                VV
              </button>
              <button
                onClick={() => setSarSubLayer('vh')}
                className={`px-2 py-0.5 rounded font-mono transition-all ${
                  sarSubLayer === 'vh' ? 'bg-purple-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                VH
              </button>
            </div>
          )}

          {evidence.length > 0 && (
            <button
              onClick={() => setShowEvidenceOverlay(!showEvidenceOverlay)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                showEvidenceOverlay
                  ? 'bg-cyan-600 text-white border border-cyan-400/40 shadow-cyan-600/30'
                  : 'bg-slate-800 text-slate-400 border border-slate-700'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Evidence ({evidence.length})</span>
            </button>
          )}

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/png,image/jpeg,image/tiff"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            title="Upload Custom Satellite GeoTIFF/PNG"
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium transition-all"
          >
            <Upload className="w-3.5 h-3.5 text-cyan-400" />
            <span>Upload</span>
          </button>
        </div>
      </div>

      {/* Sensor Metrics Banner */}
      {sarMetrics && activeLayer === 'sar' && (
        <div className="px-4 py-1.5 bg-slate-950/90 border-b border-slate-800 flex items-center justify-between text-[11px] font-mono text-slate-300 flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <span className="text-purple-300">σ⁰ (VV): <strong>{sarMetrics.mean_vv_db} dB</strong></span>
            <span className="text-purple-300">σ⁰ (VH): <strong>{sarMetrics.mean_vh_db} dB</strong></span>
            <span className="text-cyan-300">VV/VH Ratio: <strong>{sarMetrics.vv_vh_ratio} dB</strong></span>
            <span className="text-emerald-300">ΔVV Change: <strong>+{sarMetrics.temporal_delta_vv_db} dB</strong></span>
          </div>
          <div className="text-slate-400 text-[10px]">
            {sarMetrics.speckle_filter} | {sarMetrics.calibration}
          </div>
        </div>
      )}

      {opticalMetrics && activeLayer === 'optical' && (
        <div className="px-4 py-1.5 bg-slate-950/90 border-b border-slate-800 flex items-center justify-between text-[11px] font-mono text-slate-300 flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <span className="text-cyan-300">Cloud Cover: <strong>{opticalMetrics.cloud_coverage_pct}%</strong></span>
            <span className="text-emerald-300">Resolution: <strong>{opticalMetrics.resolution_m}m GSD</strong></span>
            <span className="text-slate-400">Acquisition: <strong>{opticalMetrics.acquisition_date}</strong></span>
          </div>
          <div className="text-slate-400 text-[10px]">
            10-Band Multispectral Surface Reflectance (BOA)
          </div>
        </div>
      )}

      {/* MAIN VIEWPORT */}
      <div className="relative flex-1 w-full bg-slate-950 overflow-hidden min-h-[440px]">
        {stageMode === 'interactive_map' ? (
          /* PRIMARY MODE: INTERACTIVE 4K SATELLITE LEAFLET MAP */
          <div className="relative w-full h-full min-h-[440px]">
            <div
              ref={mapContainerRef}
              style={{
                height: '100%',
                minHeight: '440px',
                width: '100%',
                position: 'absolute',
                inset: 0,
                filter: activeLayer === 'sar' ? 'contrast(1.4) saturate(0.2) hue-rotate(240deg)' : activeLayer === 'fused' ? 'contrast(1.2) saturate(1.4)' : 'none'
              }}
            />

            {/* Floating Zoom & Map Controls */}
            <div className="absolute top-4 right-4 z-[400] flex flex-col gap-1.5 bg-slate-900/90 border border-slate-700/80 rounded-xl p-1 shadow-2xl backdrop-blur-md">
              <button
                onClick={handleZoomIn}
                title="Zoom In"
                className="p-2 rounded-lg hover:bg-slate-800 text-slate-200 hover:text-cyan-400 transition-colors"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={handleZoomOut}
                title="Zoom Out"
                className="p-2 rounded-lg hover:bg-slate-800 text-slate-200 hover:text-cyan-400 transition-colors"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <div className="w-full h-px bg-slate-800 my-0.5" />
              <button
                onClick={handleResetView}
                title="Reset View"
                className="p-2 rounded-lg hover:bg-slate-800 text-slate-200 hover:text-cyan-400 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>
        ) : (
          /* SECONDARY MODE: OPTICAL RASTER TILE IMAGE VIEW */
          <div className="relative w-full h-full min-h-[440px] flex items-center justify-center p-4">
            <div className="relative w-full max-w-[820px] aspect-[4/3] max-h-[500px] rounded-2xl overflow-hidden shadow-2xl border border-slate-800 flex items-center justify-center">
              <img
                src={activeSrc}
                alt="Satellite Scene View"
                className="w-full h-full object-cover block select-none"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
