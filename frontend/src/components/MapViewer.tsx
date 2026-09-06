import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { MapPin, Search, Crosshair, RefreshCw, Layers } from 'lucide-react';
import { BoundingBox } from '../types';
import { setActiveMap } from '../utils/captureView';

interface MapViewerProps {
  centerLat: number;
  centerLng: number;
  zoom: number;
  bbox?: [number, number, number, number];
  locationName: string;
  onSearchAOI: (bbox: BoundingBox) => void;
  onMapClickLocation?: (lat: number, lng: number) => void;
  onViewportChange?: (viewportBbox: [number, number, number, number], zoom: number) => void;
  isSearching: boolean;
  navKey?: number;
}

export const MapViewer: React.FC<MapViewerProps> = ({
  centerLat,
  centerLng,
  zoom,
  bbox,
  locationName,
  onSearchAOI,
  onMapClickLocation,
  onViewportChange,
  isSearching,
  navKey,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const bboxLayerRef = useRef<L.Rectangle | null>(null);
  const markerRef = useRef<L.Marker | null>(null);

  const [currentCoords, setCurrentCoords] = useState<{ lat: number; lng: number; zoom: number }>({
    lat: centerLat,
    lng: centerLng,
    zoom: zoom,
  });

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [centerLat, centerLng],
      zoom: zoom,
      zoomControl: true,
      attributionControl: false,
    });

    // Satellite Imagery Base Layer
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 19, crossOrigin: 'anonymous' }
    ).addTo(map);

    // Reference labels layer
    L.tileLayer(
      'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 19, opacity: 0.8, crossOrigin: 'anonymous' }
    ).addTo(map);

    const broadcastBounds = () => {
      const center = map.getCenter();
      const currentZoom = map.getZoom();
      setCurrentCoords({
        lat: parseFloat(center.lat.toFixed(4)),
        lng: parseFloat(center.lng.toFixed(4)),
        zoom: currentZoom,
      });
      if (onViewportChange) {
        const bounds = map.getBounds();
        onViewportChange(
          [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()],
          currentZoom
        );
      }
    };

    map.on('moveend zoomend', broadcastBounds);

    map.on('click', (e: L.LeafletMouseEvent) => {
      if (onMapClickLocation) {
        onMapClickLocation(e.latlng.lat, e.latlng.lng);
      }
    });

    mapInstanceRef.current = map;
    setActiveMap(map);
    broadcastBounds();

    const timer = setTimeout(() => {
      map.invalidateSize();
    }, 200);

    return () => {
      clearTimeout(timer);
      setActiveMap(null);
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  const prevNavKeyRef = useRef<number | undefined>(navKey);
  const prevCenterRef = useRef<{ lat: number; lng: number }>({ lat: centerLat, lng: centerLng });

  // Fly to location ONLY on explicit navigation triggers (navKey changed)
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    map.invalidateSize();

    if (prevNavKeyRef.current !== navKey) {
      prevNavKeyRef.current = navKey;
      const targetZoom = Math.min(19, Math.max(2, zoom));
      const coordsChanged = Math.abs(prevCenterRef.current.lat - centerLat) > 0.0001 || Math.abs(prevCenterRef.current.lng - centerLng) > 0.0001;
      prevCenterRef.current = { lat: centerLat, lng: centerLng };

      const flyTarget = coordsChanged ? [centerLat, centerLng] as [number, number] : [map.getCenter().lat, map.getCenter().lng] as [number, number];
      map.flyTo(flyTarget, targetZoom, { duration: 1.0 });

      // Update marker & popup ONLY on explicit navigation, never autoPan camera
      if (markerRef.current) {
        markerRef.current.remove();
      }
      const marker = L.marker([centerLat, centerLng]).addTo(map);
      marker.bindPopup(`<strong>${locationName}</strong><br/>Sentinel-2 Real-Time AOI`, { autoPan: false }).openPopup();
      markerRef.current = marker;
    }
  }, [navKey, centerLat, centerLng, zoom, locationName]);

  // Update Bounding Box rectangle if available without moving map camera
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (bbox && bbox.length === 4) {
      const [minLon, minLat, maxLon, maxLat] = bbox;
      const bounds = L.latLngBounds([minLat, minLon], [maxLat, maxLon]);

      if (bboxLayerRef.current) {
        bboxLayerRef.current.remove();
      }

      const rect = L.rectangle(bounds, {
        color: '#06b6d4',
        weight: 2,
        fillColor: '#06b6d4',
        fillOpacity: 0.15,
        dashArray: '4, 4',
      }).addTo(map);

      bboxLayerRef.current = rect;
    }
  }, [bbox]);

  const handleSearchCurrentView = () => {
    const map = mapInstanceRef.current;
    if (!map) return;
    const bounds = map.getBounds();
    onSearchAOI({
      min_lon: bounds.getWest(),
      min_lat: bounds.getSouth(),
      max_lon: bounds.getEast(),
      max_lat: bounds.getNorth(),
    });
  };

  const handleRecenter = () => {
    const map = mapInstanceRef.current;
    if (!map) return;
    map.flyTo([centerLat, centerLng], zoom, { duration: 1.0 });
  };

  return (
    <div
      style={{ height: '360px', minHeight: '360px', width: '100%' }}
      className="relative w-full rounded-2xl overflow-hidden glass-panel border border-slate-800 shadow-xl flex flex-col z-10"
    >
      {/* Top Map Action Bar */}
      <div className="absolute top-3 left-3 right-3 z-[1000] flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-2 bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-xl px-3 py-1.5 shadow-lg pointer-events-auto">
          <MapPin className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-xs font-semibold text-slate-200">
            {locationName}
          </span>
          <span className="text-[10px] font-mono text-cyan-300 bg-cyan-950 px-1.5 py-0.5 rounded border border-cyan-800">
            {currentCoords.lat}° N, {currentCoords.lng}° E
          </span>
        </div>

        <div className="flex items-center gap-2 pointer-events-auto">
          <button
            onClick={handleRecenter}
            title="Recenter AOI"
            className="p-2 rounded-xl bg-slate-900/90 backdrop-blur-md border border-slate-700/80 text-slate-300 hover:text-cyan-300 hover:border-cyan-500/50 transition-all shadow-lg"
          >
            <Crosshair className="w-4 h-4" />
          </button>
          <button
            onClick={handleSearchCurrentView}
            disabled={isSearching}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium transition-all shadow-lg shadow-cyan-600/30 disabled:opacity-50"
          >
            {isSearching ? (
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Search className="w-3.5 h-3.5" />
            )}
            <span>Analyze AOI</span>
          </button>
        </div>
      </div>

      {/* Leaflet Map Root */}
      <div
        ref={mapContainerRef}
        style={{ height: '280px', minHeight: '280px', width: '100%' }}
        className="w-full flex-1 z-0"
      />

      {/* Bottom Metadata Ribbon */}
      <div className="px-4 py-2 bg-slate-900/90 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400 flex-shrink-0">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
          Live Optical Satellite Composite (Click map to analyze point)
        </span>
        <span className="font-mono text-slate-500">EPSG:4326 WGS84</span>
      </div>
    </div>
  );
};
