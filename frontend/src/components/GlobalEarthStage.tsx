import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import {
  Globe2,
  Sparkles,
  Compass,
  Radio,
  Layers,
  ArrowRight,
  ShieldCheck,
  Building2,
  Trees,
  Waves,
  Route
} from 'lucide-react';

interface GlobalEarthStageProps {
  onSelectPresetLocation: (query: string, name: string, lat: number, lon: number) => void;
  onSearchAOI: (bbox: any) => void;
}

const GLOBAL_FEATURED_TARGETS = [
  {
    name: 'Dubai Palm Jumeirah',
    country: 'UAE',
    lat: 25.1173,
    lon: 55.1351,
    category: 'Urban & Coastal Reclamation',
    icon: <Building2 className="w-3.5 h-3.5 text-orange-400" />,
    badgeColor: 'border-orange-500/50 bg-orange-950/70 text-orange-300',
    prompt: 'What are the infrastructure and building changes around Dubai Palm Jumeirah?',
  },
  {
    name: 'Red Fort & Old Delhi',
    country: 'India',
    lat: 28.6562,
    lon: 77.2410,
    category: 'Heritage & Urban Expansion',
    icon: <Route className="w-3.5 h-3.5 text-amber-400" />,
    badgeColor: 'border-amber-500/50 bg-amber-950/70 text-amber-300',
    prompt: 'Show building and road development around Red Fort Delhi India',
  },
  {
    name: 'Tokyo Bay & Waterfront',
    country: 'Japan',
    lat: 35.6528,
    lon: 139.8394,
    category: 'Port & Infrastructure',
    icon: <Waves className="w-3.5 h-3.5 text-cyan-400" />,
    badgeColor: 'border-cyan-500/50 bg-cyan-950/70 text-cyan-300',
    prompt: 'What are the port and transportation changes in Tokyo Bay Japan?',
  },
  {
    name: 'Eiffel Tower & Seine',
    country: 'France',
    lat: 48.8584,
    lon: 2.2945,
    category: 'Metropolitan Urban Core',
    icon: <Building2 className="w-3.5 h-3.5 text-purple-400" />,
    badgeColor: 'border-purple-500/50 bg-purple-950/70 text-purple-300',
    prompt: 'Describe the satellite land cover and developments around Eiffel Tower Paris',
  },
  {
    name: 'Sundarbans Mangrove Delta',
    country: 'India / Bangladesh',
    lat: 21.8750,
    lon: 88.8750,
    category: 'Canopy & Coastal Hydrology',
    icon: <Trees className="w-3.5 h-3.5 text-emerald-400" />,
    badgeColor: 'border-emerald-500/50 bg-emerald-950/70 text-emerald-300',
    prompt: 'Analyze forest canopy and water body changes in the Sundarbans',
  },
  {
    name: 'Golden Gate & SF Bay',
    country: 'USA',
    lat: 37.8199,
    lon: -122.4783,
    category: 'Maritime & Coastal Corridor',
    icon: <Waves className="w-3.5 h-3.5 text-blue-400" />,
    badgeColor: 'border-blue-500/50 bg-blue-950/70 text-blue-300',
    prompt: 'Compare maritime traffic and shoreline around Golden Gate San Francisco',
  },
];

export const GlobalEarthStage: React.FC<GlobalEarthStageProps> = ({
  onSelectPresetLocation,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);

  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [20.0, 30.0],
      zoom: 2.4,
      minZoom: 2,
      maxZoom: 18,
      zoomControl: false,
      attributionControl: false,
      worldCopyJump: true,
    });

    // High-Definition ArcGIS Global World Imagery Layer
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 18 }
    ).addTo(map);

    // Reference boundaries & coastlines
    L.tileLayer(
      'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 18, opacity: 0.6 }
    ).addTo(map);

    // Place Glowing Target Pins for Benchmark Regions
    GLOBAL_FEATURED_TARGETS.forEach((target) => {
      const customIcon = L.divIcon({
        className: 'custom-globe-pin',
        html: `
          <div style="position: relative; display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; cursor: pointer;">
            <div style="position: absolute; width: 24px; height: 24px; border-radius: 50%; background: rgba(6, 182, 212, 0.4); animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;"></div>
            <div style="width: 14px; height: 14px; border-radius: 50%; background: #06b6d4; border: 2px solid #ffffff; box-shadow: 0 0 15px #06b6d4;"></div>
          </div>
        `,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      });

      const marker = L.marker([target.lat, target.lon], { icon: customIcon }).addTo(map);
      marker.on('click', () => {
        onSelectPresetLocation(target.prompt, target.name, target.lat, target.lon);
      });
    });

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

  return (
    <div className="relative w-full rounded-2xl overflow-hidden glass-panel border border-slate-800 shadow-2xl flex flex-col min-h-[500px]">
      {/* Background Interactive Global Map Canvas */}
      <div
        ref={mapContainerRef}
        style={{ height: '500px', width: '100%', position: 'absolute', inset: 0, zIndex: 1 }}
      />

      {/* Futuristic Gradient Vignette */}
      <div
        className="absolute inset-0 pointer-events-none z-10"
        style={{
          background: 'radial-gradient(circle at center, transparent 40%, rgba(2, 6, 23, 0.75) 85%, rgba(2, 6, 23, 0.95) 100%)',
        }}
      />

      {/* Top Telemetry Overlay */}
      <div className="relative z-20 p-5 flex items-center justify-between flex-wrap gap-3 pointer-events-none">
        <div className="flex items-center gap-2.5 bg-slate-950/90 border border-cyan-500/40 px-3.5 py-1.5 rounded-xl shadow-lg backdrop-blur-md pointer-events-auto">
          <Globe2 className="w-4 h-4 text-cyan-400 animate-pulse" />
          <span className="text-xs font-bold font-mono text-cyan-200 uppercase tracking-wider">
            Global Earth Observation Grid — Real-Time STAC & Optical Mesh
          </span>
          <span className="px-2 py-0.5 text-[9px] font-mono rounded bg-emerald-950 border border-emerald-500/40 text-emerald-300">
            ONLINE
          </span>
        </div>

        <div className="flex items-center gap-2 bg-slate-950/90 border border-slate-800 px-3 py-1.5 rounded-xl text-xs text-slate-300 shadow backdrop-blur-md pointer-events-auto">
          <Radio className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
          <span className="text-[11px] font-mono">Any Worldwide Location Supported</span>
        </div>
      </div>
    </div>
  );
};
