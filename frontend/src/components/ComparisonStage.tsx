import React, { useState, useRef, useEffect } from 'react';
import L from 'leaflet';
import {
  GitCompare,
  Sparkles,
  Layers,
  Eye,
  ShieldCheck,
  ArrowRight,
  Activity,
  Flame,
  Route,
  Building2,
  Waves,
  Trees,
  Maximize2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  MapPin,
  Map as MapIcon,
  Image as ImageIcon,
  Radio,
  Clock,
  CheckCircle2
} from 'lucide-react';
import { EvidenceRegion, ChangeSummary } from '../types';

interface ComparisonStageProps {
  beforeImageUrl: string;
  afterImageUrl: string;
  diffImageUrl?: string;
  heatmapUrl?: string;
  sarHeatmapUrl?: string;
  heatmapBounds?: [[number, number], [number, number]];
  cvaMetrics?: {
    method?: string;
    area_changed_pct?: number;
    mean_magnitude_pct?: number;
    peak_magnitude_pct?: number;
    hotspot_coords?: [number, number];
    confidence?: number;
  };
  locationName: string;
  centerLat?: number;
  centerLng?: number;
  zoom?: number;
  bbox?: [number, number, number, number];
  evidence: EvidenceRegion[];
  changeSummary?: ChangeSummary;
  externalZoom?: number;
  externalPan?: { x: number; y: number };
  activeQuestion?: string;
  baselineYear?: string;
  baselinePeriod?: string;
  baselineTileUrl?: string;
  onSelectBaselineYear?: (year: string) => void;
}

export type MapLayerType =
  | 'archival_satellite'
  | 'optical_satellite'
  | 'osm_carto'
  | 'topo_map'
  | 'carto_dark'
  | 'sar_radar'
  | 'ndvi_vegetation'
  | 'nasa_night_lights';

export interface WaybackYearInfo {
  year: string;
  release: string;
  label: string;
  title: string;
}

export const WAYBACK_YEARS: WaybackYearInfo[] = [
  { year: '2014', release: '10', label: '2014', title: 'Wayback 2014' },
  { year: '2016', release: '388', label: '2016', title: 'Wayback 2016' },
  { year: '2018', release: '239', label: '2018', title: 'Wayback 2018' },
  { year: '2020', release: '29260', label: '2020', title: 'Wayback 2020' },
  { year: '2021', release: '26120', label: '2021', title: 'Wayback 2021' },
  { year: '2022', release: '45134', label: '2022', title: 'Wayback 2022' },
  { year: '2024', release: '16453', label: '2024', title: 'Wayback 2024' },
];

interface LayerConfig {
  id: MapLayerType;
  label: string;
  badge: string;
  url: string;
  maxZoom: number;
  maxNativeZoom?: number;
  errorTileUrl?: string;
  filterClass?: string;
  attribution: string;
}

const STATIC_LAYER_CONFIGS: Record<MapLayerType, LayerConfig> = {
  archival_satellite: {
    id: 'archival_satellite',
    label: '🛰️ Wayback Baseline (2020)',
    badge: 'Esri Wayback Historical Archive (2020)',
    url: 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/29260/{z}/{y}/{x}',
    maxZoom: 20,
    maxNativeZoom: 17,
    errorTileUrl: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Esri World Imagery Wayback Archive (2020)',
  },
  optical_satellite: {
    id: 'optical_satellite',
    label: '🛰️ Current 2026 S2',
    badge: 'Current Sentinel-2 (2026)',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    maxZoom: 20,
    maxNativeZoom: 19,
    attribution: 'Sentinel-2 / Esri World Imagery',
  },
  osm_carto: {
    id: 'osm_carto',
    label: '🗺️ OSM Streets',
    badge: 'OpenStreetMap Vector',
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    maxZoom: 20,
    maxNativeZoom: 19,
    attribution: '© OpenStreetMap contributors',
  },
  topo_map: {
    id: 'topo_map',
    label: '🏔️ Topo / Elevation',
    badge: 'USGS & Esri Topo',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    maxZoom: 20,
    maxNativeZoom: 19,
    attribution: 'Esri, USGS, NOAA',
  },
  carto_dark: {
    id: 'carto_dark',
    label: '🌃 Dark Urban Carto',
    badge: 'CartoDB Dark Matter',
    url: 'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
    maxZoom: 20,
    maxNativeZoom: 19,
    attribution: '© CARTO, OpenStreetMap',
  },
  sar_radar: {
    id: 'sar_radar',
    label: '📡 SAR Radar (VV)',
    badge: 'Sentinel-1 C-Band SAR',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    maxZoom: 20,
    maxNativeZoom: 19,
    filterClass: 'leaflet-sar-filter',
    attribution: 'ESA Copernicus Sentinel-1',
  },
  ndvi_vegetation: {
    id: 'ndvi_vegetation',
    label: '🌿 NDVI Vegetation',
    badge: 'Spectral False-Color Index',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    maxZoom: 20,
    maxNativeZoom: 19,
    filterClass: 'leaflet-ndvi-filter',
    attribution: 'Sentinel-2 MSI NIR/Red',
  },
  nasa_night_lights: {
    id: 'nasa_night_lights',
    label: '💡 Night Radiance',
    badge: 'NASA VIIRS Nighttime Lights',
    url: 'https://map1.vis.earthdata.nasa.gov/wmts-webmerc/VIIRS_CityLights_2012/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg',
    maxZoom: 20,
    maxNativeZoom: 8,
    attribution: 'NASA GIBS / VIIRS',
  },
};

export const ComparisonStage: React.FC<ComparisonStageProps> = ({
  beforeImageUrl,
  afterImageUrl,
  diffImageUrl,
  heatmapUrl,
  sarHeatmapUrl,
  heatmapBounds,
  cvaMetrics,
  locationName,
  centerLat = 28.7524,
  centerLng = 77.4990,
  zoom = 15,
  bbox,
  evidence,
  changeSummary,
  externalZoom,
  externalPan,
  activeQuestion,
  baselineYear = '2020',
  baselinePeriod,
  baselineTileUrl,
  onSelectBaselineYear,
}) => {
  const [viewMode, setViewMode] = useState<'side-by-side' | 'slider' | 'diff'>('side-by-side');
  const [sliderPosition, setSliderPosition] = useState<number>(50);
  const [showEvidence, setShowEvidence] = useState<boolean>(true);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.75);
  const [heatmapType, setHeatmapType] = useState<'optical_cva' | 'sar_logratio'>('optical_cva');
  const [stageType, setStageType] = useState<'interactive_maps' | 'raster_inspector'>('interactive_maps');
  const [cacheBuster, setCacheBuster] = useState<number>(Date.now());
  const [selectedBaselineYear, setSelectedBaselineYear] = useState<string>(baselineYear);
  const [mapsVersion, setMapsVersion] = useState<number>(0);

  useEffect(() => {
    if (baselineYear) {
      setSelectedBaselineYear(baselineYear);
    }
  }, [baselineYear]);

  useEffect(() => {
    setCacheBuster(Date.now());
  }, [beforeImageUrl, afterImageUrl, locationName, heatmapUrl, sarHeatmapUrl, selectedBaselineYear]);

  const getLayerConfig = (type: MapLayerType): LayerConfig => {
    if (type === 'archival_satellite') {
      const curated = WAYBACK_YEARS.find((y) => y.year === selectedBaselineYear);
      const releaseNum = curated?.release || '29260';
      const dynamicUrl =
        baselineTileUrl && selectedBaselineYear === baselineYear
          ? baselineTileUrl
          : `https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/${releaseNum}/{z}/{y}/{x}`;
      return {
        id: 'archival_satellite',
        label: `🛰️ Wayback Baseline (${selectedBaselineYear})`,
        badge: baselinePeriod || `Esri Wayback Archive (${selectedBaselineYear})`,
        url: dynamicUrl,
        maxZoom: 20,
        maxNativeZoom: 17,
        errorTileUrl: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attribution: `Esri World Imagery Wayback (${selectedBaselineYear})`,
      };
    }
    return STATIC_LAYER_CONFIGS[type] || STATIC_LAYER_CONFIGS.optical_satellite;
  };

  // Distinct Layer Choices for Left vs Right Maps: DEFAULT Satellite vs Satellite (Archival on Left, 2026 on Right)
  const [leftLayer, setLeftLayer] = useState<MapLayerType>('archival_satellite');
  const [rightLayer, setRightLayer] = useState<MapLayerType>('optical_satellite');

  // React to User Query Intent if comparison layers requested
  useEffect(() => {
    if (!activeQuestion) return;
    const q = activeQuestion.toLowerCase();
    if (q.includes('sar') || q.includes('radar')) {
      setLeftLayer('sar_radar');
      setRightLayer('optical_satellite');
      setHeatmapType('sar_logratio');
    } else if (q.includes('ndvi') || q.includes('vegetation') || q.includes('crop') || q.includes('greenery')) {
      setLeftLayer('optical_satellite');
      setRightLayer('ndvi_vegetation');
    } else if (q.includes('topo') || q.includes('elevation') || q.includes('terrain') || q.includes('relief')) {
      setLeftLayer('topo_map');
      setRightLayer('optical_satellite');
    } else if (q.includes('night') || q.includes('light') || q.includes('thermal')) {
      setLeftLayer('nasa_night_lights');
      setRightLayer('optical_satellite');
    } else if (q.includes('street') || q.includes('road') || q.includes('osm') || q.includes('network')) {
      setLeftLayer('osm_carto');
      setRightLayer('optical_satellite');
    } else if (q.includes('satellite') || q.includes('change') || q.includes('compare')) {
      setLeftLayer('archival_satellite');
      setRightLayer('optical_satellite');
      setHeatmapType('optical_cva');
    }
  }, [activeQuestion]);

  // Synchronized Dual Leaflet Map Refs
  const leftMapContainerRef = useRef<HTMLDivElement>(null);
  const rightMapContainerRef = useRef<HTMLDivElement>(null);
  const leftMapInstanceRef = useRef<L.Map | null>(null);
  const rightMapInstanceRef = useRef<L.Map | null>(null);
  const leftTileLayerRef = useRef<L.TileLayer | null>(null);
  const rightTileLayerRef = useRef<L.TileLayer | null>(null);
  const rightOverlayRef = useRef<L.TileLayer | null>(null);
  const rightPolygonLayerRef = useRef<L.LayerGroup | null>(null);
  const rightHeatmapOverlayRef = useRef<L.ImageOverlay | null>(null);
  const isSyncingRef = useRef<boolean>(false);

  // Single Diff Heatmap Leaflet Map Refs
  const diffMapContainerRef = useRef<HTMLDivElement>(null);
  const diffMapInstanceRef = useRef<L.Map | null>(null);
  const diffHeatmapOverlayRef = useRef<L.ImageOverlay | null>(null);

  const createTileLayerForConfig = (cfg: LayerConfig) => {
    return L.tileLayer(cfg.url, {
      maxZoom: cfg.maxZoom || 20,
      maxNativeZoom: cfg.maxNativeZoom ?? 17,
      className: cfg.filterClass || '',
      errorTileUrl: cfg.errorTileUrl || 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    });
  };

  // Initialize Synchronized Dual Interactive Leaflet Maps
  useEffect(() => {
    if (stageType !== 'interactive_maps' || viewMode !== 'side-by-side') return;

    // Cleanup existing instances if any
    if (leftMapInstanceRef.current) {
      leftMapInstanceRef.current.remove();
      leftMapInstanceRef.current = null;
    }
    if (rightMapInstanceRef.current) {
      rightMapInstanceRef.current.remove();
      rightMapInstanceRef.current = null;
    }

    if (!leftMapContainerRef.current || !rightMapContainerRef.current) return;

    // 1. Create Left Map (Baseline / Street / Topo / SAR Scene)
    const mapLeft = L.map(leftMapContainerRef.current, {
      center: [centerLat, centerLng],
      zoom: zoom,
      minZoom: 2,
      maxZoom: 20,
      zoomControl: false,
      attributionControl: false,
      scrollWheelZoom: true,
      worldCopyJump: true,
    });

    // 2. Create Right Map (Current / Grounded Optical / NDVI Scene)
    const mapRight = L.map(rightMapContainerRef.current, {
      center: [centerLat, centerLng],
      zoom: zoom,
      minZoom: 2,
      maxZoom: 20,
      zoomControl: false,
      attributionControl: false,
      scrollWheelZoom: true,
      worldCopyJump: true,
    });

    if (!mapRight.getPane('heatmapPane')) {
      const pane = mapRight.createPane('heatmapPane');
      pane.style.zIndex = '450';
    }
    if (!mapRight.getPane('vectorPane')) {
      const pane = mapRight.createPane('vectorPane');
      pane.style.zIndex = '500';
    }

    // Left Layer Setup
    const leftCfg = getLayerConfig(leftLayer);
    const leftTiles = createTileLayerForConfig(leftCfg).addTo(mapLeft);
    leftTileLayerRef.current = leftTiles;

    // Right Layer Setup
    const rightCfg = getLayerConfig(rightLayer);
    const rightTiles = createTileLayerForConfig(rightCfg).addTo(mapRight);
    rightTileLayerRef.current = rightTiles;

    // If Right Layer is satellite, add crisp road & boundary overlays
    if (rightLayer === 'optical_satellite' || rightLayer === 'sar_radar' || rightLayer === 'ndvi_vegetation') {
      const overlay = L.tileLayer(
        'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        { maxZoom: 20, maxNativeZoom: 19, opacity: 0.65 }
      ).addTo(mapRight);
      rightOverlayRef.current = overlay;
    }

    // Synchronize Pan & Zoom bidirectionally
    mapLeft.on('move', () => {
      if (isSyncingRef.current) return;
      isSyncingRef.current = true;
      mapRight.setView(mapLeft.getCenter(), mapLeft.getZoom(), { animate: false });
      isSyncingRef.current = false;
    });

    mapRight.on('move', () => {
      if (isSyncingRef.current) return;
      isSyncingRef.current = true;
      mapLeft.setView(mapRight.getCenter(), mapRight.getZoom(), { animate: false });
      isSyncingRef.current = false;
    });

    const rightPolyGroup = L.layerGroup().addTo(mapRight);
    rightPolygonLayerRef.current = rightPolyGroup;

    leftMapInstanceRef.current = mapLeft;
    rightMapInstanceRef.current = mapRight;
    setMapsVersion((v) => v + 1);

    // Invalidate size on mount and on multi-stage layout updates
    const invalidateMaps = () => {
      if (leftMapInstanceRef.current) leftMapInstanceRef.current.invalidateSize();
      if (rightMapInstanceRef.current) rightMapInstanceRef.current.invalidateSize();
    };

    invalidateMaps();
    const t1 = setTimeout(invalidateMaps, 100);
    const t2 = setTimeout(invalidateMaps, 300);
    const t3 = setTimeout(invalidateMaps, 800);

    const resizeObserver = new ResizeObserver(() => {
      invalidateMaps();
    });

    if (leftMapContainerRef.current) resizeObserver.observe(leftMapContainerRef.current);
    if (rightMapContainerRef.current) resizeObserver.observe(rightMapContainerRef.current);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      resizeObserver.disconnect();
      if (rightHeatmapOverlayRef.current && rightMapInstanceRef.current) {
        rightMapInstanceRef.current.removeLayer(rightHeatmapOverlayRef.current);
        rightHeatmapOverlayRef.current = null;
      }
      if (rightPolygonLayerRef.current && rightMapInstanceRef.current) {
        rightMapInstanceRef.current.removeLayer(rightPolygonLayerRef.current);
        rightPolygonLayerRef.current = null;
      }
      if (leftMapInstanceRef.current) {
        leftMapInstanceRef.current.remove();
        leftMapInstanceRef.current = null;
      }
      if (rightMapInstanceRef.current) {
        rightMapInstanceRef.current.remove();
        rightMapInstanceRef.current = null;
      }
    };
  }, [stageType, viewMode, centerLat, centerLng]);

  // Update Left Tile Layer dynamically when user changes selector or baseline year
  useEffect(() => {
    if (!leftMapInstanceRef.current) return;
    const cfg = getLayerConfig(leftLayer);
    if (leftTileLayerRef.current) {
      leftMapInstanceRef.current.removeLayer(leftTileLayerRef.current);
    }
    const newTiles = createTileLayerForConfig(cfg).addTo(leftMapInstanceRef.current);
    leftTileLayerRef.current = newTiles;
    leftMapInstanceRef.current.invalidateSize();
  }, [leftLayer, selectedBaselineYear, baselineTileUrl]);

  // Update Right Tile Layer dynamically when user changes selector
  useEffect(() => {
    if (!rightMapInstanceRef.current) return;
    const cfg = getLayerConfig(rightLayer);
    if (rightTileLayerRef.current) {
      rightMapInstanceRef.current.removeLayer(rightTileLayerRef.current);
    }
    const newTiles = createTileLayerForConfig(cfg).addTo(rightMapInstanceRef.current);
    rightTileLayerRef.current = newTiles;

    if (rightOverlayRef.current) {
      rightMapInstanceRef.current.removeLayer(rightOverlayRef.current);
      rightOverlayRef.current = null;
    }
    if (rightLayer === 'optical_satellite' || rightLayer === 'sar_radar' || rightLayer === 'ndvi_vegetation') {
      const overlay = L.tileLayer(
        'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        { maxZoom: 20, maxNativeZoom: 19, opacity: 0.65 }
      ).addTo(rightMapInstanceRef.current);
      rightOverlayRef.current = overlay;
    }

    rightMapInstanceRef.current.invalidateSize();
  }, [rightLayer]);

  // Update Map Position when coordinates change
  useEffect(() => {
    if (leftMapInstanceRef.current && rightMapInstanceRef.current) {
      leftMapInstanceRef.current.setView([centerLat, centerLng], zoom, { animate: true });
      rightMapInstanceRef.current.setView([centerLat, centerLng], zoom, { animate: true });
      leftMapInstanceRef.current.invalidateSize();
      rightMapInstanceRef.current.invalidateSize();
    }
  }, [centerLat, centerLng, zoom]);


  // Render Deterministic Pixel-Level Georeferenced Raster Heatmap (CVA / SAR Log-Ratio) on Right Map
  useEffect(() => {
    if (!rightMapInstanceRef.current) return;

    // Cleanup existing overlay if any
    if (rightHeatmapOverlayRef.current) {
      rightMapInstanceRef.current.removeLayer(rightHeatmapOverlayRef.current);
      rightHeatmapOverlayRef.current = null;
    }

    const activeHeatmap = heatmapType === 'sar_logratio'
      ? (sarHeatmapUrl || heatmapUrl)
      : (heatmapUrl || (diffImageUrl && !diffImageUrl.includes('s2_optical') && !diffImageUrl.includes('current_2026') ? diffImageUrl : undefined));

    if (showHeatmap && activeHeatmap) {
      const bounds: L.LatLngBoundsExpression = heatmapBounds || (
        bbox && bbox.length === 4
          ? [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]
          : [[centerLat - 0.02, centerLng - 0.025], [centerLat + 0.02, centerLng + 0.025]]
      );

      const overlay = L.imageOverlay(
        `${activeHeatmap}${activeHeatmap.includes('?') ? '&' : '?'}v=${cacheBuster}`,
        bounds,
        {
          opacity: heatmapOpacity,
          interactive: false,
          pane: rightMapInstanceRef.current.getPane('heatmapPane') ? 'heatmapPane' : 'overlayPane',
        }
      ).addTo(rightMapInstanceRef.current);

      overlay.bringToFront();
      rightHeatmapOverlayRef.current = overlay;
    }
  }, [
    showHeatmap,
    heatmapType,
    heatmapUrl,
    sarHeatmapUrl,
    diffImageUrl,
    heatmapBounds,
    bbox,
    centerLat,
    centerLng,
    heatmapOpacity,
    cacheBuster,
    viewMode,
    stageType,
    mapsVersion,
  ]);

  // Render Vector Evidence Overlay on Right Map
  useEffect(() => {
    if (!rightMapInstanceRef.current || !rightPolygonLayerRef.current) return;

    rightPolygonLayerRef.current.clearLayers();

    if (showEvidence && evidence.length > 0) {
      evidence.forEach((ev) => {
        const type = ev.change_type || 'building';
        const color =
          type === 'road' ? '#fbbf24' :
          type === 'building' ? '#f97316' :
          type === 'water' ? '#22d3ee' :
          type === 'forest' ? '#34d399' : '#a855f7';

        if (ev.polygons && ev.polygons.length > 0) {
          const deltaLat = 0.0035;
          const deltaLng = 0.0045;
          const latLngs = ev.polygons[0].map(([x, y]) => [
            centerLat + (0.5 - y) * deltaLat,
            centerLng + (x - 0.5) * deltaLng,
          ] as [number, number]);

          const poly = L.polygon(latLngs, {
            color: color,
            weight: 2,
            opacity: 0.9,
            fillColor: color,
            fillOpacity: 0.28,
            dashArray: '4, 4',
            pane: rightMapInstanceRef.current?.getPane('vectorPane') ? 'vectorPane' : 'overlayPane',
          });

          poly.bindTooltip(
            `<div class="text-xs font-mono font-bold text-slate-100 bg-slate-950/90 px-2 py-1 rounded border border-cyan-500/40 shadow-lg">
              ${ev.label}${ev.confidence != null ? ` (${Math.round(ev.confidence * 100)}%)` : ''}
            </div>`,
            { permanent: false, direction: 'top', className: 'custom-leaflet-tooltip' }
          );

          if (rightPolygonLayerRef.current) {
            rightPolygonLayerRef.current.addLayer(poly);
          }
        }
      });
    }
  }, [showEvidence, evidence, centerLat, centerLng, viewMode, stageType, mapsVersion]);

  // Initialize Diff Heatmap Interactive Leaflet Map
  useEffect(() => {
    if (viewMode !== 'diff') return;

    if (diffMapInstanceRef.current) {
      diffMapInstanceRef.current.remove();
      diffMapInstanceRef.current = null;
    }

    if (!diffMapContainerRef.current) return;

    const mapDiff = L.map(diffMapContainerRef.current, {
      center: [centerLat, centerLng],
      zoom: zoom,
      minZoom: 2,
      maxZoom: 20,
      zoomControl: false,
      attributionControl: false,
      scrollWheelZoom: true,
      worldCopyJump: true,
    });

    if (!mapDiff.getPane('heatmapPane')) {
      const pane = mapDiff.createPane('heatmapPane');
      pane.style.zIndex = '450';
    }
    if (!mapDiff.getPane('vectorPane')) {
      const pane = mapDiff.createPane('vectorPane');
      pane.style.zIndex = '500';
    }

    // Add Base Satellite TileLayer
    createTileLayerForConfig(getLayerConfig('optical_satellite')).addTo(mapDiff);

    // Add Reference Road & Boundary Overlay
    L.tileLayer(
      'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 20, maxNativeZoom: 19, opacity: 0.65 }
    ).addTo(mapDiff);

    // Add Georeferenced Heatmap Overlay
    const activeHeatmap = heatmapType === 'sar_logratio'
      ? (sarHeatmapUrl || heatmapUrl)
      : (heatmapUrl || (diffImageUrl && !diffImageUrl.includes('s2_optical') && !diffImageUrl.includes('current_2026') ? diffImageUrl : undefined));

    if (showHeatmap && activeHeatmap) {
      const bounds: L.LatLngBoundsExpression = heatmapBounds || (
        bbox && bbox.length === 4
          ? [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]
          : [[centerLat - 0.02, centerLng - 0.025], [centerLat + 0.02, centerLng + 0.025]]
      );

      const overlay = L.imageOverlay(
        `${activeHeatmap}${activeHeatmap.includes('?') ? '&' : '?'}v=${cacheBuster}`,
        bounds,
        {
          opacity: heatmapOpacity,
          interactive: false,
          pane: 'heatmapPane',
        }
      ).addTo(mapDiff);

      overlay.bringToFront();
      diffHeatmapOverlayRef.current = overlay;
    }

    // Render Evidence Polygons on Diff Map
    if (showEvidence && evidence.length > 0) {
      const diffPolyGroup = L.layerGroup().addTo(mapDiff);
      evidence.forEach((ev) => {
        const type = ev.change_type || 'building';
        const color =
          type === 'road' ? '#fbbf24' :
          type === 'building' ? '#f97316' :
          type === 'water' ? '#22d3ee' :
          type === 'forest' ? '#34d399' : '#a855f7';

        if (ev.polygons && ev.polygons.length > 0) {
          const deltaLat = 0.0035;
          const deltaLng = 0.0045;
          const latLngs = ev.polygons[0].map(([x, y]) => [
            centerLat + (0.5 - y) * deltaLat,
            centerLng + (x - 0.5) * deltaLng,
          ] as [number, number]);

          const poly = L.polygon(latLngs, {
            color: color,
            weight: 2,
            opacity: 0.9,
            fillColor: color,
            fillOpacity: 0.28,
            dashArray: '4, 4',
            pane: 'vectorPane',
          });

          poly.bindTooltip(
            `<div class="text-xs font-mono font-bold text-slate-100 bg-slate-950/90 px-2 py-1 rounded border border-cyan-500/40 shadow-lg">
              ${ev.label}${ev.confidence != null ? ` (${Math.round(ev.confidence * 100)}%)` : ''}
            </div>`,
            { permanent: false, direction: 'top', className: 'custom-leaflet-tooltip' }
          );

          diffPolyGroup.addLayer(poly);
        }
      });
    }

    diffMapInstanceRef.current = mapDiff;

    const invalidate = () => {
      if (diffMapInstanceRef.current) diffMapInstanceRef.current.invalidateSize();
    };
    invalidate();
    const t1 = setTimeout(invalidate, 100);
    const t2 = setTimeout(invalidate, 300);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      if (diffMapInstanceRef.current) {
        diffMapInstanceRef.current.remove();
        diffMapInstanceRef.current = null;
      }
    };
  }, [viewMode, centerLat, centerLng, zoom, heatmapType, heatmapUrl, sarHeatmapUrl, diffImageUrl, heatmapBounds, bbox, showHeatmap, showEvidence, evidence, heatmapOpacity, cacheBuster]);

  const handleZoomIn = () => {
    if (leftMapInstanceRef.current && rightMapInstanceRef.current) {
      leftMapInstanceRef.current.zoomIn();
    }
    if (diffMapInstanceRef.current) {
      diffMapInstanceRef.current.zoomIn();
    }
  };

  const handleZoomOut = () => {
    if (leftMapInstanceRef.current && rightMapInstanceRef.current) {
      leftMapInstanceRef.current.zoomOut();
    }
    if (diffMapInstanceRef.current) {
      diffMapInstanceRef.current.zoomOut();
    }
  };

  const handleResetZoom = () => {
    if (leftMapInstanceRef.current && rightMapInstanceRef.current) {
      leftMapInstanceRef.current.setView([centerLat, centerLng], 15);
    }
    if (diffMapInstanceRef.current) {
      diffMapInstanceRef.current.setView([centerLat, centerLng], 15);
    }
  };

  return (
    <div className="relative w-full h-full min-h-[540px] rounded-2xl overflow-hidden glass-panel flex flex-col">
      {/* Top Header Bar */}
      <div className="p-3 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between flex-wrap gap-2 z-20">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-cyan-950 border border-cyan-500/40 text-cyan-400">
            <GitCompare className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-slate-100">
                Multi-Data Comparison: {locationName}
              </span>
              <span className="px-1.5 py-0.2 text-[9px] font-mono uppercase bg-emerald-950 text-emerald-300 border border-emerald-700/60 rounded">
                Synchronized Dual View
              </span>
            </div>
            <span className="text-[11px] text-slate-400 font-mono flex items-center gap-1.5">
              <span className="text-amber-300 font-medium">Left: {getLayerConfig(leftLayer).label}</span>
              <ArrowRight className="inline w-3 h-3 text-cyan-400" />
              <span className="text-cyan-300 font-medium">Right: {getLayerConfig(rightLayer).label}</span>
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* View Mode Switches */}
          <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-0.5 text-xs font-medium">
            <button
              onClick={() => { setViewMode('side-by-side'); setStageType('interactive_maps'); }}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-all ${
                viewMode === 'side-by-side' && stageType === 'interactive_maps'
                  ? 'bg-cyan-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <MapIcon className="w-3 h-3" />
              <span>Dual Interactive Maps</span>
            </button>
            <button
              onClick={() => { setViewMode('slider'); setStageType('raster_inspector'); }}
              className={`px-2.5 py-1 rounded-md transition-all ${
                viewMode === 'slider' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Split Slider
            </button>
            <button
              onClick={() => { setViewMode('diff'); setStageType('raster_inspector'); }}
              className={`px-2.5 py-1 rounded-md transition-all flex items-center gap-1 ${
                viewMode === 'diff' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Flame className="w-3 h-3 text-amber-400" />
              <span>Diff Heatmap</span>
            </button>
          </div>

          {/* Zoom Buttons */}
          <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-0.5 text-xs font-mono">
            <button
              onClick={handleZoomOut}
              className="p-1 text-slate-400 hover:text-white transition-all"
              title="Zoom Out (-)"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleResetZoom}
              className="p-1 text-cyan-400 hover:text-cyan-200 font-bold"
              title="Reset View"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleZoomIn}
              className="p-1 text-slate-400 hover:text-white transition-all"
              title="Zoom In (+)"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Heatmap Layer Toggle & Opacity Slider */}
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
              showHeatmap
                ? 'bg-rose-600/90 text-white border border-rose-400/40 shadow-sm shadow-rose-950/50'
                : 'bg-slate-800 text-slate-400 border border-slate-700 hover:text-slate-200'
            }`}
            title="Toggle Georeferenced Pixel CVA Heatmap Overlay"
          >
            <Flame className={`w-3.5 h-3.5 ${showHeatmap ? 'text-amber-300 animate-pulse' : 'text-slate-400'}`} />
            <span>Pixel Heatmap</span>
          </button>

          {showHeatmap && (
            <div className="flex items-center gap-1.5 bg-slate-950/90 border border-slate-800 rounded-lg px-2 py-1 text-xs font-mono text-slate-300">
              <span className="text-[10px] text-slate-400">Opacity:</span>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={heatmapOpacity}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  setHeatmapOpacity(val);
                  if (rightHeatmapOverlayRef.current) {
                    rightHeatmapOverlayRef.current.setOpacity(val);
                  }
                  if (diffHeatmapOverlayRef.current) {
                    diffHeatmapOverlayRef.current.setOpacity(val);
                  }
                }}
                className="w-14 h-1.5 accent-cyan-400 cursor-pointer"
              />
              <span className="text-[10px] text-cyan-300 min-w-[28px]">{Math.round(heatmapOpacity * 100)}%</span>
            </div>
          )}

          {evidence.length > 0 && (
            <button
              onClick={() => setShowEvidence(!showEvidence)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                showEvidence
                  ? 'bg-emerald-600/90 text-white border border-emerald-400/40'
                  : 'bg-slate-800 text-slate-400 border border-slate-700'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Highlights ({evidence.length})</span>
            </button>
          )}
        </div>
      </div>

      {/* Multi-Data Quick Presets Bar */}
      <div className="px-3 py-1.5 bg-slate-950/90 border-b border-slate-800/80 flex items-center justify-between gap-2 overflow-x-auto text-[11px] font-mono">
        <span className="text-slate-400 flex items-center gap-1 shrink-0">
          <Layers className="w-3 h-3 text-cyan-400" />
          <span>Compare Presets:</span>
        </span>
        <div className="flex items-center gap-1.5 overflow-x-auto">
          {/* PRIMARY DEFAULT: Satellite vs Satellite (Archival Baseline vs Current 2026 Sentinel-2) */}
          <button
            onClick={() => { setLeftLayer('archival_satellite'); setRightLayer('optical_satellite'); }}
            className={`px-2.5 py-0.5 rounded border transition-all shrink-0 font-semibold ${
              leftLayer === 'archival_satellite' && rightLayer === 'optical_satellite'
                ? 'bg-amber-950 border-amber-400 text-amber-200 shadow-sm'
                : 'bg-slate-900 border-slate-700 text-slate-300 hover:border-slate-500'
            }`}
          >
            🛰️ Wayback ({selectedBaselineYear}) vs 🛰️ 2026
          </button>
          <button
            onClick={() => { setLeftLayer('osm_carto'); setRightLayer('optical_satellite'); }}
            className={`px-2 py-0.5 rounded border transition-all shrink-0 ${
              leftLayer === 'osm_carto' && rightLayer === 'optical_satellite'
                ? 'bg-cyan-950 border-cyan-400 text-cyan-200 font-bold'
                : 'bg-slate-900 border-slate-700 text-slate-300 hover:border-slate-500'
            }`}
          >
            🗺️ OSM vs 🛰️ Satellite
          </button>
          <button
            onClick={() => { setLeftLayer('sar_radar'); setRightLayer('optical_satellite'); }}
            className={`px-2 py-0.5 rounded border transition-all shrink-0 ${
              leftLayer === 'sar_radar' && rightLayer === 'optical_satellite'
                ? 'bg-purple-950 border-purple-400 text-purple-200 font-bold'
                : 'bg-slate-900 border-slate-700 text-slate-300 hover:border-slate-500'
            }`}
          >
            📡 SAR Radar vs 🛰️ Optical
          </button>
          <button
            onClick={() => { setLeftLayer('topo_map'); setRightLayer('optical_satellite'); }}
            className={`px-2 py-0.5 rounded border transition-all shrink-0 ${
              leftLayer === 'topo_map' && rightLayer === 'optical_satellite'
                ? 'bg-amber-950 border-amber-400 text-amber-200 font-bold'
                : 'bg-slate-900 border-slate-700 text-slate-300 hover:border-slate-500'
            }`}
          >
            🏔️ Topo Elevation vs 🛰️ Satellite
          </button>
          <button
            onClick={() => { setLeftLayer('optical_satellite'); setRightLayer('ndvi_vegetation'); }}
            className={`px-2 py-0.5 rounded border transition-all shrink-0 ${
              leftLayer === 'optical_satellite' && rightLayer === 'ndvi_vegetation'
                ? 'bg-emerald-950 border-emerald-400 text-emerald-200 font-bold'
                : 'bg-slate-900 border-slate-700 text-slate-300 hover:border-slate-500'
            }`}
          >
            🌿 Optical vs 🌿 NDVI Vegetation
          </button>
          <button
            onClick={() => { setLeftLayer('nasa_night_lights'); setRightLayer('optical_satellite'); }}
            className={`px-2 py-0.5 rounded border transition-all shrink-0 ${
              leftLayer === 'nasa_night_lights' && rightLayer === 'optical_satellite'
                ? 'bg-indigo-950 border-indigo-400 text-indigo-200 font-bold'
                : 'bg-slate-900 border-slate-700 text-slate-300 hover:border-slate-500'
            }`}
          >
            💡 Night Lights vs 🛰️ Day
          </button>
        </div>
      </div>

      {/* Main Dual Viewport Area */}
      <div className="relative flex-1 w-full bg-slate-950 flex items-center justify-center p-3 overflow-hidden select-none">
        {viewMode === 'side-by-side' && stageType === 'interactive_maps' ? (
          /* High-Resolution Synchronized Interactive Dual Leaflet Maps with Distinct Layers */
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full h-[520px] min-h-[460px]">
            {/* Left Synchronized Map */}
            <div className="relative rounded-2xl overflow-hidden border border-amber-500/40 shadow-xl bg-slate-900 flex flex-col h-full min-h-[440px]">
              {/* Left Layer Header & Layer Switcher */}
              <div className="absolute top-2.5 left-2.5 right-2.5 flex items-center justify-between gap-2 z-30 pointer-events-none flex-wrap">
                <div className="bg-slate-900/95 border border-amber-500/60 text-amber-300 text-[11px] font-mono px-2.5 py-1 rounded shadow-lg flex items-center gap-1.5 pointer-events-auto">
                  <Clock className="w-3 h-3 text-amber-400" />
                  <span className="font-semibold">{getLayerConfig(leftLayer).label}</span>
                </div>

                {/* Wayback Baseline Year Selector Pills */}
                {leftLayer === 'archival_satellite' && (
                  <div className="bg-slate-950/95 border border-amber-500/50 rounded-lg p-0.5 text-[10px] font-mono flex items-center gap-1 pointer-events-auto shadow-xl">
                    <span className="text-[9px] text-amber-300 font-semibold px-1">Era:</span>
                    {WAYBACK_YEARS.map((wy) => (
                      <button
                        key={wy.year}
                        onClick={() => {
                          setSelectedBaselineYear(wy.year);
                          if (onSelectBaselineYear) {
                            onSelectBaselineYear(wy.year);
                          }
                        }}
                        className={`px-1.5 py-0.5 rounded transition-all font-semibold ${
                          selectedBaselineYear === wy.year
                            ? 'bg-amber-500 text-slate-950 shadow'
                            : 'text-amber-200/70 hover:text-white hover:bg-amber-900/50'
                        }`}
                        title={`Switch baseline satellite imagery to ${wy.year}`}
                      >
                        {wy.label}
                      </button>
                    ))}
                  </div>
                )}

                {/* Left Layer Selector Pills */}
                <div className="bg-slate-950/95 border border-slate-700 rounded-lg p-0.5 text-[10px] font-mono flex items-center gap-1 pointer-events-auto shadow-xl flex-wrap">
                  <button
                    onClick={() => setLeftLayer('archival_satellite')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      leftLayer === 'archival_satellite' ? 'bg-amber-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    🛰️ Wayback
                  </button>
                  <button
                    onClick={() => setLeftLayer('osm_carto')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      leftLayer === 'osm_carto' ? 'bg-amber-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    🗺️ OSM Streets
                  </button>
                  <button
                    onClick={() => setLeftLayer('topo_map')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      leftLayer === 'topo_map' ? 'bg-amber-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    🏔️ Topo
                  </button>
                  <button
                    onClick={() => setLeftLayer('sar_radar')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      leftLayer === 'sar_radar' ? 'bg-purple-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    📡 SAR (VV)
                  </button>
                  <button
                    onClick={() => setLeftLayer('carto_dark')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      leftLayer === 'carto_dark' ? 'bg-amber-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    🌃 Dark Carto
                  </button>
                  <button
                    onClick={() => setLeftLayer('nasa_night_lights')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      leftLayer === 'nasa_night_lights' ? 'bg-indigo-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    💡 Night
                  </button>
                </div>
              </div>

              {/* Map Canvas */}
              <div ref={leftMapContainerRef} className="w-full h-full flex-1 z-10" />

              <div className="absolute bottom-2 left-2 right-2 bg-slate-950/85 border border-slate-800 rounded px-2.5 py-1 text-[10px] font-mono text-amber-300 flex items-center justify-between z-30 pointer-events-none">
                <span>Left: {getLayerConfig(leftLayer).badge}</span>
                <span className="text-slate-400">{getLayerConfig(leftLayer).attribution}</span>
              </div>
            </div>

            {/* Right Synchronized Map */}
            <div className="relative rounded-2xl overflow-hidden border-2 border-cyan-500/60 shadow-xl shadow-cyan-950/30 bg-slate-900 flex flex-col h-full min-h-[440px]">
              {/* Right Layer Header & Switcher */}
              <div className="absolute top-2.5 left-2.5 right-2.5 flex items-center justify-between gap-2 z-30 pointer-events-none flex-wrap">
                <div className="bg-slate-900/95 border border-cyan-400/60 text-cyan-300 text-[11px] font-mono px-2.5 py-1 rounded shadow-lg flex items-center gap-1.5 pointer-events-auto">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                  <span className="font-semibold">{getLayerConfig(rightLayer).label}</span>
                </div>

                {/* Right Layer Selector Pills */}
                <div className="bg-slate-950/95 border border-cyan-500/40 rounded-lg p-0.5 text-[10px] font-mono flex items-center gap-1 pointer-events-auto shadow-xl flex-wrap">
                  <button
                    onClick={() => setRightLayer('optical_satellite')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      rightLayer === 'optical_satellite' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    🛰️ S2 Optical (2026)
                  </button>
                  <button
                    onClick={() => setRightLayer('ndvi_vegetation')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      rightLayer === 'ndvi_vegetation' ? 'bg-emerald-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    🌿 NDVI Index
                  </button>
                  <button
                    onClick={() => setRightLayer('sar_radar')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      rightLayer === 'sar_radar' ? 'bg-purple-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    📡 SAR (VV)
                  </button>
                  <button
                    onClick={() => setRightLayer('osm_carto')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      rightLayer === 'osm_carto' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    🗺️ OSM Streets
                  </button>
                </div>
              </div>

              {/* Map Canvas */}
              <div ref={rightMapContainerRef} className="w-full h-full flex-1 z-10" />

              {/* Floating Turbo Heatmap Scale Legend */}
              {showHeatmap && (
                <div className="absolute top-12 right-2.5 z-30 bg-slate-950/90 border border-rose-500/50 rounded-lg p-2 shadow-xl text-[10px] font-mono pointer-events-auto flex flex-col gap-1 backdrop-blur-md">
                  <div className="flex items-center justify-between gap-2 text-rose-300 font-bold">
                    <span className="flex items-center gap-1">
                      <Flame className="w-3 h-3 text-amber-400" />
                      <span>Pixel CVA Heatmap</span>
                    </span>
                    <span className="text-cyan-300">{Math.round(heatmapOpacity * 100)}%</span>
                  </div>
                  <div className="w-32 h-2.5 rounded-full overflow-hidden border border-slate-700 bg-gradient-to-r from-blue-600 via-emerald-400 via-amber-400 to-rose-600" />
                  <div className="flex justify-between text-[9px] text-slate-400">
                    <span>Baseline ({selectedBaselineYear})</span>
                    <span>Peak Δ (2026)</span>
                  </div>
                </div>
              )}

              <div className="absolute bottom-2 left-2 right-2 bg-slate-950/85 border border-slate-800 rounded px-2.5 py-1 text-[10px] font-mono text-cyan-300 flex items-center justify-between z-30 pointer-events-none">
                <span>Right: {getLayerConfig(rightLayer).badge}</span>
                <span>{evidence.length} Grounded Detections</span>
              </div>
            </div>
          </div>
        ) : viewMode === 'slider' ? (
          /* Split Slider Inspector */
          <div className="relative w-full h-full min-h-[420px] max-w-[780px] rounded-2xl overflow-hidden shadow-2xl border border-slate-800 select-none">
            <img
              src={`${afterImageUrl}${afterImageUrl.includes('?') ? '&' : '?'}v=${cacheBuster}`}
              alt="After Sentinel-2 Scene"
              className="absolute inset-0 w-full h-full object-cover"
            />
            <div
              className="absolute inset-y-0 left-0 overflow-hidden"
              style={{ width: `${sliderPosition}%` }}
            >
              <img
                src={`${beforeImageUrl}${beforeImageUrl.includes('?') ? '&' : '?'}v=${cacheBuster}`}
                alt="Before Historical Scene"
                className="absolute inset-0 max-w-none h-full object-cover"
                style={{ width: '780px' }}
              />
              <div className="absolute top-3 left-3 bg-slate-900/90 border border-amber-500/60 text-amber-300 text-[10px] font-mono px-2 py-0.5 rounded shadow z-30">
                {changeSummary?.baseline_period || '2021-2025 Baseline'}
              </div>
            </div>
            <div className="absolute top-3 right-3 bg-slate-900/90 border border-cyan-500/60 text-cyan-300 text-[10px] font-mono px-2 py-0.5 rounded shadow z-30">
              {changeSummary?.current_period || 'Current (2026)'}
            </div>
            <div
              className="absolute inset-y-0 w-1 bg-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.8)] cursor-ew-resize z-30 flex items-center justify-center pointer-events-none"
              style={{ left: `${sliderPosition}%` }}
            >
              <div className="w-7 h-7 rounded-full bg-cyan-500 border-2 border-white shadow-xl flex items-center justify-center text-xs font-bold text-slate-950">
                ↔
              </div>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={sliderPosition}
              onChange={(e) => setSliderPosition(Number(e.target.value))}
              className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize z-40"
            />
          </div>
        ) : (
          /* Radiometric Difference Heatmap Superimposed View (Now Interactive Leaflet Map) */
          <div className="relative w-full h-full min-h-[420px] max-w-[820px] rounded-2xl overflow-hidden shadow-2xl border border-slate-800 bg-slate-950 select-none">
            {/* Real Interactive Leaflet Canvas */}
            <div ref={diffMapContainerRef} className="absolute inset-0 w-full h-full z-10" />

            {/* Top-Left Banner */}
            <div className="absolute top-3 left-3 bg-rose-950/90 border border-rose-600/60 text-rose-300 text-[10px] font-mono px-2.5 py-1 rounded shadow flex items-center gap-1 z-30 backdrop-blur-md">
              <Flame className="w-3.5 h-3.5 text-rose-400" />
              <span>Pixel-Level Change Vector Analysis (CVA) Superimposed Overlay</span>
            </div>

            {/* Floating Color Scale Legend */}
            <div className="absolute top-3 right-3 z-30 bg-slate-950/90 border border-rose-500/50 rounded-lg p-2 shadow-xl text-[10px] font-mono flex flex-col gap-1 backdrop-blur-md">
              <div className="flex items-center justify-between gap-2 text-rose-300 font-bold">
                <span>Spectral Δ Intensity</span>
                <span className="text-cyan-300">{Math.round(heatmapOpacity * 100)}%</span>
              </div>
              <div className="w-36 h-2.5 rounded-full overflow-hidden border border-slate-700 bg-gradient-to-r from-blue-600 via-emerald-400 via-amber-400 to-rose-600" />
              <div className="flex justify-between text-[9px] text-slate-400">
                <span>Low Δ</span>
                <span>Peak Shift</span>
              </div>
            </div>

            {/* Metrics Badge */}
            {cvaMetrics && (
              <div className="absolute bottom-3 right-3 bg-slate-950/90 border border-amber-500/60 text-amber-300 text-[10px] font-mono px-2.5 py-1 rounded shadow flex items-center gap-2 z-30 backdrop-blur-md">
                {cvaMetrics.area_changed_pct != null && (
                  cvaMetrics.area_changed_pct < 2.0 ? (
                    <span className="text-emerald-300 font-semibold flex items-center gap-1">
                      <span>✓</span>
                      <span>Stable Scene (&lt;2% Δ)</span>
                    </span>
                  ) : (
                    <span>Area Shift: +{cvaMetrics.area_changed_pct}%</span>
                  )
                )}
                {cvaMetrics.mean_magnitude_pct != null && (
                  <>
                    <span>•</span>
                    <span>Mean Δ: {cvaMetrics.mean_magnitude_pct}%</span>
                  </>
                )}
                {cvaMetrics.confidence != null && (
                  <>
                    <span>•</span>
                    <span>Confidence: {Math.round(cvaMetrics.confidence * 100)}%</span>
                  </>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Bottom Quantitative Summary Bar with Strict Null Checks */}
      {(() => {
        const areaShift = cvaMetrics?.area_changed_pct ?? changeSummary?.area_changed_pct;
        const confidence = cvaMetrics?.confidence ?? changeSummary?.confidence;
        return (
          <div className="px-4 py-2.5 bg-slate-900/95 border-t border-slate-800 flex items-center justify-between text-xs text-slate-300 flex-wrap gap-2">
            <div className="flex items-center gap-3 flex-wrap">
              <span className="text-emerald-400 font-mono font-semibold flex items-center gap-1">
                <Activity className="w-3.5 h-3.5" />
                {areaShift != null ? (
                  <span>Measured Pixel Area Shift: +{areaShift}%</span>
                ) : (
                  <span className="text-slate-500 font-normal">Area shift: n/a</span>
                )}
              </span>
              {cvaMetrics?.mean_magnitude_pct != null && (
                <span className="text-amber-400 font-mono text-[11px] flex items-center gap-1">
                  <Flame className="w-3 h-3 text-rose-400" />
                  Mean Δ: {cvaMetrics.mean_magnitude_pct}%
                </span>
              )}
              <span className="text-slate-400">
                Engine: <strong className="text-slate-200">{cvaMetrics?.method ?? 'Optical Change Vector Analysis (CVA)'}</strong>
              </span>
            </div>
            <div className="flex items-center gap-1.5 text-cyan-300 font-mono text-[11px]">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              {confidence != null ? (
                <span>Confidence: {Math.round(confidence * 100)}%</span>
              ) : (
                <span className="text-slate-500">Confidence: n/a</span>
              )}
            </div>
          </div>
        );
      })()}
    </div>
  );
};
