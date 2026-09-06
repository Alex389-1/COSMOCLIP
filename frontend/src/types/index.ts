export interface BoundingBox {
  min_lon: number;
  min_lat: number;
  max_lon: number;
  max_lat: number;
}

export interface SceneMetadata {
  scene_id: string;
  scene_name?: string;
  collection: string;
  acquisition_time: string;
  cloud_cover: number;
  bbox: [number, number, number, number];
  crs: string;
  resolution_m: number;
  provider: string;
  thumbnail_url?: string;
  assets?: Record<string, any>;
}

export interface TraceStep {
  step: string;
  status: 'ok' | 'warning' | 'error' | 'skipped';
  latency_ms: number;
  detail?: string;
  timestamp?: string;
}

export interface EvidenceRegion {
  box_2d?: [number, number, number, number]; // [ymin, xmin, ymax, xmax]
  geo_bbox?: [number, number, number, number];
  polygon?: [number, number][]; // [[x, y], [x, y], ...] normalized 0-1
  polygons?: [number, number][][]; // Multiple disjoint polygon rings
  label: string;
  confidence: number;
  category?: string;
  color?: string;
  change_type?: 'road' | 'building' | 'water' | 'forest' | 'land';
}

export interface ModelMetadata {
  name: string;
  version: string;
  adapter?: string;
  runtime: string;
}

export interface ConfidenceInfo {
  score: number;
  category: 'High' | 'Moderate' | 'Low';
  is_calibrated: boolean;
  rationale: string;
}

export interface ChangeSummary {
  baseline_period: string;
  current_period: string;
  area_changed_pct: number;
  change_category: string;
  confidence: number;
}

export interface LocationMeta {
  name: string;
  lat: number;
  lon: number;
  bbox: [number, number, number, number];
  zoom?: number;
  display_name?: string;
}

export interface GroundTruthContext {
  location: string;
  summary: string;
  headlines: string[];
  sources: string[];
  retrieval_timestamp?: string;
}

export interface QueryRequest {
  question: string;
  location_name?: string;
  scene_id?: string;
  image_data_url?: string;
  bbox?: [number, number, number, number];
  viewport_bbox?: [number, number, number, number]; // [west, south, east, north]
  viewport_zoom?: number;
  /** ms epoch — for backend staleness check */
  viewport_captured_at?: number;
  enable_grounding?: boolean;
  enable_voice_response?: boolean;
  session_id?: string;
}

export interface QueryResponse {
  run_id: string;
  task: string;
  target_entity?: string;
  query_type?: string;
  should_recenter_map?: boolean;
  answer: string;
  spoken_text: string;
  is_comparison: boolean;
  is_submeter_highres?: boolean;
  is_new_location_query?: boolean;
  resolution_badge?: string;
  image_url?: string;
  optical_url?: string;
  sar_url?: string;
  sar_vv_url?: string;
  sar_vh_url?: string;
  fused_url?: string;
  sar_diff_url?: string;
  sar_metrics?: {
    mean_vv_db?: number;
    mean_vh_db?: number;
    vv_vh_ratio?: number;
    temporal_delta_vv_db?: number;
    temporal_delta_vh_db?: number;
    speckle_filter?: string;
    calibration?: string;
  };
  optical_metrics?: {
    sensor?: string;
    cloud_coverage_pct?: number;
    resolution_m?: number;
    acquisition_date?: string;
  };
  before_image_url?: string;
  after_image_url?: string;
  baseline_year?: string;
  baseline_period?: string;
  baseline_tile_url?: string;
  heatmap_url?: string;
  sar_heatmap_url?: string;
  heatmap_bounds?: [[number, number], [number, number]];
  cva_metrics?: {
    method?: string;
    area_changed_pct?: number;
    mean_magnitude_pct?: number;
    peak_magnitude_pct?: number;
    hotspot_coords?: [number, number];
    confidence?: number;
  };
  location_meta?: LocationMeta;
  change_summary?: ChangeSummary;
  ground_truth_context?: GroundTruthContext;
  confidence: ConfidenceInfo;
  evidence: EvidenceRegion[];
  scene_id?: string;
  model: ModelMetadata;
  trace: TraceStep[];
  audio_base64?: string;
  /** Spatial intent classification: 'navigation' | 'viewport_bound' | 'followup' */
  query_intent?: 'navigation' | 'viewport_bound' | 'followup';
  /** Ground sample distance of the imagery used for this answer (metres) */
  image_gsd_m?: number;
}

export interface ToolCapability {
  tool_id: string;
  name: string;
  description: string;
  version: string;
  task_types: string[];
  modalities: string[];
  outputs: string[];
  runtime: string;
  is_active: boolean;
}
