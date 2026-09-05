import { SceneMetadata, QueryResponse, ToolCapability, BoundingBox, LocationMeta } from '../types';

const API_BASE = '/api';

export async function fetchHealth(): Promise<{ status: string; capabilities: Record<string, boolean> }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Backend health check failed');
  return res.json();
}

export async function fetchAvailableScenes(): Promise<SceneMetadata[]> {
  const res = await fetch(`${API_BASE}/imagery/scenes`);
  if (!res.ok) throw new Error('Failed to load satellite scenes');
  return res.json();
}

export async function geocodeLocation(query: string): Promise<{
  name: string;
  lat: number;
  lon: number;
  bbox: [number, number, number, number];
  zoom: number;
  image_url: string;
}> {
  const res = await fetch(`${API_BASE}/imagery/geocode`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error('Failed to geocode location');
  return res.json();
}

export async function searchScenes(
  bbox?: BoundingBox,
  maxCloudCover: number = 20.0,
  limit: number = 6
): Promise<SceneMetadata[]> {
  const res = await fetch(`${API_BASE}/imagery/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      bbox: bbox
        ? {
            min_lon: bbox.min_lon,
            min_lat: bbox.min_lat,
            max_lon: bbox.max_lon,
            max_lat: bbox.max_lat,
          }
        : undefined,
      max_cloud_cover: maxCloudCover,
      limit: limit,
    }),
  });
  if (!res.ok) throw new Error('Failed to search Copernicus STAC scenes');
  return res.json();
}

export async function submitVQAQuery(payload: {
  question: string;
  scene_id?: string;
  location_name?: string;
  image_data_url?: string;
  bbox?: [number, number, number, number];
  viewport_bbox?: [number, number, number, number];
  viewport_zoom?: number;
  enable_grounding?: boolean;
  enable_voice_response?: boolean;
  session_id?: string;
}): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Query failed' }));
    throw new Error(err.detail || 'Failed to execute query');
  }
  return res.json();
}

export async function fetchModelRegistry(): Promise<{ status: string; count: number; tools: ToolCapability[] }> {
  const res = await fetch(`${API_BASE}/models/registry`);
  if (!res.ok) throw new Error('Failed to load tool registry');
  return res.json();
}

export async function fetchVoiceConfig(): Promise<{
  gemini_live_available: boolean;
  gemini_live_model: string;
  local_web_speech_supported: boolean;
  browser_tts_supported: boolean;
  default_mode: string;
}> {
  const res = await fetch(`${API_BASE}/voice/config`);
  if (!res.ok) throw new Error('Failed to load voice configuration');
  return res.json();
}
