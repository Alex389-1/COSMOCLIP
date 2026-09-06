import L from 'leaflet';
import html2canvas from 'html2canvas';

/**
 * Downscales a canvas if either dimension exceeds maxDim (default 1024px)
 */
function downscaleCanvas(canvas: HTMLCanvasElement, maxDim = 1024): HTMLCanvasElement {
  const { width, height } = canvas;
  if (width <= maxDim && height <= maxDim) return canvas;
  const ratio = Math.min(maxDim / width, maxDim / height);
  const out = document.createElement('canvas');
  out.width = Math.round(width * ratio);
  out.height = Math.round(height * ratio);
  const ctx = out.getContext('2d');
  if (ctx) {
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(canvas, 0, 0, out.width, out.height);
  }
  return out;
}

let activeMapInstance: L.Map | null = null;

export function setActiveMap(map: L.Map | null): void {
  activeMapInstance = map;
  if (map && typeof window !== 'undefined') {
    (window as any).__ACTIVE_LEAFLET_MAP__ = map;
  }
}

export function getActiveMap(): L.Map | null {
  if (activeMapInstance) return activeMapInstance;
  if (typeof window !== 'undefined' && (window as any).__ACTIVE_LEAFLET_MAP__) {
    return (window as any).__ACTIVE_LEAFLET_MAP__;
  }
  return null;
}

/**
 * Fetches a satellite image URL (backend-served) and converts it to a base64 JPEG data URL.
 * Used when the user is on the Raster Scene mode viewing a static satellite image.
 */
export async function fetchUrlAsBase64(url: string): Promise<string> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch image: ${resp.status}`);
  const blob = await resp.blob();
  const imgBitmap = await createImageBitmap(blob);
  const maxDim = 1024;
  const ratio = Math.min(maxDim / imgBitmap.width, maxDim / imgBitmap.height, 1.0);
  const w = Math.round(imgBitmap.width * ratio);
  const h = Math.round(imgBitmap.height * ratio);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Could not get 2D context');
  ctx.drawImage(imgBitmap, 0, 0, w, h);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.88);
  console.log(`[fetchUrlAsBase64] ${imgBitmap.width}x${imgBitmap.height} → ${w}x${h}: ${Math.round(dataUrl.length / 1024)}KB`);
  return dataUrl;
}

/**
 * Loads an image tile from ArcGIS World Imagery with CORS enabled.
 */
function loadTile(z: number, x: number, y: number): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Tile load failed: ${z}/${x}/${y}`));
    img.src = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`;
    setTimeout(() => reject(new Error(`Tile timeout: ${z}/${x}/${y}`)), 4000);
  });
}

/**
 * Loads a reference places and boundaries tile from ArcGIS with CORS enabled.
 * Contains park names, playground labels, roads, and civic landmarks.
 */
function loadRefTile(z: number, x: number, y: number): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Ref tile failed: ${z}/${x}/${y}`));
    img.src = `https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/${z}/${y}/${x}`;
    setTimeout(() => reject(new Error(`Ref tile timeout: ${z}/${x}/${y}`)), 4000);
  });
}

/**
 * Primary capture: composites ArcGIS World Imagery tiles to match exactly what the
 * Leaflet map is showing at the current zoom/center. Uses Leaflet's map.project()
 * world pixel coordinate math so that the captured canvas matches the user's viewport
 * with 100% pixel-perfect precision and no tile offsets or aspect ratio distortion.
 */
async function captureViaTileCompositor(map: L.Map): Promise<string> {
  const zoom = Math.round(map.getZoom());
  const center = map.getCenter();
  const size = map.getSize();
  const containerW = Math.max(size.x, 300);
  const containerH = Math.max(size.y, 300);

  // Preserve the exact viewport aspect ratio, capping max dimension at 1280 for fast VLM transmission
  const scale = Math.min(1.0, 1280 / Math.max(containerW, containerH));
  const canvasW = Math.round(containerW * scale);
  const canvasH = Math.round(containerH * scale);

  // Exact global pixel coordinates at this zoom level
  const centerPx = map.project(center, zoom);
  const topLeftPx = {
    x: centerPx.x - containerW / 2,
    y: centerPx.y - containerH / 2,
  };

  const minTx = Math.floor(topLeftPx.x / 256);
  const maxTx = Math.floor((topLeftPx.x + containerW) / 256);
  const minTy = Math.floor(topLeftPx.y / 256);
  const maxTy = Math.floor((topLeftPx.y + containerH) / 256);

  const canvas = document.createElement('canvas');
  canvas.width = canvasW;
  canvas.height = canvasH;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('No 2D context');

  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, canvasW, canvasH);

  const n = Math.pow(2, zoom);
  const tileTasks: Promise<void>[] = [];

  for (let ty = minTy; ty <= maxTy; ty++) {
    for (let tx = minTx; tx <= maxTx; tx++) {
      if (ty < 0 || ty >= n) continue;
      const wrappedTx = ((tx % n) + n) % n;

      const drawX = Math.round((tx * 256 - topLeftPx.x) * scale);
      const drawY = Math.round((ty * 256 - topLeftPx.y) * scale);
      const drawW = Math.ceil(256 * scale);
      const drawH = Math.ceil(256 * scale);

      const task = async () => {
        try {
          const satImg = await loadTile(zoom, wrappedTx, ty);
          ctx.drawImage(satImg, drawX, drawY, drawW, drawH);
        } catch {
          // Leave background if tile fetch failed
        }

        try {
          const refImg = await loadRefTile(zoom, wrappedTx, ty);
          ctx.drawImage(refImg, drawX, drawY, drawW, drawH);
        } catch {
          // Labels optional
        }
      };
      tileTasks.push(task());
    }
  }

  await Promise.all(tileTasks);

  const dataUrl = canvas.toDataURL('image/jpeg', 0.90);
  console.log(
    `[captureViaTileCompositor] center=(${center.lat.toFixed(4)}, ${center.lng.toFixed(4)}), zoom=${zoom}, ` +
    `canvas=${canvasW}x${canvasH}, size=${Math.round(dataUrl.length / 1024)}KB`
  );
  return dataUrl;
}

/**
 * Captures the current visible Leaflet map view as a base64 JPEG data URL.
 */
export async function captureCurrentView(map?: L.Map | null): Promise<string> {
  let targetMap = map || getActiveMap();

  if (!targetMap && typeof document !== 'undefined') {
    const el = document.querySelector('.leaflet-container') as any;
    if (el && el._leaflet_map) {
      targetMap = el._leaflet_map;
    }
  }

  if (!targetMap) {
    throw new Error('No Leaflet map instance available for screenshot capture');
  }

  // Strategy 1: tile compositor (accurate, CORS-safe, includes reference labels)
  try {
    const dataUrl = await captureViaTileCompositor(targetMap);
    if (dataUrl.length > 10000) {
      if ((import.meta as any).env?.DEV) {
        (window as any).__LAST_CAPTURED_VIEW__ = dataUrl;
      }
      return dataUrl;
    }
    throw new Error('Tile compositor produced a suspiciously small image');
  } catch (tileErr) {
    console.warn('[captureView] Tile compositor failed, falling back to html2canvas:', tileErr);
  }

  // Strategy 2: html2canvas DOM rasterizer fallback
  const container = targetMap?.getContainer?.() || (document.querySelector('.leaflet-container') as HTMLElement);
  if (!container) throw new Error('No map DOM container for html2canvas');
  const canvas = await html2canvas(container, { useCORS: true, allowTaint: false, logging: false });
  const scaled = downscaleCanvas(canvas, 1024);
  const dataUrl = scaled.toDataURL('image/jpeg', 0.85);
  console.log(`[captureCurrentView] html2canvas fallback: ${Math.round(dataUrl.length / 1024)}KB`);
  if ((import.meta as any).env?.DEV) {
    (window as any).__LAST_CAPTURED_VIEW__ = dataUrl;
  }
  return dataUrl;
}

/**
 * Determines whether a query is asking for visual/spatial analysis of the active screen view,
 * versus a simple location lookup or map navigation command.
 *
 * Screen analysis queries that REQUIRE capturing screenshot:
 * - "what do you see", "can you see cars", "describe the building on screen"
 * - "is there a playground", "how many buildings are here", "analyze this area"
 * - "is the building hexagonal", "what is this structure", "compare changes"
 *
 * Simple location / navigation queries that should NEVER capture screenshot:
 * - "show this location on map", "show tokyo on map", "take me to Paris"
 * - "where is KIET college", "navigate to London", "zoom to level 18", "Dubai"
 */
export function isScreenAnalysisQuery(question: string): boolean {
  if (!question) return false;
  const q = question.toLowerCase().trim();

  // 1. Explicit pure navigation / map showing patterns -> NOT screen analysis
  const isPureNavigation =
    /^(show|take|navigate|go|fly|zoom|where is|where's|locate|find)\s+(me\s+to\s+|to\s+|the\s+location\s+|this\s+location\s+on\s+map|[\w\s,]+on\s+map)/i.test(q)
    || /^(where\s+is|where's|locate\s+|find\s+)/i.test(q)
    || /show\s+([\w\s]+)\s+on\s+map/i.test(q)
    || /^take\s+me\s+to\b/i.test(q)
    || /^navigate\s+to\b/i.test(q)
    || /^zoom\s+to\b/i.test(q)
    || /^show\s+this\s+location/i.test(q);

  // Analytical keywords that denote visual feature extraction, counting, or description
  const hasAnalysisPrompt = /\b(what|describe|tell me about|analyze|analysis|count|how many|is there|are there|can you see|do you see|structure|features|identify|detect|compare|difference|change|changes|playground|car|cars|vehicle|vehicles|building|buildings|road|roads|street|trees|vegetation|water|roof|hexagonal|stadium|park|bridge)\b/i.test(q);

  // If it's a pure navigation command without explicit visual inspection prompt, return false
  if (isPureNavigation && !hasAnalysisPrompt) {
    return false;
  }

  // 2. Clear deictic & visual inspection triggers for current view/screen
  const hasDeicticOrScreen = /\b(this|these|here|current|screen|view|viewport|visible|seen|image|photo|satellite view)\b/i.test(q);
  const hasVisualTarget = /\b(building|buildings|structure|house|houses|road|roads|street|car|cars|vehicle|vehicles|playground|park|field|water|river|lake|bridge|trees|vegetation|area|land|site|zone|feature|features|roof|hexagonal)\b/i.test(q);

  // Questions like "describe the building on screen", "what is this structure", "analyze this area", "can you see cars"
  if (hasAnalysisPrompt && (hasDeicticOrScreen || hasVisualTarget)) {
    return true;
  }

  // Direct VQA / inspection question beginnings
  if (/^(what is this|what do you see|what can you see|describe what|what's here|what is here|can you see|are cars visible|is there any|how many|analyze|tell me about the)/i.test(q)) {
    return true;
  }

  // Compare / change detection
  if (/\b(compare|change|changes|difference|heatmap|differ)\b/i.test(q)) {
    return true;
  }

  // Pure navigation or command verbs default to false
  if (/^(show|take|navigate|go|fly|zoom|where)\b/i.test(q)) {
    return false;
  }

  return false;
}
