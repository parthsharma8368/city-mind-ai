/**
 * Map drawing for the showcase panels: the road network is drawn ONCE per canvas size into an offscreen layer; every
 * animation frame then only blits that layer and draws the cars and signals on top (the Replay page's canvas redraws all
 * ~16,000 lane polylines on every frame, which cannot keep four panels at 60 fps).
 */
import type { NetworkGeoJSON } from "../api/client";

export interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface Projection {
  /** screen px = ax + x * scale, ay - y * scale for x, y in metres relative to `origin`. */
  scale: number;
  ax: number;
  ay: number;
  width: number;
  height: number;
}

// The city centre of InTAS: far-out road branches would shrink the interesting part to a dot.
const CITY_MAX_X = 219_500;
const CITY_MAX_Y = 456_500;
const FALLBACK: Bounds = { minX: 209_500, maxX: 218_500, minY: 446_500, maxY: 455_500 };

export function computeBounds(geo: NetworkGeoJSON): Bounds {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const feat of geo.features) {
    if (feat.geometry.type !== "LineString") continue;
    for (const [cx, cy] of feat.geometry.coordinates as number[][]) {
      if (cx >= CITY_MAX_X || cy >= CITY_MAX_Y) continue;
      if (cx < minX) minX = cx;
      if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;
    }
  }
  return Number.isFinite(minX) ? { minX, maxX, minY, maxY } : FALLBACK;
}

/**
 * Fit the bounds into width x height (CSS px) with padding, centred. Cars arrive as metres relative to `origin`, so the
 * origin offset is folded into the constants once: px = ax + x * scale.
 */
export function makeProjection(b: Bounds, width: number, height: number, origin: [number, number], pad = 14): Projection {
  const rangeX = b.maxX - b.minX || 1;
  const rangeY = b.maxY - b.minY || 1;
  const scale = Math.min((width - pad * 2) / rangeX, (height - pad * 2) / rangeY);
  const offX = pad + (width - pad * 2 - rangeX * scale) / 2;
  const offY = pad + (height - pad * 2 - rangeY * scale) / 2;
  return {
    scale,
    ax: offX + (origin[0] - b.minX) * scale,
    ay: height - offY - (origin[1] - b.minY) * scale,
    width,
    height,
  };
}

export function projectPoint(p: Projection, x: number, y: number): [number, number] {
  return [p.ax + x * p.scale, p.ay - y * p.scale];
}

/** Junction positions (metres relative to `origin`) keyed by tls id. */
export function junctionPositions(geo: NetworkGeoJSON, origin: [number, number]): Map<string, [number, number]> {
  const out = new Map<string, [number, number]>();
  for (const feat of geo.features) {
    if (feat.geometry.type !== "Point" || !feat.properties.is_tls || !feat.properties.tls_id) continue;
    const [x, y] = feat.geometry.coordinates as number[];
    out.set(feat.properties.tls_id, [x - origin[0], y - origin[1]]);
  }
  return out;
}

/** Draw the static road layer (background, roads) at CSS size width x height, scaled by dpr. */
export function drawRoadLayer(
  ctx: CanvasRenderingContext2D,
  geo: NetworkGeoJSON,
  proj: Projection,
  origin: [number, number],
  dpr: number,
): void {
  const { width: W, height: H } = proj;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const bg = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.75);
  bg.addColorStop(0, "#0c1424");
  bg.addColorStop(1, "#05080f");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  const draw = (tlsLanes: boolean) => {
    ctx.beginPath();
    for (const feat of geo.features) {
      if (feat.geometry.type !== "LineString") continue;
      if (!!feat.properties.is_tls !== tlsLanes) continue;
      const coords = feat.geometry.coordinates as number[][];
      for (let i = 0; i < coords.length; i++) {
        const sx = proj.ax + (coords[i][0] - origin[0]) * proj.scale;
        const sy = proj.ay - (coords[i][1] - origin[1]) * proj.scale;
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
    }
    ctx.strokeStyle = tlsLanes ? "rgba(96, 165, 250, 0.34)" : "rgba(120, 140, 175, 0.20)";
    ctx.lineWidth = tlsLanes ? 1.0 : 0.7;
    ctx.stroke();
  };
  draw(false);
  draw(true);

  // soft vignette so the map melts into the panel
  const vg = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.max(W, H) * 0.78);
  vg.addColorStop(0, "rgba(5,8,15,0)");
  vg.addColorStop(1, "rgba(5,8,15,0.72)");
  ctx.fillStyle = vg;
  ctx.fillRect(0, 0, W, H);
}
