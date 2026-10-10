import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { api } from "../api/client";
import type { NetworkGeoJSON } from "../api/client";
import { isQueuedSpeed, signalKind, summarizeSignals } from "../utils/trajectory";

export interface VehiclePoint {
  vehicle_id: string;
  x: number;
  y: number;
  speed: number;
}

interface NetworkCanvasProps {
  height?: number;
  tlsStates?: Record<string, { phase: number; state: string }>;
  vehicles?: VehiclePoint[];
  currentTime?: number;
}

const SIGNAL_COLORS = { green: "#10b981", yellow: "#f59e0b", red: "#ef4444" } as const;
// Junctions with no recorded state are drawn as a neutral grey ring, never as a
// guessed phase (the old schedule/hash fallbacks fabricated signal colours).
const NO_SIGNAL_COLOR = "#94a3b8";

export const NetworkCanvas: React.FC<NetworkCanvasProps> = ({
  height = 380,
  tlsStates = {},
  vehicles = [],
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [geoJSON, setGeoJSON] = useState<NetworkGeoJSON | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const animFrameRef = useRef<number>(0);
  const tickRef = useRef(0);
  const dprRef = useRef(typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1);
  const reducedMotionRef = useRef(
    typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
  );

  useEffect(() => {
    // api.getNetworkGeoJSON is a module-level cached promise: every canvas
    // shares ONE download of the 4.9 MB GeoJSON per session.
    let cancelled = false;
    api.getNetworkGeoJSON()
      .then((geo) => { if (!cancelled) setGeoJSON(geo); })
      .catch((e: Error) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  const tlsIds = useMemo(
    () => (geoJSON?.features ?? [])
      .filter((f) => f.geometry.type === "Point" && f.properties.is_tls)
      .map((f) => f.properties.tls_id ?? ""),
    [geoJSON],
  );
  const signalSummary = useMemo(() => summarizeSignals(tlsIds, tlsStates), [tlsIds, tlsStates]);
  const stoppedCount = useMemo(() => vehicles.filter((v) => isQueuedSpeed(v.speed)).length, [vehicles]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !geoJSON) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Backing store is scaled by devicePixelRatio (sharp on retina); draw in CSS px.
    const dpr = dprRef.current;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;
    tickRef.current += 0.05;
    const t = tickRef.current;

    // Filter out outlying road branches for crisp Ingolstadt center view
    // Main city bounds: X ~ 209500 to 218500, Y ~ 446500 to 455500
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const feat of geoJSON.features) {
      if (feat.geometry.type === "LineString") {
        const coords = feat.geometry.coordinates as number[][];
        for (const [cx, cy] of coords) {
          if (cx < 219500 && cy < 456500) {
            if (cx < minX) minX = cx;
            if (cx > maxX) maxX = cx;
            if (cy < minY) minY = cy;
            if (cy > maxY) maxY = cy;
          }
        }
      }
    }

    if (!isFinite(minX)) {
      minX = 209500; maxX = 218500; minY = 446500; maxY = 455500;
    }

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const pad = 24;

    const scaleX = (W - pad * 2) / rangeX;
    const scaleY = (H - pad * 2) / rangeY;
    const scale = Math.min(scaleX, scaleY);

    // Center the map inside the canvas viewport
    const offsetX = pad + (W - pad * 2 - rangeX * scale) / 2;
    const offsetY = pad + (H - pad * 2 - rangeY * scale) / 2;

    const toScreen = (x: number, y: number): [number, number] => [
      offsetX + (x - minX) * scale,
      H - offsetY - (y - minY) * scale,  // flip Y axis
    ];

    // Clear Background
    ctx.clearRect(0, 0, W, H);
    const bg = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) / 1.4);
    bg.addColorStop(0, "#0b1526");
    bg.addColorStop(1, "#050811");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Subtle background grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.02)";
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let gx = 0; gx < W; gx += gridSize) {
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }
    for (let gy = 0; gy < H; gy += gridSize) {
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }

    // 1. Draw Roads & Lanes (Crisp road network)
    for (const feat of geoJSON.features) {
      if (feat.geometry.type !== "LineString") continue;
      const coords = feat.geometry.coordinates as number[][];
      const isTLS = feat.properties.is_tls;

      ctx.beginPath();
      for (let i = 0; i < coords.length; i++) {
        const [sx, sy] = toScreen(coords[i][0], coords[i][1]);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.strokeStyle = isTLS ? "rgba(56, 189, 248, 0.45)" : "rgba(100, 116, 139, 0.25)";
      ctx.lineWidth = isTLS ? 1.4 : 0.75;
      ctx.stroke();
    }

    // 2. Draw Active Vehicles (Moving dots in magenta / lavender / cyan)
    // Drawn BEFORE traffic lights so signals sit prominently on top at intersections!
    if (vehicles && vehicles.length > 0) {
      for (const veh of vehicles) {
        const [vx, vy] = toScreen(veh.x, veh.y);
        if (vx < -10 || vx > W + 10 || vy < -10 || vy > H + 10) continue;

        ctx.beginPath();
        const isQueue = isQueuedSpeed(veh.speed);
        // Queued (<2 m/s): Hot Pink/Magenta (#ec4899), Moderate (2-8 m/s): Lavender (#a855f7), Cruising (>8 m/s): Cyan (#38bdf8)
        const carColor = isQueue ? "#ec4899" : veh.speed > 8 ? "#38bdf8" : "#a855f7";

        ctx.arc(vx, vy, isQueue ? 2.8 : 2.2, 0, Math.PI * 2);
        ctx.fillStyle = carColor;
        ctx.fill();

        if (isQueue) {
          ctx.strokeStyle = "rgba(236, 72, 153, 0.5)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }

    // 3. Draw TLS Junctions (Traffic Lights) — ON TOP OF VEHICLES & ROADS
    for (const feat of geoJSON.features) {
      if (feat.geometry.type !== "Point") continue;
      // Only draw signalized intersections
      if (!feat.properties.is_tls) continue;

      const coords = feat.geometry.coordinates as number[];
      const [sx, sy] = toScreen(coords[0], coords[1]);
      if (sx < -20 || sx > W + 20 || sy < -20 || sy > H + 20) continue;

      const tlsId = feat.properties.tls_id ?? "";
      // Only a real recorded state is coloured; otherwise neutral "no signal data".
      const kind = signalKind(tlsStates[tlsId]?.state);
      const signalColor = kind ? SIGNAL_COLORS[kind] : NO_SIGNAL_COLOR;

      // ── Traffic Light Visual Badge Rendering ───────────────────────────
      // 1. Ambient pulsing halo in signal color
      const pulse = reducedMotionRef.current ? 0.85 : 0.7 + 0.3 * Math.abs(Math.sin(t * 1.5 + sx * 0.03));
      const haloR = 10 * pulse;
      if (kind) {
        const glow = ctx.createRadialGradient(sx, sy, 2, sx, sy, haloR);
        glow.addColorStop(0, signalColor + "99");
        glow.addColorStop(1, signalColor + "00");
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(sx, sy, haloR, 0, Math.PI * 2);
        ctx.fill();
      }

      // 2. High-contrast protective collar (dark bezel + crisp white outline)
      ctx.beginPath();
      ctx.arc(sx, sy, 4.8, 0, Math.PI * 2);
      ctx.fillStyle = "#030712";
      ctx.fill();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // 3. Bright saturated LED signal lens
      ctx.beginPath();
      ctx.arc(sx, sy, 3.2, 0, Math.PI * 2);
      ctx.fillStyle = kind ? signalColor : "#1e293b"; // dark, ringed lens = no recorded state
      ctx.fill();
      if (!kind) {
        ctx.strokeStyle = NO_SIGNAL_COLOR;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // 4. Specular highlight (glass reflection)
      ctx.beginPath();
      ctx.arc(sx - 1, sy - 1, 0.9, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      ctx.fill();
    }
  }, [geoJSON, tlsStates, vehicles]);

  useEffect(() => {
    if (!geoJSON) return;
    let running = true;
    const loop = () => {
      if (!running) return;
      draw();
      animFrameRef.current = requestAnimationFrame(loop);
    };
    animFrameRef.current = requestAnimationFrame(loop);
    return () => {
      running = false;
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [geoJSON, draw]);

  // Resize canvas to match container
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = (cssWidth: number) => {
      const dpr = window.devicePixelRatio || 1;
      dprRef.current = dpr;
      canvas.width = Math.round(cssWidth * dpr);
      canvas.height = Math.round(height * dpr);
    };
    const ro = new ResizeObserver(() => {
      resize(canvas.offsetWidth);
      draw();
    });
    resize(canvas.parentElement?.offsetWidth || 800);
    if (canvas.parentElement) ro.observe(canvas.parentElement);
    return () => ro.disconnect();
  }, [height, draw]);

  if (loading) {
    return (
      <div className="network-canvas-wrapper" style={{ height, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ textAlign: "center", color: "var(--text-muted)" }}>
          <span className="spinner" style={{ width: 28, height: 28, borderTopColor: "var(--accent-cyan)" }} />
          <p style={{ marginTop: "0.75rem", fontSize: "0.875rem" }}>Loading InTAS Ingolstadt network &amp; signals…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="network-canvas-wrapper" style={{ height, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ textAlign: "center", color: "var(--text-muted)" }}>
          <p style={{ color: "var(--signal-red)", marginBottom: "0.5rem" }}>⚠ Network map unavailable</p>
          <p style={{ fontSize: "0.8rem" }}>Start the API server to load the GeoJSON</p>
        </div>
      </div>
    );
  }

  return (
    <div className="network-canvas-wrapper" style={{ height, position: "relative", borderRadius: 8, overflow: "hidden" }}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={
          `Ingolstadt road network map. Signals: ${signalSummary.green} green, ${signalSummary.yellow} yellow, ` +
          `${signalSummary.red} red, ${signalSummary.unknown} with no signal data.` +
          (vehicles.length ? ` ${vehicles.length} vehicles, ${stoppedCount} stopped.` : "")
        }
        style={{ width: "100%", height: "100%", display: "block" }}
      />
      <div style={{
        position: "absolute", bottom: 10, right: 12, fontSize: "0.72rem",
        color: "var(--text-muted)", fontFamily: "var(--font-mono)",
        display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap",
        background: "rgba(8, 12, 22, 0.92)", padding: "0.4rem 0.8rem", borderRadius: 8,
        border: "1px solid var(--border-subtle)", backdropFilter: "blur(8px)",
        boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: 5, color: "#10b981", fontWeight: 600 }}>
          <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: "50%", background: "#10b981", border: "1.5px solid #fff", boxShadow: "0 0 6px #10b981" }} /> Green
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 5, color: "#f59e0b", fontWeight: 600 }}>
          <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: "50%", background: "#f59e0b", border: "1.5px solid #fff", boxShadow: "0 0 6px #f59e0b" }} /> Yellow
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 5, color: "#ef4444", fontWeight: 600 }}>
          <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: "50%", background: "#ef4444", border: "1.5px solid #fff", boxShadow: "0 0 6px #ef4444" }} /> Red
        </span>
        {signalSummary.unknown > 0 && (
          <span style={{ display: "flex", alignItems: "center", gap: 5, color: NO_SIGNAL_COLOR, fontWeight: 600 }}>
            <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: "50%", background: "transparent", border: `1.5px solid ${NO_SIGNAL_COLOR}` }} /> No signal data
          </span>
        )}
        {vehicles.length > 0 && (
          <>
            <span style={{ color: "var(--border-subtle)" }}>│</span>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "#38bdf8" }} /> Fast Car
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "#a855f7" }} /> Moving Car
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "#ec4899" }} /> Stopped Car
            </span>
            <span style={{ color: "var(--accent-cyan)", fontWeight: 600 }}>({vehicles.length} veh)</span>
          </>
        )}
      </div>
    </div>
  );
};
