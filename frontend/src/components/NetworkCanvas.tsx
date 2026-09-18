import React, { useEffect, useRef, useState, useCallback } from "react";
import { api } from "../api/client";
import type { NetworkGeoJSON } from "../api/client";

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
}

function getSignalColor(state: string): string {
  if (!state) return "#22d3ee";
  const s = state.toLowerCase();
  if (s.includes("g")) return "#10b981";
  if (s.includes("y")) return "#f59e0b";
  if (s.includes("r")) return "#ef4444";
  return "#22d3ee";
}

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

  useEffect(() => {
    api.getNetworkGeoJSON()
      .then(setGeoJSON)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !geoJSON) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
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

    // 2. Draw TLS Junctions (Traffic Lights)
    for (const feat of geoJSON.features) {
      if (feat.geometry.type !== "Point") continue;
      // Only draw real signalized intersections (is_tls)
      if (!feat.properties.is_tls) continue;

      const coords = feat.geometry.coordinates as number[];
      const [sx, sy] = toScreen(coords[0], coords[1]);
      if (sx < -20 || sx > W + 20 || sy < -20 || sy > H + 20) continue;

      const tlsId = feat.properties.tls_id ?? "";
      const stateInfo = tlsStates[tlsId];
      const color = stateInfo ? getSignalColor(stateInfo.state) : "#22d3ee";

      // Subtle pulse glow
      const pulse = 0.5 + 0.5 * Math.abs(Math.sin(t + sx * 0.02));
      const glowR = 6 * pulse;
      const grd = ctx.createRadialGradient(sx, sy, 0, sx, sy, glowR);
      grd.addColorStop(0, color + "66");
      grd.addColorStop(1, color + "00");
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(sx, sy, glowR, 0, Math.PI * 2);
      ctx.fill();

      // Sharp core pin
      ctx.beginPath();
      ctx.arc(sx, sy, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }

    // 3. Draw Active Vehicles
    if (vehicles && vehicles.length > 0) {
      for (const veh of vehicles) {
        const [vx, vy] = toScreen(veh.x, veh.y);
        if (vx < -10 || vx > W + 10 || vy < -10 || vy > H + 10) continue;

        ctx.beginPath();
        // Queued (<2 m/s): Red, Slow (2-8 m/s): Amber, Cruising (>8 m/s): Emerald
        const isQueue = veh.speed < 2;
        const color = isQueue ? "#ef4444" : veh.speed > 8 ? "#10b981" : "#f59e0b";

        ctx.arc(vx, vy, isQueue ? 2.8 : 2.2, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Extra highlight on queued cars
        if (isQueue) {
          ctx.strokeStyle = "rgba(239, 68, 68, 0.4)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
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
    const ro = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth;
      canvas.height = height;
      draw();
    });
    if (canvas.parentElement) {
      ro.observe(canvas.parentElement);
      canvas.width = canvas.parentElement.offsetWidth || 800;
    }
    canvas.height = height;
    return () => ro.disconnect();
  }, [height, draw]);

  if (loading) {
    return (
      <div className="network-canvas-wrapper" style={{ height, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ textAlign: "center", color: "var(--text-muted)" }}>
          <span className="spinner" style={{ width: 28, height: 28, borderTopColor: "var(--accent-cyan)" }} />
          <p style={{ marginTop: "0.75rem", fontSize: "0.875rem" }}>Loading InTAS Ingolstadt network…</p>
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
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
      <div style={{
        position: "absolute", bottom: 10, right: 12, fontSize: "0.72rem",
        color: "var(--text-muted)", fontFamily: "var(--font-mono)",
        display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap",
        background: "rgba(10,15,28,0.85)", padding: "0.3rem 0.65rem", borderRadius: 6,
        border: "1px solid var(--border-subtle)", backdropFilter: "blur(6px)",
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ display: "inline-block", width: 7, height: 7, borderRadius: "50%", background: "#10b981" }} /> Green
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ display: "inline-block", width: 7, height: 7, borderRadius: "50%", background: "#f59e0b" }} /> Yellow
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ display: "inline-block", width: 7, height: 7, borderRadius: "50%", background: "#ef4444" }} /> Red
        </span>
        {vehicles.length > 0 && (
          <>
            <span style={{ color: "var(--border-subtle)" }}>|</span>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "#10b981" }} /> Cruising
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "#ef4444" }} /> Stopped/Queue
            </span>
            <span style={{ color: "var(--accent-cyan)", fontWeight: 600 }}>{vehicles.length} veh</span>
          </>
        )}
      </div>
    </div>
  );
};
