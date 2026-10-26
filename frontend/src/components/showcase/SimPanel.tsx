import React, { useEffect, useMemo, useRef } from "react";
import type { NetworkGeoJSON } from "../../api/client";
import { Interpolator, frameIndexAt, makeCarSample, STOPPED_MS, type FrameSet } from "../../showcase/frames";
import { drawRoadLayer, junctionPositions, makeProjection, type Bounds } from "../../showcase/mapView";

/** What the page's single animation loop calls on every panel. */
export interface PanelHandle {
  draw(simTime: number): void;
}

interface Props {
  id: string;
  label: string;
  kind: "baseline" | "model";
  color: string;
  frames: FrameSet;
  geo: NetworkGeoJSON;
  bounds: Bounds;
  /** Largest stopped-car count of any panel, so every sparkline shares one scale and the panels compare at a glance. */
  maxStopped: number;
  register: (id: string, handle: PanelHandle | null) => void;
}

const SIGNAL = { g: "#34d399", y: "#fbbf24", r: "#f87171" } as const;
const SLOW_MS = 6;

export const SimPanel: React.FC<Props> = React.memo(({ id, label, kind, color, frames, geo, bounds, maxStopped, register }) => {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layerRef = useRef<HTMLCanvasElement | null>(null);
  const projRef = useRef<ReturnType<typeof makeProjection> | null>(null);
  const dprRef = useRef(1);
  const carsEl = useRef<HTMLSpanElement>(null);
  const stoppedEl = useRef<HTMLSpanElement>(null);
  const speedEl = useRef<HTMLSpanElement>(null);
  const cursorEl = useRef<SVGLineElement>(null);
  const statsAt = useRef(0);

  const interp = useMemo(() => new Interpolator(frames), [frames]);
  const sample = useMemo(() => makeCarSample(frames), [frames]);
  const junctions = useMemo(() => {
    const pos = junctionPositions(geo, frames.origin);
    return frames.tlsIds.map((tid) => pos.get(tid) ?? null);
  }, [geo, frames]);

  const sparkPoints = useMemo(() => {
    const n = frames.stopped.length;
    const top = Math.max(1, maxStopped);
    return Array.from(frames.stopped, (s, k) => `${((k / Math.max(1, n - 1)) * 100).toFixed(2)},${(22 - (s / top) * 20).toFixed(2)}`).join(" ");
  }, [frames, maxStopped]);

  // Road layer: rebuilt only when the panel changes size.
  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const build = () => {
      const width = Math.max(120, Math.round(wrap.clientWidth));
      const height = Math.round(width * 0.84);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      dprRef.current = dpr;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const layer = document.createElement("canvas");
      layer.width = canvas.width;
      layer.height = canvas.height;
      const proj = makeProjection(bounds, width, height, frames.origin);
      const lctx = layer.getContext("2d");
      if (lctx) drawRoadLayer(lctx, geo, proj, frames.origin, dpr);
      layerRef.current = layer;
      projRef.current = proj;
    };
    build();
    let timer = 0;
    const ro = new ResizeObserver(() => {
      window.clearTimeout(timer);
      timer = window.setTimeout(build, 120);
    });
    ro.observe(wrap);
    return () => {
      window.clearTimeout(timer);
      ro.disconnect();
    };
  }, [geo, bounds, frames.origin]);

  useEffect(() => {
    const handle: PanelHandle = {
      draw(simTime: number) {
        const canvas = canvasRef.current;
        const layer = layerRef.current;
        const proj = projRef.current;
        const ctx = canvas?.getContext("2d");
        if (!canvas || !layer || !proj || !ctx) return;
        const dpr = dprRef.current;

        ctx.globalCompositeOperation = "source-over";
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.drawImage(layer, 0, 0);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // cars, three colour buckets (stopped / slow / moving), additive so dense queues glow
        const s = interp.sample(simTime, sample);
        ctx.globalCompositeOperation = "lighter";
        const { ax, ay, scale } = proj;
        const paint = (fill: string, test: (v: number) => boolean) => {
          ctx.fillStyle = fill;
          ctx.beginPath();
          for (let i = 0; i < s.n; i++) {
            if (!test(s.v[i])) continue;
            ctx.rect(ax + s.x[i] * scale - 1, ay - s.y[i] * scale - 1, 2, 2);
          }
          ctx.fill();
        };
        paint("rgba(120, 200, 255, 0.78)", (v) => v >= SLOW_MS);
        paint("rgba(255, 190, 90, 0.85)", (v) => v >= STOPPED_MS && v < SLOW_MS);
        paint("rgba(255, 82, 82, 0.95)", (v) => v < STOPPED_MS);
        ctx.globalCompositeOperation = "source-over";

        // signals
        const k = frameIndexAt(frames, simTime);
        const states = frames.tls[k] ?? "";
        for (let i = 0; i < junctions.length; i++) {
          const p = junctions[i];
          const c = states[i] as keyof typeof SIGNAL;
          if (!p || !SIGNAL[c]) continue;
          const sx = ax + p[0] * scale;
          const sy = ay - p[1] * scale;
          ctx.fillStyle = SIGNAL[c] + "40";
          ctx.beginPath();
          ctx.arc(sx, sy, 5, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = SIGNAL[c];
          ctx.beginPath();
          ctx.arc(sx, sy, 2.2, 0, Math.PI * 2);
          ctx.fill();
        }

        // live numbers (8 Hz is plenty and keeps React out of the animation loop)
        const now = performance.now();
        if (now - statsAt.current > 120) {
          statsAt.current = now;
          if (carsEl.current) carsEl.current.textContent = frames.cars[k].toLocaleString("en-US");
          if (stoppedEl.current) stoppedEl.current.textContent = frames.stopped[k].toLocaleString("en-US");
          if (speedEl.current) speedEl.current.textContent = (frames.meanSpeed[k] * 3.6).toFixed(1);
          if (cursorEl.current) {
            const x = Math.max(0, Math.min(1, (simTime - frames.start) / Math.max(1, frames.end - frames.start))) * 100;
            cursorEl.current.setAttribute("x1", x.toFixed(2));
            cursorEl.current.setAttribute("x2", x.toFixed(2));
          }
        }
      },
    };
    register(id, handle);
    return () => register(id, null);
  }, [id, register, interp, sample, junctions, frames]);

  return (
    <article className="sc-panel" aria-label={`${label}: recorded simulation`}>
      <header className="sc-panel-head">
        <span className="sc-dot" style={{ background: color, boxShadow: `0 0 10px ${color}88` }} aria-hidden="true" />
        <h3>{label}</h3>
        <span className={`sc-chip ${kind}`}>{kind === "model" ? "Final model" : "Baseline"}</span>
      </header>
      <div className="sc-canvas-wrap" ref={wrapRef}>
        <canvas ref={canvasRef} role="img" aria-label={`Map of Ingolstadt with the cars of ${label}; red dots are stopped cars`} />
      </div>
      <dl className="sc-live">
        <div><dt>Cars</dt><dd><span ref={carsEl}>0</span></dd></div>
        <div><dt>Stopped</dt><dd className="warn"><span ref={stoppedEl}>0</span></dd></div>
        <div><dt>Avg km/h</dt><dd><span ref={speedEl}>0.0</span></dd></div>
      </dl>
      <div className="sc-spark" aria-hidden="true">
        <svg viewBox="0 0 100 24" preserveAspectRatio="none">
          <polyline points={sparkPoints} fill="none" stroke={color} strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
          <line ref={cursorEl} x1="0" x2="0" y1="0" y2="24" stroke="rgba(255,255,255,.55)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        </svg>
        <span>stopped cars, 07:20 to 07:30</span>
      </div>
    </article>
  );
});
SimPanel.displayName = "SimPanel";
