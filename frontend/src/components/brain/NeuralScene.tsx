import React, { useEffect, useRef } from "react";
import { evaluate, type Evaluation, type Mlp } from "../../brain/forward";
import { DECISION_S, OBS_SIZE, ToyJunction, type JunctionState } from "../../brain/junction";
import { buildScene, pickActiveUnits, project, type Camera, type Projected, type Scene, type SceneEdge } from "../../brain/scene";

export interface ObserveInfo {
  step: number;
  obs: Float32Array;
  state: JunctionState;
}

export interface DecideInfo {
  step: number;
  probs: number[];
  value: number;
  action: number;
  /** The safety layer refused the "advance" (minimum green not reached). */
  blocked: boolean;
  /** The safety layer ended the phase (maximum green). */
  forced: boolean;
}

interface Props {
  mlp: Mlp;
  paused: boolean;
  /** Playback speed multiplier for the decision clock. */
  speed: number;
  /** Increment to bring the camera back to its starting view. */
  resetKey: number;
  onObserve: (info: ObserveInfo) => void;
  onDecide: (info: DecideInfo) => void;
}

type Rgb = readonly [number, number, number];
const INPUT: Rgb = [196, 218, 255];
const PALETTE = {
  actor: { pos: [34, 211, 238] as Rgb, neg: [168, 85, 247] as Rgb },
  critic: { pos: [251, 191, 36] as Rgb, neg: [251, 113, 133] as Rgb },
};

/** Real seconds one decision takes at 1x: the wave crosses the network, the answer is applied, then the figure rests. */
const CYCLE_S = 3.1;
const WAVE_S = 1.7;
const FRONT_FROM = -0.3;
const FRONT_TO = 3.5;
const DECIDE_AT_FRONT = 3.1;
const PULSES_PER_BLOCK = 26;
const LEVELS = 6;
const MONO = '11px "JetBrains Mono", ui-monospace, Menlo, monospace';

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
const smooth = (x: number) => {
  const t = clamp01(x);
  return t * t * (3 - 2 * t);
};
const rgba = (c: Rgb, a: number) => `rgba(${c[0]},${c[1]},${c[2]},${a.toFixed(3)})`;

const sprites = new Map<string, HTMLCanvasElement>();
/** A soft round glow of one colour, drawn once and stamped for every neuron and photon. */
function glow(c: Rgb): HTMLCanvasElement {
  const key = c.join(",");
  let s = sprites.get(key);
  if (!s) {
    s = document.createElement("canvas");
    s.width = s.height = 64;
    const g = s.getContext("2d");
    if (g) {
      const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
      grad.addColorStop(0, rgba(c, 1));
      grad.addColorStop(0.25, rgba(c, 0.55));
      grad.addColorStop(0.6, rgba(c, 0.12));
      grad.addColorStop(1, rgba(c, 0));
      g.fillStyle = grad;
      g.fillRect(0, 0, 64, 64);
    }
    sprites.set(key, s);
  }
  return s;
}

interface Pulse {
  edge: SceneEdge;
  /** How late (in front units) this photon leaves its neuron, so the wave looks like many separate signals. */
  delay: number;
}

interface Ambient {
  edge: SceneEdge;
  t: number;
  rate: number;
}

function nodeColor(branch: "input" | "actor" | "critic", a: number): Rgb {
  if (branch === "input") return INPUT;
  return a >= 0 ? PALETTE[branch].pos : PALETTE[branch].neg;
}

export const NeuralScene: React.FC<Props> = ({ mlp, paused, speed, resetKey, onObserve, onDecide }) => {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const live = useRef({ paused, speed, onObserve, onDecide });
  const user = useRef({ yaw: 0, pitch: 0 });

  useEffect(() => {
    live.current = { paused, speed, onObserve, onDecide };
  }, [paused, speed, onObserve, onDecide]);
  useEffect(() => {
    user.current = { yaw: 0, pitch: 0 };
  }, [resetKey]);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!wrap || !canvas || !ctx) return;
    const reduce = !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    const { picks, valueRange } = pickActiveUnits(mlp);
    const scene: Scene = buildScene(mlp, picks, valueRange);
    const { nodes, edges } = scene;
    const N = nodes.length;
    const junction = new ToyJunction(7);
    const obs = new Float32Array(OBS_SIZE);
    const prev = new Float32Array(N);
    const cur = new Float32Array(N);
    const shown = new Float32Array(N);
    const proj: Projected[] = nodes.map(() => ({ sx: 0, sy: 0, depth: 1, scale: 1 }));
    const order = Array.from({ length: N }, (_, i) => i);
    const strong = [...edges].sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 220);
    const buckets: number[][] = Array.from({ length: 2 * 2 * LEVELS }, () => []);
    const actorOut = nodes.filter((n) => n.layer === 3 && n.branch === "actor");
    const criticOut = nodes.find((n) => n.layer === 3 && n.branch === "critic");
    const dust = Array.from({ length: 70 }, (_, i) => ({
      x: ((i * 0.618034) % 1) * 17 - 8.5, y: ((i * 0.381966 + 0.13) % 1) * 9 - 4.5, z: ((i * 0.754877 + 0.4) % 1) * 9 - 4.5, s: 0.4 + ((i * 0.31) % 1),
    }));

    let width = 0;
    let height = 0;
    let dpr = 1;
    let backdrop: CanvasGradient | null = null;
    const resize = () => {
      width = Math.max(240, wrap.clientWidth);
      height = Math.max(200, wrap.clientHeight);
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      backdrop = ctx.createRadialGradient(width * 0.5, height * 0.5, 0, width * 0.5, height * 0.5, Math.max(width, height) * 0.7);
      backdrop.addColorStop(0, "#0d1626");
      backdrop.addColorStop(0.55, "#070c16");
      backdrop.addColorStop(1, "#04070d");
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    // ── the decision loop: observe -> wave through both networks -> decide -> rest ──────────────────────────────────
    let step = 0;
    let tickT = 0;
    let decided = true;
    let front = FRONT_FROM;
    let evaluation: Evaluation | null = null;
    const values = { prev: 0, cur: 0 };
    let pulses: Pulse[] = [];
    let decisionPulse: { node: number; t: number } | null = null;
    const ambient: Ambient[] = [];
    let ambientTimer = 0;
    let rng = 12345;
    const rand = () => ((rng = (rng * 1664525 + 1013904223) >>> 0) / 4294967296);

    const fillActivations = (e: Evaluation, out: Float32Array) => {
      const [lo, hi] = scene.valueRange;
      for (const n of nodes) {
        if (n.layer === 0) out[n.id] = obs[n.unit];
        else if (n.layer === 3) out[n.id] = n.branch === "actor" ? e.probs[n.unit] : clamp01((e.value - lo) / (hi - lo));
        else {
          const trace = n.branch === "actor" ? e.actor : e.critic;
          out[n.id] = (n.layer === 1 ? trace.h1 : trace.h2)[n.unit];
        }
      }
    };

    const beginTick = () => {
      junction.observe(obs);
      evaluation = evaluate(mlp, obs);
      values.prev = step === 0 ? evaluation.value : values.cur;
      values.cur = evaluation.value;
      prev.set(cur);
      fillActivations(evaluation, cur);
      if (step === 0) prev.set(cur);
      step += 1;
      tickT = 0;
      decided = false;
      front = FRONT_FROM;
      // The photons that travel with the wave: the strongest real contributions of each block this tick.
      const chosen: Pulse[] = [];
      for (const block of [0, 1, 2] as const) {
        const ranked = edges
          .filter((e) => e.block === block)
          .map((e) => ({ e, c: Math.abs(e.w * cur[e.from]) }))
          .sort((a, b) => b.c - a.c)
          .slice(0, PULSES_PER_BLOCK);
        for (const { e } of ranked) chosen.push({ edge: e, delay: rand() * 0.3 });
      }
      pulses = chosen;
      live.current.onObserve({ step, obs: Float32Array.from(obs), state: { ...junction.state, queues: [...junction.state.queues], upstream: [...junction.state.upstream], downstream: [...junction.state.downstream] } });
    };

    const decide = () => {
      if (!evaluation) return;
      decided = true;
      const e = evaluation;
      junction.step(e.action === 1);
      const target = actorOut[e.action];
      if (target) decisionPulse = { node: target.id, t: 0 };
      live.current.onDecide({ step, probs: e.probs, value: e.value, action: e.action, blocked: junction.state.blocked, forced: junction.state.forced });
    };

    // ── drawing ──────────────────────────────────────────────────────────────────────────────────────────────────────
    const gridLines: [number, number, number, number][] = [];
    for (let x = -8; x <= 8; x += 2) gridLines.push([x, -3.9, -4.5, x], [x, -3.9, 4.5, x]);
    const lit = (layer: number) => smooth((front - (layer - 0.35)) / 0.7);

    const draw = (time: number, camera: Camera) => {
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
      ctx.fillStyle = backdrop ?? "#070c16";
      ctx.fillRect(0, 0, width, height);

      // floor grid
      const a = { sx: 0, sy: 0, depth: 1, scale: 1 };
      const b = { sx: 0, sy: 0, depth: 1, scale: 1 };
      ctx.strokeStyle = "rgba(120,160,220,0.07)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < gridLines.length; i += 2) {
        const [x, y, z0] = gridLines[i];
        const z1 = gridLines[i + 1][2];
        project({ x, y, z: z0 }, camera, width, height, a);
        project({ x, y, z: z1 }, camera, width, height, b);
        ctx.moveTo(a.sx, a.sy);
        ctx.lineTo(b.sx, b.sy);
      }
      for (let z = -4.5; z <= 4.5; z += 1.5) {
        project({ x: -8, y: -3.9, z }, camera, width, height, a);
        project({ x: 8, y: -3.9, z }, camera, width, height, b);
        ctx.moveTo(a.sx, a.sy);
        ctx.lineTo(b.sx, b.sy);
      }
      ctx.stroke();

      // drifting dust
      ctx.globalCompositeOperation = "lighter";
      for (const d of dust) {
        const x = ((d.x + time * 0.12 * d.s + 8.5) % 17) - 8.5;
        project({ x, y: d.y + Math.sin(time * 0.3 * d.s + d.x) * 0.15, z: d.z }, camera, width, height, a);
        ctx.fillStyle = `rgba(150,190,255,${(0.05 + 0.08 * d.s) * Math.max(0.3, 1.4 - a.depth)})`;
        ctx.fillRect(a.sx, a.sy, 1.4, 1.4);
      }

      // neurons: breathe and drift a little, glow with their (wave-delayed) activation
      for (const n of nodes) {
        const k = reduce ? 0 : 1;
        const p = project(
          { x: n.pos.x + k * 0.05 * Math.sin(time * 0.55 + n.seed * 1.7), y: n.pos.y + k * 0.05 * Math.sin(time * 0.47 + n.seed * 2.3), z: n.pos.z + k * 0.05 * Math.sin(time * 0.61 + n.seed) },
          camera, width, height, proj[n.id],
        );
        proj[n.id] = p;
        const l = lit(n.layer);
        shown[n.id] = prev[n.id] + (cur[n.id] - prev[n.id]) * l;
      }

      // connections: brightness = this tick's real contribution (weight x source activation); colour = its sign
      for (const bucket of buckets) bucket.length = 0;
      for (let i = 0; i < edges.length; i++) {
        const e = edges[i];
        const c = e.w * shown[e.from];
        const strength = Math.pow(Math.min(1, Math.abs(c) / (scene.weightScale[e.block] * 0.9)), 1.15);
        const depth = (proj[e.from].depth + proj[e.to].depth) / 2;
        const alpha = (0.03 + 0.78 * strength) * Math.max(0.3, Math.min(1, 1.6 - 0.65 * depth)) * (0.35 + 0.65 * lit(e.block + 1));
        const level = Math.min(LEVELS - 1, Math.floor(alpha / 0.13));
        buckets[((e.branch === "actor" ? 0 : 1) * 2 + (c >= 0 ? 0 : 1)) * LEVELS + level].push(i);
      }
      for (let bi = 0; bi < buckets.length; bi++) {
        const list = buckets[bi];
        if (!list.length) continue;
        const level = bi % LEVELS;
        const negative = Math.floor(bi / LEVELS) % 2 === 1;
        const branch = bi >= 2 * LEVELS ? "critic" : "actor";
        ctx.strokeStyle = rgba(negative ? PALETTE[branch].neg : PALETTE[branch].pos, (level + 0.6) * 0.13);
        ctx.lineWidth = 0.6 + level * 0.32;
        ctx.beginPath();
        for (const i of list) {
          const e = edges[i];
          ctx.moveTo(proj[e.from].sx, proj[e.from].sy);
          ctx.lineTo(proj[e.to].sx, proj[e.to].sy);
        }
        ctx.stroke();
      }

      // photons riding the wave along the strongest connections
      if (!reduce && !decided) {
        for (const pu of pulses) {
          const head = (front - pu.edge.block - pu.delay) / 0.75;
          if (head <= 0 || head >= 1) continue;
          const from = proj[pu.edge.from];
          const to = proj[pu.edge.to];
          const col = PALETTE[pu.edge.branch].pos;
          const x = from.sx + (to.sx - from.sx) * head;
          const y = from.sy + (to.sy - from.sy) * head;
          const t0 = Math.max(0, head - 0.14);
          ctx.strokeStyle = rgba(col, 0.55 * Math.sin(head * Math.PI));
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(from.sx + (to.sx - from.sx) * t0, from.sy + (to.sy - from.sy) * t0);
          ctx.lineTo(x, y);
          ctx.stroke();
          const g = 9 * Math.max(0.6, 1.2 - from.depth * 0.4);
          ctx.globalAlpha = Math.sin(head * Math.PI);
          ctx.drawImage(glow(col), x - g, y - g, g * 2, g * 2);
          ctx.globalAlpha = 1;
        }
      }
      // ambient life between decisions: faint signals wandering over strong connections
      for (const am of ambient) {
        const from = proj[am.edge.from];
        const to = proj[am.edge.to];
        const col = PALETTE[am.edge.branch].pos;
        const x = from.sx + (to.sx - from.sx) * am.t;
        const y = from.sy + (to.sy - from.sy) * am.t;
        ctx.globalAlpha = 0.5 * Math.sin(am.t * Math.PI);
        ctx.drawImage(glow(col), x - 6, y - 6, 12, 12);
        ctx.globalAlpha = 1;
      }

      // neurons, far to near
      order.sort((i, j) => proj[j].depth - proj[i].depth);
      for (const i of order) {
        const n = nodes[i];
        const p = proj[i];
        const a2 = shown[i];
        const mag = n.layer === 0 || n.layer === 3 ? clamp01(a2) : Math.min(1, Math.abs(a2));
        const col = n.layer === 3 ? PALETTE[n.branch === "critic" ? "critic" : "actor"].pos : nodeColor(n.branch, a2);
        const fade = Math.max(0.35, Math.min(1, 1.55 - 0.6 * p.depth));
        const breathe = reduce ? 1 : 1 + 0.08 * Math.sin(time * 1.7 + n.seed);
        const big = n.layer === 3 ? 1.9 : 1;
        const r = (0.075 + 0.075 * mag) * big * p.scale * breathe;
        const g = r * 4.4;
        ctx.globalAlpha = (0.22 + 0.7 * mag) * fade;
        ctx.drawImage(glow(col), p.sx - g, p.sy - g, g * 2, g * 2);
        ctx.globalAlpha = 1;
        ctx.fillStyle = rgba(col, (0.4 + 0.6 * mag) * fade);
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, r, 0, Math.PI * 2);
        ctx.fill();
        if (n.layer === 3) {
          ctx.strokeStyle = rgba(col, 0.5 * fade);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, r * 1.55, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // the decision: a ring leaves the chosen output
      if (decisionPulse) {
        const p = proj[decisionPulse.node];
        const t = decisionPulse.t;
        ctx.strokeStyle = rgba(PALETTE.actor.pos, 0.9 * (1 - t));
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, 12 + t * 70, 0, Math.PI * 2);
        ctx.stroke();
      }

      // labels
      ctx.globalCompositeOperation = "source-over";
      ctx.font = MONO;
      ctx.textBaseline = "middle";
      const label = (text: string, x: number, y: number, depth: number, alpha: number, align: CanvasTextAlign = "center") => {
        ctx.textAlign = align;
        ctx.fillStyle = `rgba(176,196,228,${(alpha * Math.max(0.45, 1.5 - 0.55 * depth)).toFixed(3)})`;
        ctx.fillText(text, x, y);
      };
      const caption = (text: string, layer: number, branch: "input" | "actor" | "critic", yOffset: number) => {
        const sample = nodes.find((n) => n.layer === layer && n.branch === branch);
        if (!sample) return;
        const top = project({ x: sample.pos.x, y: (branch === "critic" ? -1.95 : branch === "actor" ? 1.95 : 0) + yOffset, z: 0 }, camera, width, height, a);
        label(text, top.sx, top.sy, top.depth, 0.85);
      };
      if (width >= 560) {
        caption(`${OBS_SIZE} observation numbers`, 0, "input", 2.35);
        for (const br of ["actor", "critic"] as const) {
          const off = br === "actor" ? 1.78 : -1.78;
          caption("hidden 1 · 256", 1, br, off);
          caption("hidden 2 · 256", 2, br, off);
        }
      }
      for (const n of actorOut) {
        const p = proj[n.id];
        const text = `${n.unit === 0 ? "keep" : "advance"} ${(clamp01(shown[n.id]) * 100).toFixed(0)}%`;
        label(text, p.sx + 0.34 * p.scale + 8, p.sy, p.depth, 0.95, "left");
      }
      if (criticOut) {
        const p = proj[criticOut.id];
        const v = lit(3) > 0.5 ? values.cur : values.prev;
        label(`value ${v.toFixed(0)}`, p.sx + 0.34 * p.scale + 8, p.sy, p.depth, 0.95, "left");
      }
    };

    // ── the clock ────────────────────────────────────────────────────────────────────────────────────────────────────
    let raf = 0;
    let last = performance.now();
    let time = 0;
    const perf = { frames: 0, drawMs: 0 };
    (window as unknown as { __brainPerf?: typeof perf }).__brainPerf = perf;
    beginTick();

    const frame = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      time += dt;
      const state = live.current;
      if (!state.paused) {
        tickT += dt * state.speed;
        const waveS = reduce ? 0.05 : WAVE_S;
        front = FRONT_FROM + (FRONT_TO - FRONT_FROM) * Math.min(1, tickT / waveS);
        if (!decided && front >= DECIDE_AT_FRONT) decide();
        if (tickT >= (reduce ? 2.2 : CYCLE_S)) beginTick();
        if (decisionPulse) {
          decisionPulse.t += (dt * state.speed) / 0.9;
          if (decisionPulse.t >= 1) decisionPulse = null;
        }
        if (!reduce) {
          ambientTimer -= dt * state.speed;
          if (ambientTimer <= 0 && ambient.length < 26 && strong.length) {
            ambientTimer = 0.09;
            ambient.push({ edge: strong[Math.floor(rand() * strong.length)], t: 0, rate: 0.55 + rand() * 0.6 });
          }
          for (let i = ambient.length - 1; i >= 0; i--) {
            ambient[i].t += dt * state.speed * ambient[i].rate;
            if (ambient[i].t >= 1) ambient.splice(i, 1);
          }
        }
      }
      const camera: Camera = {
        yaw: -0.5 + user.current.yaw + (reduce ? 0 : Math.sin(time * 0.27) * 0.3),
        pitch: 0.3 + user.current.pitch + (reduce ? 0 : Math.sin(time * 0.21) * 0.05),
        dist: 15,
      };
      const t0 = performance.now();
      draw(time, camera);
      perf.drawMs += performance.now() - t0;
      perf.frames += 1;
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    // drag to orbit
    let drag: { x: number; y: number } | null = null;
    const down = (e: PointerEvent) => {
      drag = { x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId);
      wrap.classList.add("dragging");
    };
    const move = (e: PointerEvent) => {
      if (!drag) return;
      user.current.yaw = Math.max(-1.3, Math.min(1.3, user.current.yaw + (e.clientX - drag.x) * 0.006));
      user.current.pitch = Math.max(-0.25, Math.min(0.85, user.current.pitch + (e.clientY - drag.y) * 0.004));
      drag = { x: e.clientX, y: e.clientY };
    };
    const up = () => {
      drag = null;
      wrap.classList.remove("dragging");
    };
    const reset = () => {
      user.current = { yaw: 0, pitch: 0 };
    };
    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", up);
    canvas.addEventListener("dblclick", reset);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", up);
      canvas.removeEventListener("dblclick", reset);
    };
  }, [mlp]);

  return (
    <div className="br-stage" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label="3D figure of the two neural networks: the actor above chooses keep or advance, the critic below scores the situation. Drag to rotate."
      />
      <span className="br-hint">drag to rotate · double-click to reset · one decision every {DECISION_S} s of traffic</span>
    </div>
  );
};
