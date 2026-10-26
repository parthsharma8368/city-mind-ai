/**
 * The 3D figure's geometry, kept free of canvas and React: which neurons are drawn, where they sit in space, which
 * connections link them (with their real trained weights), and the perspective projection used to put them on screen.
 *
 * Layers run left to right: 0 observation (22 numbers, shared), 1 and 2 the two hidden layers, 3 the output. The actor
 * (above) and the critic (below) are separate networks that read the same observation, so layers 1-3 exist once per branch.
 * Only the most active neurons of each 256-wide hidden layer are drawn (`SHOWN`), chosen by how much they vary on a warm-up run.
 */
import { evaluate, type Dense, type Mlp } from "./forward";
import { OBS_SIZE, ToyJunction } from "./junction";

export const SHOWN = 16;
export type Branch = "input" | "actor" | "critic";

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface SceneNode {
  id: number;
  /** 0 observation, 1 hidden 1, 2 hidden 2, 3 output. */
  layer: 0 | 1 | 2 | 3;
  branch: Branch;
  /** Index inside the full layer (observation slot, hidden unit, or output index). */
  unit: number;
  pos: Vec3;
  /** Random offset so neurons breathe out of step with each other. */
  seed: number;
}

export interface SceneEdge {
  from: number;
  to: number;
  /** Real weight of this connection. */
  w: number;
  branch: "actor" | "critic";
  /** The edge links layer `block` to layer `block + 1`. */
  block: 0 | 1 | 2;
}

export interface Scene {
  nodes: SceneNode[];
  edges: SceneEdge[];
  /** Per block, the 90th percentile of |w| among its edges (the scale for line brightness). */
  weightScale: number[];
  /** Typical range of the critic's value on the warm-up run, to map it onto 0..1. */
  valueRange: [number, number];
}

export interface Picks {
  actor: { h1: number[]; h2: number[] };
  critic: { h1: number[]; h2: number[] };
}

export const LAYER_X = [-5.3, -1.9, 1.5, 4.7] as const;
export const BRANCH_Y = { actor: 1.95, critic: -1.95 } as const;
const INPUT_RADIUS = 1.7;
const HIDDEN_RADIUS = 1.1;
/** Share of the strongest connections of each block that are drawn (the rest would be a grey wall). */
const KEEP_SHARE = 0.55;

function topByVariance(variance: Float64Array, count: number): number[] {
  return Array.from(variance.keys())
    .sort((a, b) => variance[b] - variance[a])
    .slice(0, count)
    .sort((a, b) => a - b);
}

/** Runs the toy junction under the policy for `steps` decisions and picks the units that vary most (and the value range). */
export function pickActiveUnits(mlp: Mlp, steps = 400, count = SHOWN): { picks: Picks; valueRange: [number, number] } {
  const junction = new ToyJunction(99);
  const obs = new Float32Array(OBS_SIZE);
  const n1 = mlp.actor.l1.rows;
  const n2 = mlp.actor.l2.rows;
  const stats = {
    a1: { m: new Float64Array(n1), s: new Float64Array(n1) },
    a2: { m: new Float64Array(n2), s: new Float64Array(n2) },
    c1: { m: new Float64Array(mlp.critic.l1.rows), s: new Float64Array(mlp.critic.l1.rows) },
    c2: { m: new Float64Array(mlp.critic.l2.rows), s: new Float64Array(mlp.critic.l2.rows) },
  };
  const welford = (acc: { m: Float64Array; s: Float64Array }, x: Float32Array, k: number) => {
    for (let i = 0; i < x.length; i++) {
      const d = x[i] - acc.m[i];
      acc.m[i] += d / k;
      acc.s[i] += d * (x[i] - acc.m[i]);
    }
  };
  const values: number[] = [];
  for (let k = 1; k <= steps; k++) {
    junction.observe(obs);
    const e = evaluate(mlp, obs);
    welford(stats.a1, e.actor.h1, k);
    welford(stats.a2, e.actor.h2, k);
    welford(stats.c1, e.critic.h1, k);
    welford(stats.c2, e.critic.h2, k);
    values.push(e.value);
    junction.step(e.action === 1);
  }
  values.sort((a, b) => a - b);
  const lo = values[Math.floor(values.length * 0.03)];
  const hi = values[Math.ceil(values.length * 0.97) - 1];
  return {
    picks: {
      actor: { h1: topByVariance(stats.a1.s, count), h2: topByVariance(stats.a2.s, count) },
      critic: { h1: topByVariance(stats.c1.s, count), h2: topByVariance(stats.c2.s, count) },
    },
    valueRange: hi > lo ? [lo, hi] : [lo - 1, lo + 1],
  };
}

function ringPosition(i: number, n: number, radius: number, cx: number, cy: number, twist: number): Vec3 {
  const a = (i / n) * Math.PI * 2 - Math.PI / 2 + twist;
  return { x: cx, y: cy + Math.cos(a) * radius * -1, z: Math.sin(a) * radius };
}

function percentile(values: number[], q: number): number {
  if (!values.length) return 1;
  const s = [...values].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))] || 1;
}

export function buildScene(mlp: Mlp, picks: Picks, valueRange: [number, number]): Scene {
  const nodes: SceneNode[] = [];
  let seedCounter = 0;
  const add = (layer: SceneNode["layer"], branch: Branch, unit: number, pos: Vec3): number => {
    const id = nodes.length;
    // A fixed pseudo-random offset per neuron (golden-ratio sequence), no Math.random so the figure is reproducible.
    seedCounter += 0.6180339887;
    nodes.push({ id, layer, branch, unit, pos, seed: (seedCounter % 1) * Math.PI * 2 });
    return id;
  };

  const inputIds: number[] = [];
  for (let i = 0; i < OBS_SIZE; i++) inputIds.push(add(0, "input", i, ringPosition(i, OBS_SIZE, INPUT_RADIUS, LAYER_X[0], 0, 0)));

  const ids = { actor: { h1: [] as number[], h2: [] as number[], out: [] as number[] }, critic: { h1: [] as number[], h2: [] as number[], out: [] as number[] } };
  for (const branch of ["actor", "critic"] as const) {
    const cy = BRANCH_Y[branch];
    picks[branch].h1.forEach((u, i) => ids[branch].h1.push(add(1, branch, u, ringPosition(i, picks[branch].h1.length, HIDDEN_RADIUS, LAYER_X[1], cy, 0.2))));
    picks[branch].h2.forEach((u, i) => ids[branch].h2.push(add(2, branch, u, ringPosition(i, picks[branch].h2.length, HIDDEN_RADIUS, LAYER_X[2], cy, 0.55))));
  }
  const outputs = mlp.actor.out.rows;
  for (let o = 0; o < outputs; o++) ids.actor.out.push(add(3, "actor", o, { x: LAYER_X[3], y: BRANCH_Y.actor + (o === 0 ? 0.62 : -0.62), z: 0 }));
  ids.critic.out.push(add(3, "critic", 0, { x: LAYER_X[3], y: BRANCH_Y.critic, z: 0 }));

  const edges: SceneEdge[] = [];
  const weightScale: number[] = [];
  const block = (b: 0 | 1 | 2, branch: "actor" | "critic", srcIds: number[], dstIds: number[], layer: Dense) => {
    const list: SceneEdge[] = [];
    for (const to of dstIds) {
      for (const from of srcIds) {
        const w = layer.w[nodes[to].unit * layer.cols + nodes[from].unit];
        list.push({ from, to, w, branch, block: b });
      }
    }
    const cut = percentile(list.map((e) => Math.abs(e.w)), 1 - KEEP_SHARE);
    const kept = list.filter((e) => Math.abs(e.w) >= cut);
    edges.push(...kept);
    return kept.map((e) => Math.abs(e.w));
  };
  const perBlock: number[][] = [[], [], []];
  for (const branch of ["actor", "critic"] as const) {
    const net = mlp[branch];
    perBlock[0].push(...block(0, branch, inputIds, ids[branch].h1, net.l1));
    perBlock[1].push(...block(1, branch, ids[branch].h1, ids[branch].h2, net.l2));
    perBlock[2].push(...block(2, branch, ids[branch].h2, ids[branch].out, net.out));
  }
  for (const b of [0, 1, 2]) weightScale.push(percentile(perBlock[b], 0.9));
  return { nodes, edges, weightScale, valueRange };
}

// ── camera ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

export interface Camera {
  /** Radians around the vertical axis. */
  yaw: number;
  /** Radians of look-down. */
  pitch: number;
  /** Distance of the camera from the centre of the figure, in world units. */
  dist: number;
}

export interface Projected {
  sx: number;
  sy: number;
  /** Larger means farther from the camera; 1 at the centre of the figure. */
  depth: number;
  /** Pixels per world unit at this point (perspective scaling). */
  scale: number;
}

/** Pixels per world unit at the centre of the figure so the whole scene (about 16 x 8.4 units) fits the canvas (a phone gets a tighter fit). */
export function fitScale(width: number, height: number): number {
  return Math.min(width / (width < 560 ? 12.6 : 16.2), height / 8.6);
}

export function project(p: Vec3, cam: Camera, width: number, height: number, out: Projected = { sx: 0, sy: 0, depth: 1, scale: 1 }): Projected {
  const cy = Math.cos(cam.yaw);
  const sy = Math.sin(cam.yaw);
  const x1 = p.x * cy + p.z * sy;
  const z1 = -p.x * sy + p.z * cy;
  const cp = Math.cos(cam.pitch);
  const sp = Math.sin(cam.pitch);
  const y2 = p.y * cp - z1 * sp;
  const z2 = p.y * sp + z1 * cp;
  const depth = (cam.dist + z2) / cam.dist;
  const scale = fitScale(width, height) / Math.max(0.2, depth);
  out.sx = width / 2 + x1 * scale;
  out.sy = height / 2 - y2 * scale;
  out.depth = depth;
  out.scale = scale;
  return out;
}
