/**
 * The trained networks, run in the browser: the same tanh multilayer perceptrons the policy uses (SB3's MlpPolicy keeps the
 * actor and the critic as two separate 22 -> 256 -> 256 -> out stacks). Pure functions, no React / canvas.
 */
import type { PolicyDTO, PolicyDenseDTO, PolicyNetDTO } from "../api/client";

export interface Dense {
  rows: number;
  cols: number;
  /** Row-major: weight from input c to output r is w[r * cols + c]. */
  w: Float32Array;
  b: Float32Array;
}

export interface Net {
  l1: Dense;
  l2: Dense;
  out: Dense;
}

export interface Mlp {
  label: string;
  obsNames: string[];
  actions: string[];
  actor: Net;
  critic: Net;
}

function toDense(d: PolicyDenseDTO, where: string): Dense {
  const [rows, cols] = d.shape;
  if (d.w.length !== rows * cols || d.b.length !== rows) throw new Error(`${where}: weights do not match shape ${rows}x${cols}`);
  return { rows, cols, w: Float32Array.from(d.w), b: Float32Array.from(d.b) };
}

function toNet(n: PolicyNetDTO, name: string): Net {
  const net = { l1: toDense(n.l1, `${name}.l1`), l2: toDense(n.l2, `${name}.l2`), out: toDense(n.out, `${name}.out`) };
  if (net.l2.cols !== net.l1.rows || net.out.cols !== net.l2.rows) throw new Error(`${name}: layer sizes do not chain`);
  return net;
}

export function buildMlp(dto: PolicyDTO): Mlp {
  const actor = toNet(dto.actor, "actor");
  const critic = toNet(dto.critic, "critic");
  if (actor.l1.cols !== critic.l1.cols || actor.l1.cols !== dto.obs_names.length) throw new Error("actor and critic must read the same observation");
  return { label: dto.label, obsNames: dto.obs_names, actions: dto.actions, actor, critic };
}

/** Number of weights and biases of one net, or of the whole model. */
export function parameterCount(n: Net): number {
  return [n.l1, n.l2, n.out].reduce((s, d) => s + d.w.length + d.b.length, 0);
}

/** out = act(W x + b); `tanh` for the hidden layers, identity for the output layer. */
export function denseForward(d: Dense, x: ArrayLike<number>, out: Float32Array, activate: boolean): Float32Array {
  for (let r = 0; r < d.rows; r++) {
    let s = d.b[r];
    const base = r * d.cols;
    for (let c = 0; c < d.cols; c++) s += d.w[base + c] * x[c];
    out[r] = activate ? Math.tanh(s) : s;
  }
  return out;
}

export interface NetTrace {
  h1: Float32Array;
  h2: Float32Array;
  out: Float32Array;
}

export function runNet(net: Net, obs: ArrayLike<number>): NetTrace {
  const h1 = denseForward(net.l1, obs, new Float32Array(net.l1.rows), true);
  const h2 = denseForward(net.l2, h1, new Float32Array(net.l2.rows), true);
  const out = denseForward(net.out, h2, new Float32Array(net.out.rows), false);
  return { h1, h2, out };
}

/** Softmax of the actor's two scores: [P(keep), P(advance)]. */
export function softmax(logits: ArrayLike<number>): number[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const e = Array.from({ length: logits.length }, (_, i) => Math.exp(logits[i] - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

export interface Evaluation {
  actor: NetTrace;
  critic: NetTrace;
  /** P(keep), P(advance). */
  probs: number[];
  /** The critic's estimate of the future reward of this situation. */
  value: number;
  /** What the deterministic policy does (evaluation uses the most probable action). */
  action: number;
}

export function evaluate(mlp: Mlp, obs: ArrayLike<number>): Evaluation {
  const actor = runNet(mlp.actor, obs);
  const critic = runNet(mlp.critic, obs);
  const probs = softmax(actor.out);
  return { actor, critic, probs, value: critic.out[0], action: probs[1] > probs[0] ? 1 : 0 };
}
