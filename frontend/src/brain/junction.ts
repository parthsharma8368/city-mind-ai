/**
 * A small made-up junction that produces the 22 numbers the policy reads, so the figure has something to react to. It follows
 * the observation layout of src/environment/local_observation.py (one-hot active phase, yellow flag, time in phase against its
 * own minimum / maximum, per-phase demand, worst queue, cars waiting outside, phase count, two upstream and two downstream
 * queues) and the safety layer's rules (a phase is kept at least `MIN_GREEN`, forced off after `MAX_GREEN`). The weights that read
 * these numbers are the real trained ones; only the traffic is invented.
 */
export const OBS_SIZE = 22;
export const PHASES = 3;
export const DECISION_S = 5;
export const MIN_GREEN = 10;
export const MAX_GREEN = 45;
const SERVICE_PER_STEP = 5;
const QUEUE_CAP = 60;
const DEMAND_K = 8;
const PENDING_K = 20;

/** Small deterministic generator, so a given seed always plays the same story. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface JunctionState {
  /** Halted cars waiting for each green phase. */
  queues: number[];
  phase: number;
  /** Seconds the current phase has been green. */
  elapsed: number;
  /** A yellow / all-red clearance is running. */
  transition: boolean;
  /** Cars waiting outside the network to enter at this junction. */
  pending: number;
  upstream: number[];
  downstream: number[];
  /** The last "advance" was refused because the phase had not reached its minimum green. */
  blocked: boolean;
  /** The phase was ended by the safety layer's maximum green. */
  forced: boolean;
  /** Seconds of simulated time since the start. */
  clock: number;
}

export class ToyJunction {
  private rand: () => number;
  state: JunctionState;

  constructor(seed = 7) {
    this.rand = mulberry32(seed);
    this.state = {
      queues: [11, 3, 2], phase: 0, elapsed: 0, transition: false, pending: 3,
      upstream: [0.18, 0.1], downstream: [0.1, 0.05], blocked: false, forced: false, clock: 0,
    };
  }

  /** Writes the 22-number observation into `out` (allocating when none is given). */
  observe(out: Float32Array = new Float32Array(OBS_SIZE)): Float32Array {
    const s = this.state;
    out.fill(0);
    if (!s.transition) out[s.phase] = 1;
    out[6] = s.transition ? 1 : 0;
    out[7] = Math.min(s.elapsed / MIN_GREEN, 3) / 3;
    out[8] = Math.min(s.elapsed / MAX_GREEN, 1.5) / 1.5;
    for (let i = 0; i < PHASES; i++) out[9 + i] = s.queues[i] / (Math.abs(s.queues[i]) + DEMAND_K);
    const worst = Math.max(...s.queues);
    out[15] = worst / (worst + DEMAND_K);
    out[16] = s.pending / (s.pending + PENDING_K);
    out[17] = PHASES / 6;
    out[18] = s.upstream[0];
    out[19] = s.upstream[1];
    out[20] = s.downstream[0];
    out[21] = s.downstream[1];
    return out;
  }

  /** One 5 s decision: `advance` = move to the next green, otherwise keep this one. */
  step(advance: boolean): void {
    const s = this.state;
    const r = this.rand;
    s.blocked = false;
    s.forced = false;
    if (s.transition) {
      s.transition = false;
      s.phase = (s.phase + 1) % PHASES;
      s.elapsed = 0;
    } else {
      let end = advance;
      if (end && s.elapsed < MIN_GREEN) {
        end = false;
        s.blocked = true;
      }
      if (!end && s.elapsed + DECISION_S > MAX_GREEN) {
        end = true;
        s.forced = true;
      }
      if (end) s.transition = true;
      else s.elapsed += DECISION_S;
    }
    // Cars arrive in waves (a slow rush that peaks on a different approach each time) and the green phase discharges its queue.
    const t = s.clock / 60;
    for (let i = 0; i < PHASES; i++) {
      const rate = 1.1 * (0.65 + 0.6 * (0.5 + 0.5 * Math.sin(t + i * 2.1)));
      const arrivals = Math.floor(rate * 2 * r() + r());
      const served = !s.transition && i === s.phase ? SERVICE_PER_STEP : 0;
      s.queues[i] = Math.max(0, Math.min(QUEUE_CAP, s.queues[i] + arrivals - served));
    }
    s.pending = Math.max(0, Math.min(14, s.pending + (r() - 0.48) * 2));
    for (const arr of [s.upstream, s.downstream]) {
      for (let i = 0; i < arr.length; i++) arr[i] = Math.max(0, Math.min(0.7, arr[i] + (r() - 0.5) * 0.08));
    }
    s.clock += DECISION_S;
  }
}
