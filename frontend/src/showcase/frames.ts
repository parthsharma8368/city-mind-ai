/**
 * Playback engine for the showcase: turns the compact frames of the API into typed arrays and answers "where is every
 * car at time t" with linear interpolation between the recorded 5 s frames, so 10x playback is smooth at the display's
 * frame rate. Pure functions and one small class: no React, no canvas (unit tested in frames.test.ts).
 */
import type { ShowcaseFramesDTO } from "../api/client";

/** A car slower than this (m/s) counts as stopped, i.e. part of a queue. */
export const STOPPED_MS = 1;
/** Cars that moved further than this (m) between two frames (re-routed / re-inserted) are not interpolated. */
export const MAX_INTERPOLATION_M = 140;

export interface FrameSet {
  controller: string;
  start: number;
  end: number;
  step: number;
  /** Frame k holds cars [offsets[k], offsets[k + 1]) of the flat arrays. */
  offsets: Int32Array;
  times: Float64Array;
  vid: Int32Array;
  /** Metres relative to `origin` (add the origin back for map coordinates). */
  x: Float32Array;
  y: Float32Array;
  v: Float32Array;
  origin: [number, number];
  vehicleCount: number;
  tlsIds: string[];
  /** One g/y/r/n character per junction, per frame. */
  tls: string[];
  /** Per frame: cars in the network, stopped cars, mean speed (m/s). */
  cars: Int32Array;
  stopped: Int32Array;
  meanSpeed: Float32Array;
  maxCars: number;
}

export function buildFrameSet(dto: ShowcaseFramesDTO): FrameSet {
  const n = dto.times.length;
  const offsets = new Int32Array(n + 1);
  for (let k = 0; k < n; k++) offsets[k + 1] = offsets[k] + (dto.counts[k] ?? 0);
  const total = offsets[n];
  if (dto.vid.length < total || dto.x.length < total || dto.y.length < total || dto.v.length < total) {
    throw new Error(`Frame payload for ${dto.controller} is shorter than its per-frame counts`);
  }
  const x = Float32Array.from(dto.x);
  const y = Float32Array.from(dto.y);
  const v = Float32Array.from(dto.v);
  const cars = new Int32Array(n);
  const stopped = new Int32Array(n);
  const meanSpeed = new Float32Array(n);
  let maxCars = 0;
  for (let k = 0; k < n; k++) {
    const a = offsets[k];
    const b = offsets[k + 1];
    let stop = 0;
    let sum = 0;
    for (let r = a; r < b; r++) {
      if (v[r] < STOPPED_MS) stop++;
      sum += v[r];
    }
    cars[k] = b - a;
    stopped[k] = stop;
    meanSpeed[k] = b > a ? sum / (b - a) : 0;
    if (cars[k] > maxCars) maxCars = cars[k];
  }
  return {
    controller: dto.controller,
    start: dto.times[0] ?? dto.start,
    end: dto.times[n - 1] ?? dto.end,
    step: dto.step,
    offsets,
    times: Float64Array.from(dto.times),
    vid: Int32Array.from(dto.vid),
    x, y, v,
    origin: dto.origin,
    vehicleCount: dto.vehicle_count,
    tlsIds: dto.tls_ids,
    tls: dto.tls,
    cars, stopped, meanSpeed, maxCars,
  };
}

/** Index of the recorded frame at or before simulated time t (clamped to the recording). */
export function frameIndexAt(fs: FrameSet, t: number): number {
  const n = fs.times.length;
  if (n <= 1) return 0;
  const k = Math.floor((t - fs.times[0]) / fs.step);
  return Math.max(0, Math.min(n - 1, k));
}

/** Reusable output of `Interpolator.sample` (typed arrays are preallocated once for the largest frame). */
export interface CarSample {
  n: number;
  x: Float32Array;
  y: Float32Array;
  v: Float32Array;
}

export function makeCarSample(fs: FrameSet): CarSample {
  const cap = Math.max(1, fs.maxCars);
  return { n: 0, x: new Float32Array(cap), y: new Float32Array(cap), v: new Float32Array(cap) };
}

/**
 * Interpolates every car of frame k towards its position in frame k + 1. Cars are matched by index (`vid`), through a
 * lookup table that is rebuilt only when the playback crosses into a new frame, so sampling costs O(cars) per call.
 */
export class Interpolator {
  private readonly fs: FrameSet;
  private readonly lookup: Int32Array;
  private mappedFrame = -1;

  constructor(fs: FrameSet) {
    this.fs = fs;
    this.lookup = new Int32Array(Math.max(1, fs.vehicleCount)).fill(-1);
  }

  private prepare(k: number): void {
    if (k === this.mappedFrame) return;
    const { offsets, vid } = this.fs;
    const n = this.fs.times.length;
    if (this.mappedFrame >= 0 && this.mappedFrame + 1 < n) {
      for (let r = offsets[this.mappedFrame + 1]; r < offsets[this.mappedFrame + 2]; r++) this.lookup[vid[r]] = -1;
    }
    if (k + 1 < n) {
      for (let r = offsets[k + 1]; r < offsets[k + 2]; r++) this.lookup[vid[r]] = r;
    }
    this.mappedFrame = k;
  }

  sample(t: number, out: CarSample): CarSample {
    const fs = this.fs;
    const n = fs.times.length;
    if (n === 0) {
      out.n = 0;
      return out;
    }
    const k = frameIndexAt(fs, t);
    this.prepare(k);
    const hasNext = k + 1 < n;
    const span = hasNext ? fs.times[k + 1] - fs.times[k] : 1;
    const a = hasNext ? Math.max(0, Math.min(1, (t - fs.times[k]) / span)) : 0;
    const maxSq = MAX_INTERPOLATION_M * MAX_INTERPOLATION_M;
    let i = 0;
    for (let r = fs.offsets[k]; r < fs.offsets[k + 1]; r++, i++) {
      const j = hasNext ? this.lookup[fs.vid[r]] : -1;
      const x0 = fs.x[r];
      const y0 = fs.y[r];
      const v0 = fs.v[r];
      if (j >= 0) {
        const dx = fs.x[j] - x0;
        const dy = fs.y[j] - y0;
        if (dx * dx + dy * dy <= maxSq) {
          out.x[i] = x0 + dx * a;
          out.y[i] = y0 + dy * a;
          out.v[i] = v0 + (fs.v[j] - v0) * a;
          continue;
        }
      }
      out.x[i] = x0;
      out.y[i] = y0;
      out.v[i] = v0;
    }
    out.n = i;
    return out;
  }
}

/** Format simulated seconds since midnight as HH:MM:SS. */
export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600) % 24;
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}
