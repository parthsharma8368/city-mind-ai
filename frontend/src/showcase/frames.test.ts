import { describe, expect, it } from "vitest";
import type { ShowcaseFramesDTO } from "../api/client";
import { Interpolator, MAX_INTERPOLATION_M, buildFrameSet, formatClock, frameIndexAt, makeCarSample } from "./frames";

/** Three frames 5 s apart. Car 0 drives east, car 1 stands still, car 2 appears in frame 1 and jumps far in frame 2. */
function dto(): ShowcaseFramesDTO {
  return {
    controller: "t", seed: 42, start: 100, end: 110, step: 5, origin: [1000, 2000],
    times: [100, 105, 110],
    counts: [2, 3, 3],
    vid: [0, 1, 0, 1, 2, 0, 1, 2],
    x: [0, 50, 40, 50, 10, 80, 50, 10 + MAX_INTERPOLATION_M + 60],
    y: [0, 50, 0, 50, 10, 0, 50, 10],
    v: [8, 0, 8, 0, 4, 8, 0, 4],
    vehicle_count: 3,
    tls_ids: ["J1"],
    tls: ["g", "r", "y"],
  };
}

describe("buildFrameSet", () => {
  it("indexes the flat arrays per frame and precomputes the live numbers", () => {
    const fs = buildFrameSet(dto());
    expect(Array.from(fs.offsets)).toEqual([0, 2, 5, 8]);
    expect(Array.from(fs.cars)).toEqual([2, 3, 3]);
    expect(Array.from(fs.stopped)).toEqual([1, 1, 1]);          // car 1 is at 0 m/s in every frame
    expect(fs.meanSpeed[0]).toBeCloseTo(4);
    expect(fs.maxCars).toBe(3);
    expect(fs.start).toBe(100);
    expect(fs.end).toBe(110);
  });

  it("refuses a payload shorter than its counts", () => {
    const bad = dto();
    bad.x = bad.x.slice(0, 3);
    expect(() => buildFrameSet(bad)).toThrow(/shorter/);
  });
});

describe("frameIndexAt", () => {
  const fs = buildFrameSet(dto());
  it("floors to the recorded frame and clamps to the recording", () => {
    expect(frameIndexAt(fs, 100)).toBe(0);
    expect(frameIndexAt(fs, 104.9)).toBe(0);
    expect(frameIndexAt(fs, 105)).toBe(1);
    expect(frameIndexAt(fs, 99)).toBe(0);
    expect(frameIndexAt(fs, 500)).toBe(2);
  });
});

describe("Interpolator", () => {
  it("moves a car smoothly between two frames and keeps a stopped car still", () => {
    const fs = buildFrameSet(dto());
    const it = new Interpolator(fs);
    const out = makeCarSample(fs);
    it.sample(102.5, out);                                      // halfway through frame 0 -> 1
    expect(out.n).toBe(2);
    expect(out.x[0]).toBeCloseTo(20);                           // car 0: 0 -> 40
    expect(out.x[1]).toBeCloseTo(50);                           // car 1 does not move
    expect(out.v[0]).toBeCloseTo(8);
  });

  it("is exact on a frame boundary and survives jumping around in time (restart / skip)", () => {
    const fs = buildFrameSet(dto());
    const it = new Interpolator(fs);
    const out = makeCarSample(fs);
    it.sample(105, out);
    expect(out.n).toBe(3);
    expect(out.x[0]).toBeCloseTo(40);
    it.sample(110, out);                                        // last frame: nothing to interpolate towards
    expect(out.n).toBe(3);
    expect(out.x[0]).toBeCloseTo(80);
    it.sample(101, out);                                        // back to the start
    expect(out.n).toBe(2);
    expect(out.x[0]).toBeCloseTo(8);
  });

  it("does not drag a car across the map when it re-appears far away", () => {
    const fs = buildFrameSet(dto());
    const it = new Interpolator(fs);
    const out = makeCarSample(fs);
    it.sample(107.5, out);                                      // frame 1 -> 2: car 2 jumps by more than MAX_INTERPOLATION_M
    expect(out.x[2]).toBeCloseTo(10);                           // held at its frame-1 position
    expect(out.x[0]).toBeCloseTo(60);                           // car 0 is still interpolated: 40 -> 80
  });
});

describe("formatClock", () => {
  it("formats simulated seconds as HH:MM:SS", () => {
    expect(formatClock(26400)).toBe("07:20:00");
    expect(formatClock(27000 - 1)).toBe("07:29:59");
    expect(formatClock(-5)).toBe("00:00:00");
  });
});
