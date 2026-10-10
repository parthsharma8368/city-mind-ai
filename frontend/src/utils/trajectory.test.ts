import { describe, it, expect } from "vitest";
import {
  sampleAndHold, tlsFramesToStateMap, msToKmh, isQueuedSpeed, meanSpeedMs, meanSpeedKmh,
  countQueued, buildQueueTrend, signalKind, summarizeSignals,
} from "./trajectory";

describe("sampleAndHold", () => {
  const frames = {
    "0.0": ["a"],
    "5.0": ["b"],
    "10.0": ["c"],
  };

  it("returns the exact frame when the timestamp was recorded", () => {
    expect(sampleAndHold(frames, 5)).toEqual(["b"]);
  });

  it("holds the most recent prior frame between recorded timestamps", () => {
    expect(sampleAndHold(frames, 7.3)).toEqual(["b"]);
  });

  it("returns the last frame when time is past every recorded timestamp", () => {
    expect(sampleAndHold(frames, 999)).toEqual(["c"]);
  });

  it("returns an empty array when time is before every recorded timestamp", () => {
    expect(sampleAndHold(frames, -1)).toEqual([]);
  });

  it("returns an empty array for an empty frame dict", () => {
    expect(sampleAndHold({}, 5)).toEqual([]);
  });
});

describe("tlsFramesToStateMap", () => {
  it("indexes rows by tls_id", () => {
    const map = tlsFramesToStateMap([
      { tls_id: "A", phase: 1, state: "Grr" },
      { tls_id: "B", phase: 0, state: "rGG" },
    ]);
    expect(map).toEqual({
      A: { phase: 1, state: "Grr" },
      B: { phase: 0, state: "rGG" },
    });
  });

  it("returns an empty object for no rows", () => {
    expect(tlsFramesToStateMap([])).toEqual({});
  });
});

describe("speed helpers", () => {
  it("converts m/s to km/h", () => {
    expect(msToKmh(10)).toBeCloseTo(36);
  });
  it("treats <2 m/s as queued, and missing speed as queued", () => {
    expect(isQueuedSpeed(1.99)).toBe(true);
    expect(isQueuedSpeed(2)).toBe(false);
    expect(isQueuedSpeed(undefined)).toBe(true);
  });
  it("averages speed, returning 0 for an empty list and ignoring NaN speeds", () => {
    expect(meanSpeedMs([])).toBe(0);
    expect(meanSpeedMs([{ speed: 2 }, { speed: 4 }])).toBe(3);
    expect(meanSpeedMs([{ speed: NaN }, { speed: 4 }])).toBe(2);
    expect(meanSpeedKmh([{ speed: 10 }])).toBeCloseTo(36);
  });
  it("counts queued vehicles", () => {
    expect(countQueued([{ speed: 0 }, { speed: 1 }, { speed: 5 }])).toBe(2);
  });
});

describe("buildQueueTrend", () => {
  it("returns one point per recorded timestamp, sorted numerically (not lexically)", () => {
    const trend = buildQueueTrend({
      "100.0": [{ speed: 0 }],
      "5.0": [{ speed: 0 }, { speed: 9 }],
      "25.0": [],
    });
    expect(trend.map((p) => p.time)).toEqual([5, 25, 100]);
    expect(trend[0]).toEqual({ time: 5, queued: 1, total: 2 });
    expect(trend[1]).toEqual({ time: 25, queued: 0, total: 0 });
  });
  it("handles integer-style keys ('5') as well as '5.0'", () => {
    const trend = buildQueueTrend({ "5": [{ speed: 0 }] });
    expect(trend).toEqual([{ time: 5, queued: 1, total: 1 }]);
  });
  it("is empty for no frames", () => {
    expect(buildQueueTrend({})).toEqual([]);
  });
});

describe("signal classification", () => {
  it("never invents a colour without a recorded state string", () => {
    expect(signalKind("")).toBeNull();
    expect(signalKind(undefined)).toBeNull();
    expect(signalKind(null)).toBeNull();
  });
  it("classifies recorded SUMO state strings (yellow wins, then green, then red)", () => {
    expect(signalKind("rrGGGg")).toBe("green");
    expect(signalKind("rryyrr")).toBe("yellow");
    expect(signalKind("rrrrrr")).toBe("red");
    expect(signalKind("GGyy")).toBe("yellow");
  });
  it("counts junctions with no recorded state as unknown", () => {
    const s = summarizeSignals(["a", "b", "c", "d"], { a: { phase: 0, state: "GG" }, b: { phase: 1, state: "rr" } });
    expect(s).toEqual({ green: 1, yellow: 0, red: 1, unknown: 2 });
  });
});
