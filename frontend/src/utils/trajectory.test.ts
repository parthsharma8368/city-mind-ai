import { describe, it, expect } from "vitest";
import { sampleAndHold, tlsFramesToStateMap } from "./trajectory";

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
