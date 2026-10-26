import { describe, expect, it } from "vitest";
import { MAX_GREEN, MIN_GREEN, OBS_SIZE, ToyJunction } from "./junction";

describe("ToyJunction", () => {
  it("produces a 22-number observation in [0, 1] with one active phase", () => {
    const j = new ToyJunction(3);
    for (let i = 0; i < 80; i++) {
      const o = j.observe();
      expect(o).toHaveLength(OBS_SIZE);
      for (const v of o) expect(v >= 0 && v <= 1).toBe(true);
      expect(o[0] + o[1] + o[2] + o[3] + o[4] + o[5]).toBe(j.state.transition ? 0 : 1);
      expect(o[6]).toBe(j.state.transition ? 1 : 0);
      j.step(i % 3 === 0);
    }
  });

  it("is deterministic for a seed", () => {
    const a = new ToyJunction(9);
    const b = new ToyJunction(9);
    for (let i = 0; i < 40; i++) {
      a.step(i % 4 === 0);
      b.step(i % 4 === 0);
    }
    expect(a.observe()).toEqual(b.observe());
  });

  it("refuses to end a green before its minimum", () => {
    const j = new ToyJunction(1);
    j.step(true); // elapsed 0 < MIN_GREEN
    expect(j.state.blocked).toBe(true);
    expect(j.state.transition).toBe(false);
    expect(j.state.phase).toBe(0);
  });

  it("goes through a clearance step and then starts the next phase", () => {
    const j = new ToyJunction(1);
    while (j.state.elapsed < MIN_GREEN) j.step(false);
    j.step(true);
    expect(j.state.transition).toBe(true);
    j.step(false);
    expect(j.state.transition).toBe(false);
    expect(j.state.phase).toBe(1);
    expect(j.state.elapsed).toBe(0);
  });

  it("forces a phase off at the maximum green", () => {
    const j = new ToyJunction(1);
    let forced = false;
    for (let i = 0; i < 20 && !forced; i++) {
      j.step(false);
      forced = j.state.forced;
    }
    expect(forced).toBe(true);
    expect(j.state.transition).toBe(true);
    expect(j.state.elapsed).toBeLessThanOrEqual(MAX_GREEN);
  });

  it("keeps queues inside their bounds", () => {
    const j = new ToyJunction(5);
    for (let i = 0; i < 300; i++) {
      j.step(false);
      for (const q of j.state.queues) expect(q >= 0 && q <= 60).toBe(true);
    }
  });
});
