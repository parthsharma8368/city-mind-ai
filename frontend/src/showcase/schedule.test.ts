import { describe, expect, it } from "vitest";
import { BENCH_SECTIONS, CARD_DISSOLVE_MS, COUNT_UP_MS, buildSchedule, easeOutCubic, revealAt, simTimeAt } from "./schedule";

describe("buildSchedule", () => {
  const s = buildSchedule(600, 10, 4);

  it("plays the 10 minute window at 10x in one minute", () => {
    expect(s.simDurationMs).toBe(60_000);
  });

  it("reveals the four controllers one after another, in order", () => {
    expect(s.cardAtMs).toHaveLength(4);
    for (let i = 1; i < 4; i++) expect(s.cardAtMs[i]).toBeGreaterThan(s.cardAtMs[i - 1] + CARD_DISSOLVE_MS);
  });

  it("starts the benchmarks only after every stored result has loaded", () => {
    expect(s.metricsLoadedAtMs).toBe(s.cardAtMs[3] + CARD_DISSOLVE_MS + COUNT_UP_MS);
    expect(s.deltasAtMs).toBeGreaterThan(s.metricsLoadedAtMs);
    expect(s.benchAtMs).toHaveLength(BENCH_SECTIONS);
    expect(s.benchAtMs[0]).toBeGreaterThan(s.deltasAtMs);
    expect(Math.min(...s.benchAtMs)).toBe(s.benchAtMs[0]);
  });

  it("finishes the whole show around the end of the simulation", () => {
    expect(s.completeAtMs).toBeGreaterThanOrEqual(s.simDurationMs);
    expect(s.completeAtMs).toBeLessThan(s.simDurationMs + 10_000);
  });
});

describe("revealAt", () => {
  const s = buildSchedule(600, 10, 4);

  it("shows nothing at the start", () => {
    const r = revealAt(0, s);
    expect(r).toMatchObject({ slotsVisible: false, cardsShown: 0, reading: null, metricsLoaded: false, deltasShown: false, benchShown: 0, complete: false });
  });

  it("reads one controller at a time", () => {
    const r1 = revealAt(s.cardAtMs[0] + 10, s);
    expect(r1.cardsShown).toBe(1);
    expect(r1.reading).toBe(0);
    const between = revealAt(s.cardAtMs[0] + CARD_DISSOLVE_MS + COUNT_UP_MS + 50, s);
    expect(between.cardsShown).toBe(1);
    expect(between.reading).toBeNull();
    expect(revealAt(s.cardAtMs[2] + 1, s).reading).toBe(2);
  });

  it("never shows a benchmark section before all results are loaded", () => {
    for (let t = 0; t < s.metricsLoadedAtMs; t += 250) expect(revealAt(t, s).benchShown).toBe(0);
    expect(revealAt(s.metricsLoadedAtMs, s).metricsLoaded).toBe(true);
  });

  it("adds the benchmark sections in order until everything is on screen", () => {
    let last = 0;
    for (let t = 0; t <= s.completeAtMs; t += 100) {
      const shown = revealAt(t, s).benchShown;
      expect(shown).toBeGreaterThanOrEqual(last);
      last = shown;
    }
    const end = revealAt(s.completeAtMs, s);
    expect(end).toMatchObject({ cardsShown: 4, benchShown: BENCH_SECTIONS, deltasShown: true, complete: true });
  });

  it("copes with no controllers", () => {
    const empty = buildSchedule(600, 10, 0);
    expect(revealAt(empty.completeAtMs, empty).cardsShown).toBe(0);
  });
});

describe("simTimeAt / easing", () => {
  it("advances 10 simulated seconds per real second and stops at the end of the window", () => {
    expect(simTimeAt(0, 26400, 27000, 10)).toBe(26400);
    expect(simTimeAt(1000, 26400, 27000, 10)).toBe(26410);
    expect(simTimeAt(999_999, 26400, 27000, 10)).toBe(27000);
    expect(simTimeAt(-500, 26400, 27000, 10)).toBe(26400);
  });

  it("eases from 0 to 1", () => {
    expect(easeOutCubic(0)).toBe(0);
    expect(easeOutCubic(1)).toBe(1);
    expect(easeOutCubic(0.5)).toBeGreaterThan(0.5);
    expect(easeOutCubic(2)).toBe(1);
  });
});
