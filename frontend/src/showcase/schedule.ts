/**
 * Timeline of the showcase, in milliseconds since the Start click (the master clock stops while paused):
 *
 *   0 s            the four recorded runs start playing at 10x
 *   3 s            the result slots appear as empty outlines
 *   5 s, 12 s ...  one controller's stored results dissolve in (then its numbers count up)
 *   ~28 s          all stored results are loaded: the comparison line appears
 *   ~31 s          benchmark sections dissolve in one after another, the last one is the 10-seed test
 *
 * Nothing here depends on the DOM, so the order can be tested exactly (schedule.test.ts).
 */

export const SLOT_OUTLINE_MS = 3_000;
export const FIRST_CARD_MS = 5_000;
export const CARD_GAP_MS = 7_000;
export const CARD_DISSOLVE_MS = 1_100;
export const COUNT_UP_MS = 1_600;
export const BENCH_FIRST_GAP_MS = 3_000;
export const BENCH_GAP_MS = 5_500;
export const BENCH_SECTIONS = 4;

export interface Schedule {
  /** Real milliseconds the recorded window takes to play at the chosen speed. */
  simDurationMs: number;
  slotsAtMs: number;
  /** When controller i's results start to dissolve in. */
  cardAtMs: number[];
  /** All stored results are on screen and counted up. */
  metricsLoadedAtMs: number;
  /** The "model vs baseline" line. */
  deltasAtMs: number;
  /** When benchmark section j starts to dissolve in. */
  benchAtMs: number[];
  /** Everything is on screen. */
  completeAtMs: number;
}

export function buildSchedule(windowSeconds: number, speed: number, nControllers: number): Schedule {
  const simDurationMs = Math.round((windowSeconds / Math.max(speed, 0.1)) * 1000);
  const cardAtMs = Array.from({ length: nControllers }, (_, i) => FIRST_CARD_MS + i * CARD_GAP_MS);
  const lastCard = cardAtMs.length ? cardAtMs[cardAtMs.length - 1] : FIRST_CARD_MS;
  const metricsLoadedAtMs = lastCard + CARD_DISSOLVE_MS + COUNT_UP_MS;
  const deltasAtMs = metricsLoadedAtMs + 800;
  const firstBench = deltasAtMs + BENCH_FIRST_GAP_MS;
  const benchAtMs = Array.from({ length: BENCH_SECTIONS }, (_, j) => firstBench + j * BENCH_GAP_MS);
  return {
    simDurationMs,
    slotsAtMs: SLOT_OUTLINE_MS,
    cardAtMs,
    metricsLoadedAtMs,
    deltasAtMs,
    benchAtMs,
    completeAtMs: Math.max(simDurationMs, benchAtMs[benchAtMs.length - 1] + 1_500),
  };
}

export interface RevealState {
  slotsVisible: boolean;
  /** Controllers whose results have started to dissolve in (a prefix of the controller order). */
  cardsShown: number;
  /** The result being read right now (null before the first / after the last). */
  reading: number | null;
  metricsLoaded: boolean;
  deltasShown: boolean;
  benchShown: number;
  complete: boolean;
}

export function revealAt(elapsedMs: number, s: Schedule): RevealState {
  const cardsShown = s.cardAtMs.filter((t) => elapsedMs >= t).length;
  const readingIdx = cardsShown - 1;
  const reading =
    cardsShown > 0 && elapsedMs < s.cardAtMs[readingIdx] + CARD_DISSOLVE_MS + COUNT_UP_MS ? readingIdx : null;
  return {
    slotsVisible: elapsedMs >= s.slotsAtMs,
    cardsShown,
    reading,
    metricsLoaded: elapsedMs >= s.metricsLoadedAtMs,
    deltasShown: elapsedMs >= s.deltasAtMs,
    // Benchmarks only start once every stored result has loaded (the schedule guarantees it; the check keeps it true).
    benchShown: elapsedMs >= s.metricsLoadedAtMs ? s.benchAtMs.filter((t) => elapsedMs >= t).length : 0,
    complete: elapsedMs >= s.completeAtMs,
  };
}

/** Simulated time (s since midnight) at `elapsedMs`, clamped to the recorded window. */
export function simTimeAt(elapsedMs: number, windowStart: number, windowEnd: number, speed: number): number {
  return Math.min(windowEnd, windowStart + (Math.max(0, elapsedMs) / 1000) * speed);
}

/** 0..1 ease-out used by the number count-up. */
export function easeOutCubic(p: number): number {
  const c = Math.max(0, Math.min(1, p));
  return 1 - Math.pow(1 - c, 3);
}
