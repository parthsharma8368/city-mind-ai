import type { ShowcaseMetrics } from "../api/client";

/** Mean over the evaluated seeds of one measure, or null when it is missing / not finite. */
export function measureMean(m: ShowcaseMetrics | null | undefined, key: string): number | null {
  const found = m?.measures.find((x) => x.key === key);
  return found && Number.isFinite(found.mean) ? found.mean : null;
}

/** Percentage difference of `value` against `base` (null when undefined). */
export function pctDiff(value: number | null, base: number | null): number | null {
  if (value === null || base === null || base === 0) return null;
  return ((value - base) / Math.abs(base)) * 100;
}
