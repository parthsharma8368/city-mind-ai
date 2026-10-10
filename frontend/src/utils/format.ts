/**
 * Defensive number formatting. API payloads are typed as non-null, but a
 * null / missing / NaN field must render as a dash instead of throwing
 * ("Cannot read properties of null (reading 'toFixed')") and blanking a page.
 */
export const NO_VALUE = "—";

export function isNum(n: unknown): n is number {
  return typeof n === "number" && Number.isFinite(n);
}

/** Fixed-decimal formatting; returns `fallback` for null/undefined/NaN/Infinity. */
export function fmt(n: number | null | undefined, digits = 1, fallback = NO_VALUE): string {
  return isNum(n) ? n.toFixed(digits) : fallback;
}

/** Integer count with thousands separators (en-US so output is locale-stable). */
export function fmtInt(n: number | null | undefined, fallback = NO_VALUE): string {
  return isNum(n) ? Math.round(n).toLocaleString("en-US") : fallback;
}

/** Percentage change of `value` relative to `base`; null when it is undefined (base 0 / non-finite). */
export function pctChange(value: number | null | undefined, base: number | null | undefined): number | null {
  if (!isNum(value) || !isNum(base) || base === 0) return null;
  const pct = ((value - base) / Math.abs(base)) * 100;
  return Number.isFinite(pct) ? pct : null;
}

/**
 * Mirrors the backend's starvation threshold: a controller completing more
 * than 2% fewer trips than the baseline is "starving" -- any wait / queue /
 * speed advantage it shows is an artifact of vehicles never finishing.
 */
export const STARVATION_GAP_PCT = -2;

export function isStarving(
  tripsCompleted: number | null | undefined,
  baselineTripsCompleted: number | null | undefined,
): boolean {
  const gap = pctChange(tripsCompleted, baselineTripsCompleted);
  return gap !== null && gap < STARVATION_GAP_PCT;
}

/** Chart value formatting: counts get thousands separators and no decimals. */
export function formatChartValue(v: number | null | undefined, opts: { isCount?: boolean; decimals?: number } = {}): string {
  if (opts.isCount) return fmtInt(v);
  return fmt(v, opts.decimals ?? 3);
}
