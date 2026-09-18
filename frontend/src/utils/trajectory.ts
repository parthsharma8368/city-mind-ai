import type { TLSFrame } from "../api/client";

/**
 * Sample-and-hold lookup into a { "12.0": [...] } frame dict: exact match if
 * the timestamp was recorded, otherwise the most recent frame at or before
 * `time`. Works for vehicle-frame dicts and TLS-frame dicts alike — both are
 * keyed the same way by the replay API.
 */
export function sampleAndHold<T>(dict: Record<string, T[]>, time: number): T[] {
  const exact = dict[time.toFixed(1)] ?? dict[String(time)];
  if (exact) return exact;

  let bestKey: string | null = null;
  let bestTime = -Infinity;
  for (const key of Object.keys(dict)) {
    const t = parseFloat(key);
    if (t <= time && t > bestTime) {
      bestTime = t;
      bestKey = key;
    }
  }
  return bestKey !== null ? dict[bestKey] : [];
}

/** Convert one time-slice of TLS rows into the {tls_id: {phase, state}} shape NetworkCanvas expects. */
export function tlsFramesToStateMap(rows: TLSFrame[]): Record<string, { phase: number; state: string }> {
  const res: Record<string, { phase: number; state: string }> = {};
  for (const row of rows) {
    res[row.tls_id] = { phase: row.phase, state: row.state };
  }
  return res;
}
