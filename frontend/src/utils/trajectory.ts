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

/** SUMO reports speed in m/s everywhere; the UI always displays km/h. */
export function msToKmh(speedMs: number): number {
  return speedMs * 3.6;
}

/** A vehicle under this speed (m/s) is considered stopped/queued, not just slow. */
export const QUEUED_SPEED_THRESHOLD_MS = 2;

export function isQueuedSpeed(speedMs: number | undefined): boolean {
  return (speedMs ?? 0) < QUEUED_SPEED_THRESHOLD_MS;
}

/** Mean speed (m/s, SUMO's native unit) across a list of vehicles carrying a `speed` field. */
export function meanSpeedMs(vehicles: Array<{ speed: number }>): number {
  if (vehicles.length === 0) return 0;
  const sum = vehicles.reduce((acc, v) => acc + (v.speed || 0), 0);
  return sum / vehicles.length;
}

/** Mean speed (km/h) across a list of vehicles carrying a `speed` (m/s) field. */
export function meanSpeedKmh(vehicles: Array<{ speed: number }>): number {
  return msToKmh(meanSpeedMs(vehicles));
}

/** Count of vehicles under the queued-speed threshold. */
export function countQueued(vehicles: Array<{ speed: number }>): number {
  return vehicles.filter((v) => isQueuedSpeed(v.speed)).length;
}
