import { pctChange } from "./format";

export type MetricKey = "mean_waiting_time" | "mean_queue" | "throughput" | "mean_speed" | "trips_completed" | "backlog";

/**
 * Delta of a metric vs the baseline plus how to colour it. Starving controllers
 * never get a green tag on metrics they only "win" by not finishing trips.
 */
export function metricDelta(label: MetricKey, myVal: number | null | undefined, baseVal: number | null | undefined, starving: boolean) {
  const diff = pctChange(myVal, baseVal);
  if (diff === null) return null;
  // Higher-is-better metrics: throughput, mean_speed, trips_completed.
  // Everything else (including backlog) is lower-is-better.
  const lowerIsBetter = label !== "throughput" && label !== "mean_speed" && label !== "trips_completed";
  const improved = lowerIsBetter ? diff < 0 : diff > 0;
  // A starving controller (completes >2% fewer trips than the baseline) only
  // "wins" on wait / queue / speed / throughput because vehicles never finish:
  // never paint that green. trips_completed and backlog stay honest.
  const suspect = starving && improved && label !== "trips_completed" && label !== "backlog";
  const tone: "positive" | "negative" | "neutral" = suspect ? "neutral" : improved ? "positive" : "negative";
  return { diff, improved, suspect, tone };
}

