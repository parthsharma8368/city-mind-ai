import { describe, it, expect } from "vitest";
import { metricDelta } from "./compare";

describe("metricDelta", () => {
  it("paints an honest wait-time win green for a non-starving controller", () => {
    const d = metricDelta("mean_waiting_time", 5, 10, false)!;
    expect(d.improved).toBe(true);
    expect(d.tone).toBe("positive");
  });
  it("never paints a starving controller's wait/queue/speed 'win' green", () => {
    for (const [label, mine, base] of [
      ["mean_waiting_time", 5, 10],
      ["mean_queue", 0.4, 0.5],
      ["mean_speed", 12, 10],
      ["throughput", 0.9, 0.8],
    ] as const) {
      const d = metricDelta(label, mine, base, true)!;
      expect(d.improved).toBe(true);
      expect(d.suspect).toBe(true);
      expect(d.tone).toBe("neutral");
    }
  });
  it("keeps trips_completed and backlog honest (red) for a starving controller", () => {
    expect(metricDelta("trips_completed", 9000, 11000, true)!.tone).toBe("negative");
    expect(metricDelta("backlog", 4000, 3000, true)!.tone).toBe("negative");
  });
  it("reports direction of change by the sign of diff, independent of good/bad", () => {
    const up = metricDelta("throughput", 1.2, 1, false)!; // higher is better and went up
    expect(up.diff).toBeGreaterThan(0);
    expect(up.tone).toBe("positive");
    const down = metricDelta("throughput", 0.8, 1, false)!;
    expect(down.diff).toBeLessThan(0);
    expect(down.tone).toBe("negative");
  });
  it("returns null when the baseline is unusable", () => {
    expect(metricDelta("mean_queue", 1, 0, false)).toBeNull();
    expect(metricDelta("mean_queue", NaN, 1, false)).toBeNull();
  });
});
