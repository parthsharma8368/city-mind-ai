import { describe, expect, it } from "vitest";
import type { ShowcaseMetrics } from "../api/client";
import { measureMean, pctDiff } from "./measures";

const metrics = (mean: number): ShowcaseMetrics => ({
  controller: "x", label: "X", kind: "model", seeds: [1], n_runs: 1, window: "w", source: "s",
  measures: [{ key: "lag", label: "Lag", better: "lower", format: "{:.1f}", mean, min: mean, max: mean, per_seed: [mean] }],
});

describe("measureMean", () => {
  it("returns the mean, or null when the measure or the value is missing", () => {
    expect(measureMean(metrics(235.4), "lag")).toBe(235.4);
    expect(measureMean(metrics(235.4), "nope")).toBeNull();
    expect(measureMean(null, "lag")).toBeNull();
    expect(measureMean(metrics(Number.NaN), "lag")).toBeNull();
  });
});

describe("pctDiff", () => {
  it("is the signed percentage against the baseline and null when undefined", () => {
    expect(pctDiff(235, 276.8)).toBeCloseTo(-15.1, 1);
    expect(pctDiff(110, 100)).toBeCloseTo(10);
    expect(pctDiff(1, 0)).toBeNull();
    expect(pctDiff(null, 5)).toBeNull();
  });
});
