import { describe, it, expect } from "vitest";
import { fmt, fmtInt, pctChange, isStarving, formatChartValue, NO_VALUE } from "./format";

describe("fmt / fmtInt", () => {
  it("renders null, undefined, NaN and Infinity as a dash instead of throwing", () => {
    for (const bad of [null, undefined, NaN, Infinity, -Infinity]) {
      expect(fmt(bad, 2)).toBe(NO_VALUE);
      expect(fmtInt(bad)).toBe(NO_VALUE);
    }
  });
  it("keeps a real zero distinct from missing data", () => {
    expect(fmt(0, 1)).toBe("0.0");
    expect(fmtInt(0)).toBe("0");
  });
  it("formats counts with thousands separators and no decimals", () => {
    expect(fmtInt(9547)).toBe("9,547");
    expect(fmtInt(9546.6)).toBe("9,547");
    expect(formatChartValue(11500, { isCount: true })).toBe("11,500");
    expect(formatChartValue(0.60234, { decimals: 2 })).toBe("0.60");
    expect(formatChartValue(1, {})).toBe("1.000");
  });
});

describe("pctChange / isStarving", () => {
  it("returns null instead of Infinity/NaN for a zero or missing baseline", () => {
    expect(pctChange(5, 0)).toBeNull();
    expect(pctChange(5, null)).toBeNull();
    expect(pctChange(undefined, 5)).toBeNull();
  });
  it("computes signed percent change", () => {
    expect(pctChange(90, 100)).toBeCloseTo(-10);
    expect(pctChange(110, 100)).toBeCloseTo(10);
  });
  it("flags only controllers completing >2% fewer trips than baseline", () => {
    expect(isStarving(9547, 11621)).toBe(true);
    expect(isStarving(11500, 11621)).toBe(false); // -1.04%
    expect(isStarving(11621 * 0.985, 11621)).toBe(false); // -1.5%: inside the threshold
    expect(isStarving(null, 11621)).toBe(false);
    expect(isStarving(100, 0)).toBe(false);
  });
});
