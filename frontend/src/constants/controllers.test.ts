import { describe, it, expect } from "vitest";
import {
  CONTROLLER_LABELS, CONTROLLER_COLORS, CONTROLLER_HEX, CONTROLLER_SHORT_LABELS,
  replayWindowStart, isPeakAligned, controllerLabel, controllerShortLabel,
} from "./controllers";

describe("controller metadata", () => {
  it("keeps label, colour, hex and short-label maps on exactly the same controller ids", () => {
    const ids = Object.keys(CONTROLLER_LABELS).sort();
    expect(Object.keys(CONTROLLER_COLORS).sort()).toEqual(ids);
    expect(Object.keys(CONTROLLER_HEX).sort()).toEqual(ids);
    expect(Object.keys(CONTROLLER_SHORT_LABELS).sort()).toEqual(ids);
  });
  it("hex values are valid 6-digit colours", () => {
    for (const hex of Object.values(CONTROLLER_HEX)) expect(hex).toMatch(/^#[0-9a-f]{6}$/i);
  });
  it("short labels are short enough for phone-width chart axes", () => {
    for (const label of Object.values(CONTROLLER_SHORT_LABELS)) expect(label.length).toBeLessThanOrEqual(10);
  });
  it("falls back to the raw id for unknown controllers", () => {
    expect(controllerLabel("mystery")).toBe("mystery");
    expect(controllerShortLabel("mystery")).toBe("mystery");
  });
});

describe("replay windows", () => {
  it("requests the peak window for current controllers and midnight only for legacy ippo", () => {
    expect(replayWindowStart("ppo")).toBe(25200);
    expect(replayWindowStart("fixed_time")).toBe(25200);
    expect(replayWindowStart("ippo")).toBe(0);
  });
  it("marks ippo as not peak-aligned so it can't be paired with peak baselines", () => {
    expect(isPeakAligned("ippo")).toBe(false);
    expect(isPeakAligned("ppo")).toBe(true);
    expect(isPeakAligned("ippo_phase_select")).toBe(true);
  });
});
