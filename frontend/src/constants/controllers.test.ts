import { describe, it, expect } from "vitest";
import {
  CONTROLLER_LABELS, CONTROLLER_COLORS, CONTROLLER_HEX, CONTROLLER_SHORT_LABELS,
  REPLAY_WINDOW_START, controllerLabel, controllerShortLabel,
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

describe("controller ids", () => {
  it("covers the baseline, fixed time and the two final models", () => {
    expect(Object.keys(CONTROLLER_LABELS).sort()).toEqual(["fixed_time", "ippo_A", "ippo_B", "sumo_actuated"]);
  });
});

describe("replay window", () => {
  it("every recording covers the 07:00-08:00 peak", () => {
    expect(REPLAY_WINDOW_START).toBe(25200);
  });
});
