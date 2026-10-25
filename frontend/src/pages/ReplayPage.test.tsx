import { describe, it, expect, vi, afterEach, beforeEach, type MockInstance } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { ReplayPage } from "./ReplayPage";
import { api } from "../api/client";

vi.mock("../components/NetworkCanvas", () => ({ NetworkCanvas: () => <div data-testid="canvas" /> }));

const frames = (t: number) => ({
  frames: { [`${t}.0`]: [{ vehicle_id: "v", x: 0, y: 0, speed: 1 }] },
  tls_frames: {},
  frame_count: 1,
});

let replay: MockInstance<typeof api.getReplayFrames>;
beforeEach(() => {
  vi.spyOn(api, "getComparisonMetrics").mockResolvedValue({ results: [] });
  replay = vi.spyOn(api, "getReplayFrames").mockImplementation(async (_c, _s, _d, start = 0) => frames(start));
});
afterEach(() => vi.restoreAllMocks());

const renderPage = () => render(<MemoryRouter><ReplayPage /></MemoryRouter>);

describe("ReplayPage final models", () => {
  it("offers the two final models in the side-by-side picker", async () => {
    renderPage();
    const group = await screen.findByRole("group", { name: /RL controller to compare/i });
    expect(group.textContent).toContain("MODEL A");
    expect(group.textContent).toContain("MODEL B");
  });

  it("requests the peak window for both sides of the split view, SUMO actuated against model A", async () => {
    renderPage();
    await waitFor(() => expect(replay).toHaveBeenCalledTimes(2));
    expect(replay.mock.calls.map((c: unknown[]) => c[0])).toEqual(["sumo_actuated", "ippo_A"]);
    expect(replay.mock.calls.map((c: unknown[]) => c[3])).toEqual([25200, 25200]);
  });

  it("lists the four recorded runs in single view", async () => {
    renderPage();
    fireEvent.click(await screen.findByRole("button", { name: /Single View/i }));
    for (const label of [/Model A \(final\)/, /Model B \(final\)/, /SUMO actuated \(baseline\)/, /^Fixed time$/]) {
      expect(await screen.findByRole("button", { name: label })).toBeInTheDocument();
    }
  });

  it("exposes accessible names and pressed state on the scrubber and toggles", async () => {
    renderPage();
    expect(await screen.findByLabelText("Replay position")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Side-by-Side Dual View/i })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Playback speed 2x" })).toHaveAttribute("aria-pressed", "false");
  });
});
