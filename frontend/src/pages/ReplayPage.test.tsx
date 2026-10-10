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

describe("ReplayPage legacy ippo handling", () => {
  it("does not offer the midnight-only ippo recording in the side-by-side RL picker", async () => {
    renderPage();
    const group = await screen.findByRole("group", { name: /RL controller to compare/i });
    expect(group.textContent).toContain("PPO");
    expect(group.textContent).not.toMatch(/^IPPO|\bIPPO\b(?!_)/);
    expect(group.textContent).toContain("PHASE-SELECT");
  });

  it("only ever requests peak-window data for both sides of the split view", async () => {
    renderPage();
    await waitFor(() => expect(replay).toHaveBeenCalledTimes(2));
    const starts = replay.mock.calls.map((c: unknown[]) => c[3]);
    expect(starts).toEqual([25200, 25200]);
  });

  it("explains that ippo is not replayable under the current environment in single view", async () => {
    renderPage();
    fireEvent.click(await screen.findByRole("button", { name: /Single View/i }));
    fireEvent.click(await screen.findByRole("button", { name: /IPPO \(Parameter-Shared\)/i }));
    const note = await screen.findByRole("note");
    expect(note.textContent).toMatch(/Not replayable under the current environment/);
    expect(note.textContent).toMatch(/midnight/);
  });

  it("exposes accessible names and pressed state on the scrubber and toggles", async () => {
    renderPage();
    expect(await screen.findByLabelText("Replay position")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Side-by-Side Dual View/i })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Playback speed 2x" })).toHaveAttribute("aria-pressed", "false");
  });
});
