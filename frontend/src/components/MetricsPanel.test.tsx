import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MetricsPanel } from "./MetricsPanel";

describe("MetricsPanel", () => {
  it("shows a dash, not a fake 0, when no data has loaded", () => {
    const { container } = render(<MetricsPanel />);
    expect(container.textContent).not.toMatch(/0\.0 s|0\.0 km\/h/);
    expect(screen.getAllByLabelText("no data").length).toBeGreaterThanOrEqual(4);
  });

  it("keeps the episode mean separate from the instantaneous queued count", () => {
    render(<MetricsPanel avgWait={7.6} avgQueue={0.602} avgSpeed={10.28} arrivedCount={9547} queuedNow={246} vehiclesNow={1200} />);
    expect(screen.getByText("0.60 veh")).toBeInTheDocument(); // REST mean_queue, not replaced by 246
    expect(screen.getByText("9,547")).toBeInTheDocument();
    expect(screen.getByText(/Queued now/)).toBeInTheDocument();
    expect(screen.getByText(/246/)).toBeInTheDocument();
  });

  it("does not crash on NaN values", () => {
    render(<MetricsPanel avgWait={NaN} avgQueue={null} avgSpeed={Infinity} arrivedCount={undefined} />);
    expect(screen.getAllByLabelText("no data").length).toBeGreaterThanOrEqual(4);
  });
});
