import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { ComparisonPage } from "./ComparisonPage";
import { api } from "../api/client";
import type { ComparisonResult } from "../api/client";

afterEach(() => {
  vi.restoreAllMocks();
});

const baseResult: ComparisonResult = {
  results: [
    { controller: "fixed_time", label: "Fixed-Time", mean_waiting_time: 3.19, mean_queue: 0.057, throughput: 0.71, mean_speed: 10.66, total_arrived: 150, cumulative_reward: 0 },
    { controller: "ppo", label: "PPO (RL)", mean_waiting_time: 13.59, mean_queue: 0.002, throughput: 0.84, mean_speed: 0.88, total_arrived: 226, cumulative_reward: -2757 },
  ],
  improvement: { waiting_time_improvement_pct: -325.9, throughput_improvement_pct: 17.9 },
};

describe("ComparisonPage improvement banner", () => {
  it("renders a NEGATIVE badge with 'increase' wording when PPO is worse than the baseline", async () => {
    vi.spyOn(api, "getComparisonMetrics").mockResolvedValue(baseResult);
    render(<ComparisonPage />);

    const badge = await waitFor(() => screen.getByText(/wait time increase/i));
    expect(badge.textContent).toContain("325.9%");
    expect(badge.className).toContain("negative");
    expect(badge.className).not.toContain("positive");
  });

  it("renders a POSITIVE badge with 'reduction' wording when PPO genuinely beats the baseline", async () => {
    const winning: ComparisonResult = {
      ...baseResult,
      improvement: { waiting_time_improvement_pct: 25.0 },
    };
    vi.spyOn(api, "getComparisonMetrics").mockResolvedValue(winning);
    render(<ComparisonPage />);

    const badge = await waitFor(() => screen.getByText(/wait time reduction/i));
    expect(badge.textContent).toContain("25.0%");
    expect(badge.className).toContain("positive");
  });
});
