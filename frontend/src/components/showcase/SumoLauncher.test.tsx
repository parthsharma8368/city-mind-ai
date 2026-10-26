import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { SumoLauncher } from "./SumoLauncher";
import { api } from "../../api/client";

const CONTROLLERS = [
  { id: "sumo_actuated", label: "SUMO actuated (baseline)", kind: "baseline", available: true },
  { id: "fixed_time", label: "Fixed time", kind: "baseline", available: true },
  { id: "ippo_A", label: "Model A (final)", kind: "model", available: true },
  { id: "ippo_B", label: "Model B (final)", kind: "model", available: false },
] as const;

describe("SumoLauncher", () => {
  beforeEach(() => {
    vi.spyOn(api, "getSimulationStatus").mockResolvedValue({ gui_running: false, running_controllers: [], controllers: [...CONTROLLERS] });
    vi.spyOn(api, "launchSimulationGui").mockResolvedValue({ status: "launched", message: "SUMO-GUI is starting" });
    vi.spyOn(api, "stopSimulationGui").mockResolvedValue({ status: "stopped", message: "Stopped 1 SUMO-GUI process(es)." });
  });
  afterEach(() => vi.restoreAllMocks());

  it("opens a controller in SUMO by id only (never a model path)", async () => {
    render(<SumoLauncher colors={{}} />);
    await screen.findByRole("button", { name: /Model file missing/ });   // the server's status has arrived
    const buttons = screen.getAllByRole("button", { name: /Open in SUMO/ });
    expect(buttons).toHaveLength(3);                       // Model B's file is missing, so its button says so instead
    fireEvent.click(buttons[2]);
    await waitFor(() => expect(api.launchSimulationGui).toHaveBeenCalledWith("ippo_A", undefined));
    expect(await screen.findByText(/SUMO-GUI is starting/)).toBeTruthy();
    expect(screen.getByRole("button", { name: /Opening/ })).toBeTruthy();
  });

  it("explains an unavailable model instead of failing silently", async () => {
    render(<SumoLauncher colors={{}} />);
    const missing = await screen.findByRole("button", { name: /Model file missing/ });
    expect((missing as HTMLButtonElement).disabled).toBe(true);
  });

  it("shows the server's reason when a launch is refused", async () => {
    vi.spyOn(api, "launchSimulationGui").mockRejectedValue(new Error("API /simulation/launch-gui → 501: scripts/run_visual_demo.py is missing"));
    render(<SumoLauncher colors={{}} />);
    fireEvent.click((await screen.findAllByRole("button", { name: /Open in SUMO/ }))[0]);
    expect((await screen.findByRole("alert")).textContent).toMatch(/run_visual_demo.py is missing/);
  });

  it("tiles the four windows when asked to open them all", async () => {
    vi.spyOn(api, "getSimulationStatus").mockResolvedValue({ gui_running: false, running_controllers: [], controllers: CONTROLLERS.map((c) => ({ ...c, available: true })) });
    render(<SumoLauncher colors={{}} />);
    fireEvent.click(await screen.findByRole("button", { name: /Open all four/ }));
    await waitFor(() => expect(api.launchSimulationGui).toHaveBeenCalledTimes(4), { timeout: 6000 });
    const calls = vi.mocked(api.launchSimulationGui).mock.calls;
    expect(calls.map((c) => c[0])).toEqual(["sumo_actuated", "fixed_time", "ippo_A", "ippo_B"]);
    expect(new Set(calls.map((c) => c[1]?.pos)).size).toBe(4);   // four different window positions
  });
});
