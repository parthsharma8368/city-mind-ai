import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { api, type PolicyDTO } from "../api/client";
import { BrainPage } from "./BrainPage";

const dense = (rows: number, cols: number) => ({ shape: [rows, cols] as [number, number], w: new Array(rows * cols).fill(0.01), b: new Array(rows).fill(0) });
const net = (out: number) => ({ l1: dense(4, 22), l2: dense(4, 4), out: dense(out, 4) });
const dto = (controller: string): PolicyDTO => ({
  controller, label: controller, obs_names: Array.from({ length: 22 }, (_, i) => `number ${i}`),
  actions: ["keep this green", "advance to the next green"], activation: "tanh", actor: net(2), critic: net(1),
});

describe("BrainPage", () => {
  afterEach(() => vi.restoreAllMocks());

  it("loads the weights of Model A, shows the parameter counts and the 22 inputs", async () => {
    const spy = vi.spyOn(api, "getPolicy").mockResolvedValue(dto("ippo_A"));
    render(<BrainPage />);
    expect(screen.getByRole("status").textContent).toMatch(/Loading the weights of Model A/);
    expect(await screen.findByText(/Actor says/)).toBeTruthy();
    expect(spy.mock.calls[0][0]).toBe("ippo_A");
    // actor: (4*22+4) + (4*4+4) + (2*4+2) = 122 weights and biases; the critic's single output leaves 117
    expect(screen.getByText(/· 122 weights/)).toBeTruthy();
    expect(screen.getByText(/· 117 weights/)).toBeTruthy();
    expect(screen.getAllByText(/^number \d+$/)).toHaveLength(22);
    expect(screen.getByRole("img", { name: /3D figure/ })).toBeTruthy();
  });

  it("switches to Model B by controller id", async () => {
    const spy = vi.spyOn(api, "getPolicy").mockImplementation(async (id) => dto(id));
    render(<BrainPage />);
    await screen.findByText(/Actor says/);
    fireEvent.click(screen.getByRole("button", { name: "Model B" }));
    await waitFor(() => expect(spy.mock.calls.map((c) => c[0])).toEqual(["ippo_A", "ippo_B"]));
    expect(await screen.findByText(/trained ones of Model B/)).toBeTruthy();
  });

  it("explains a failed load and retries", async () => {
    const spy = vi.spyOn(api, "getPolicy").mockRejectedValueOnce(new Error("API /showcase/policy/ippo_A → 404: Model file is missing")).mockResolvedValue(dto("ippo_A"));
    render(<BrainPage />);
    expect((await screen.findByRole("alert")).textContent).toMatch(/Model file is missing/);
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText(/Actor says/)).toBeTruthy();
    expect(spy).toHaveBeenCalledTimes(2);
  });
});
