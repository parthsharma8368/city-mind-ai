import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { api, _fetch, API_BASE, API_DOCS_URL, WS_SIMULATION, isAbortError } from "./client";

function jsonResponse(body: unknown, init: { status?: number; ok?: boolean } = {}) {
  const status = init.status ?? 200;
  return {
    ok: init.ok ?? (status >= 200 && status < 300),
    status,
    json: async () => body,
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
  } as unknown as Response;
}

let fetchMock: ReturnType<typeof vi.fn>;
beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("_fetch error handling", () => {
  it("throws with path, status and truncated body text on non-OK responses", async () => {
    fetchMock.mockResolvedValue(jsonResponse("x".repeat(500), { status: 503 }));
    const err = await _fetch("/metrics/summary").catch((e: Error) => e);
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).message).toContain("/metrics/summary");
    expect((err as Error).message).toContain("503");
    expect((err as Error).message.length).toBeLessThan(260); // 200-char body cap + prefix
  });

  it("still throws a useful error when the error body can't be read", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 500, text: async () => { throw new Error("boom"); } } as unknown as Response);
    await expect(_fetch("/x")).rejects.toThrow(/500/);
  });

  it("turns an invalid-JSON 200 into a descriptive error", async () => {
    fetchMock.mockResolvedValue({ ok: true, status: 200, json: async () => { throw new SyntaxError("bad"); } } as unknown as Response);
    await expect(_fetch("/x")).rejects.toThrow(/not valid JSON/);
  });

  it("propagates network failures and recognises aborts", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(_fetch("/x")).rejects.toThrow("Failed to fetch");
    expect(isAbortError(new DOMException("aborted", "AbortError"))).toBe(true);
    expect(isAbortError(new Error("nope"))).toBe(false);
  });

  it("never sends an API key header (no secret is bundled into the client)", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ ok: 1 }));
    await _fetch("/x");
    const init = fetchMock.mock.calls[0][1] as RequestInit | undefined;
    const headers = new Headers(init?.headers);
    expect(headers.has("X-API-Key")).toBe(false);
  });
});

describe("URL building", () => {
  beforeEach(() => fetchMock.mockResolvedValue(jsonResponse({})));
  const lastUrl = () => String(fetchMock.mock.calls.at(-1)![0]);

  it("encodes path and query parameters", async () => {
    await api.getExperimentSummary("run/../a b");
    expect(lastUrl()).toBe(`${API_BASE}/experiments/run%2F..%2Fa%20b/summary`);
    await api.getXAIJunction("tls 1/2", "pp&o");
    expect(lastUrl()).toBe(`${API_BASE}/xai/explain/tls%201%2F2?controller=pp%26o`);
  });

  it("sends launch parameters as query string on a POST", async () => {
    await api.launchSimulationGui("ippo_A", { delay: 80, duration: 3600, pos: "0,40", size: "800,500" });
    const url = new URL(lastUrl());
    expect(url.pathname.endsWith("/simulation/launch-gui")).toBe(true);
    expect(url.searchParams.get("controller")).toBe("ippo_A");
    expect(url.searchParams.get("model")).toBeNull();   // the server only ever loads its own two model files
    expect(url.searchParams.get("delay")).toBe("80");
    expect(url.searchParams.get("duration")).toBe("3600");
    expect(url.searchParams.get("pos")).toBe("0,40");
    expect(url.searchParams.get("size")).toBe("800,500");
    expect((fetchMock.mock.calls.at(-1)![1] as RequestInit).method).toBe("POST");
  });

  it("omits the window options when unset and builds stop-gui with and without a controller", async () => {
    await api.launchSimulationGui("fixed_time");
    expect(new URL(lastUrl()).searchParams.has("pos")).toBe(false);
    expect(new URL(lastUrl()).searchParams.has("size")).toBe(false);
    await api.stopSimulationGui();
    expect(lastUrl().endsWith("/simulation/stop-gui")).toBe(true);
    await api.stopSimulationGui("ippo_B");
    expect(new URL(lastUrl()).searchParams.get("controller")).toBe("ippo_B");
  });

  it("builds replay queries and forwards the abort signal", async () => {
    const ctrl = new AbortController();
    await api.getReplayFrames("ippo_A", 42, "1.0x", 25200, 25500, ctrl.signal);
    const url = new URL(lastUrl());
    expect(url.searchParams.get("start_time")).toBe("25200");
    expect(url.searchParams.get("end_time")).toBe("25500");
    expect(url.searchParams.get("demand_id")).toBe("1.0x");
    expect((fetchMock.mock.calls.at(-1)![1] as RequestInit).signal).toBe(ctrl.signal);
  });
});

describe("configuration defaults", () => {
  it("derives docs and websocket URLs from the same origin as the API", () => {
    expect(API_DOCS_URL).toBe(`${API_BASE.replace(/\/api$/, "")}/docs`);
    expect(WS_SIMULATION.startsWith("ws")).toBe(true);
  });
});

describe("network GeoJSON cache", () => {
  it("downloads the GeoJSON once for many callers and retries after a failure", async () => {
    vi.resetModules();
    const mod = await import("./client");
    fetchMock.mockRejectedValueOnce(new TypeError("offline"));
    await expect(mod.api.getNetworkGeoJSON()).rejects.toThrow("offline");

    fetchMock.mockResolvedValue(jsonResponse({ type: "FeatureCollection", features: [] }));
    const [a, b, c] = await Promise.all([mod.api.getNetworkGeoJSON(), mod.api.getNetworkGeoJSON(), mod.api.getNetworkGeoJSON()]);
    await mod.api.getNetworkGeoJSON();
    expect(a).toBe(b);
    expect(b).toBe(c);
    // 1 failed attempt + exactly 1 successful download shared by all callers
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
