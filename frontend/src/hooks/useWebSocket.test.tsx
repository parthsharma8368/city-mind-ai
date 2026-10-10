import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { StrictMode } from "react";
import { renderHook, act } from "@testing-library/react";
import { useWebSocket, backoffDelay } from "./useWebSocket";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  closed = false;
  url: string;
  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }
  close() {
    this.closed = true;
    // Real browsers fire onclose asynchronously after close().
    queueMicrotask(() => this.onclose?.());
  }
  get live() {
    return !this.closed;
  }
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const live = () => FakeWebSocket.instances.filter((w) => w.live);

describe("backoffDelay", () => {
  it("doubles per attempt and is capped", () => {
    const mid = () => 0.5; // zero jitter
    expect(backoffDelay(0, 3000, 30000, mid)).toBe(3000);
    expect(backoffDelay(1, 3000, 30000, mid)).toBe(6000);
    expect(backoffDelay(2, 3000, 30000, mid)).toBe(12000);
    expect(backoffDelay(10, 3000, 30000, mid)).toBe(30000);
  });
  it("applies at most +-20% jitter and never exceeds the cap", () => {
    expect(backoffDelay(0, 1000, 30000, () => 0)).toBe(800);
    expect(backoffDelay(0, 1000, 30000, () => 1)).toBe(1200);
    expect(backoffDelay(20, 1000, 30000, () => 1)).toBe(30000);
  });
});

describe("useWebSocket", () => {
  it("leaves exactly one live socket under StrictMode and closes it on unmount", async () => {
    const { unmount } = renderHook(() => useWebSocket({ url: "ws://x" }), { wrapper: StrictMode });
    await act(async () => { await Promise.resolve(); });
    expect(live()).toHaveLength(1);
    unmount();
    await act(async () => { await Promise.resolve(); vi.advanceTimersByTime(60000); });
    expect(live()).toHaveLength(0);
    // and no reconnect was scheduled after unmount
    expect(FakeWebSocket.instances.every((w) => w.closed)).toBe(true);
  });

  it("reports status and delivers parsed messages", async () => {
    const onMessage = vi.fn();
    const { result } = renderHook(() => useWebSocket<{ n: number }>({ url: "ws://x", onMessage }));
    const ws = FakeWebSocket.instances[0];
    act(() => ws.onopen?.());
    expect(result.current.status).toBe("connected");
    act(() => ws.onmessage?.({ data: JSON.stringify({ n: 1 }) }));
    act(() => ws.onmessage?.({ data: "not json" })); // ignored
    expect(result.current.lastMessage).toEqual({ n: 1 });
    expect(onMessage).toHaveBeenCalledTimes(1);
  });

  it("reconnects with growing delays and resets after a successful open", () => {
    const rand = vi.spyOn(Math, "random").mockReturnValue(0.5);
    renderHook(() => useWebSocket({ url: "ws://x", reconnectDelayMs: 1000, maxReconnectDelayMs: 8000 }));
    const drop = (i: number) => act(() => FakeWebSocket.instances[i].onclose?.());

    drop(0);
    act(() => vi.advanceTimersByTime(999));
    expect(FakeWebSocket.instances).toHaveLength(1);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances).toHaveLength(2); // after 1000ms

    drop(1);
    act(() => vi.advanceTimersByTime(1999));
    expect(FakeWebSocket.instances).toHaveLength(2);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances).toHaveLength(3); // after 2000ms

    act(() => FakeWebSocket.instances[2].onopen?.()); // success resets the backoff
    drop(2);
    act(() => vi.advanceTimersByTime(1000));
    expect(FakeWebSocket.instances).toHaveLength(4);
    rand.mockRestore();
  });

  it("disconnect() closes the socket and stays disconnected", async () => {
    const { result } = renderHook(() => useWebSocket({ url: "ws://x", reconnectDelayMs: 100 }));
    act(() => FakeWebSocket.instances[0].onopen?.());
    act(() => result.current.disconnect());
    await act(async () => { await Promise.resolve(); vi.advanceTimersByTime(60000); });
    expect(result.current.status).toBe("disconnected");
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(live()).toHaveLength(0);
  });

  it("does not connect when disabled", () => {
    renderHook(() => useWebSocket({ url: "ws://x", enabled: false }));
    expect(FakeWebSocket.instances).toHaveLength(0);
  });
});
