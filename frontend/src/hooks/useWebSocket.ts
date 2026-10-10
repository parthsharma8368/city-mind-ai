/**
 * useWebSocket — Reusable WebSocket hook with auto-reconnect (exponential
 * backoff + jitter). Each effect run owns exactly one socket and a
 * `closedByUs` flag, so React StrictMode's mount/unmount/mount cycle, a URL
 * change, or an explicit disconnect() can never leave a duplicate socket or a
 * pending reconnect behind.
 */
import { useEffect, useRef, useState, useCallback } from "react";

export type WsStatus = "connecting" | "connected" | "disconnected";

interface UseWebSocketOptions<T> {
  url: string;
  onMessage?: (data: T) => void;
  /** Base reconnect delay; doubles per failed attempt up to maxReconnectDelayMs. */
  reconnectDelayMs?: number;
  maxReconnectDelayMs?: number;
  enabled?: boolean;
}

/** Exponential backoff with +-20% jitter. `random` is injectable for tests. */
export function backoffDelay(attempt: number, baseMs: number, maxMs: number, random: () => number = Math.random): number {
  const exp = Math.min(maxMs, baseMs * 2 ** Math.max(0, attempt));
  const jitter = 1 + (random() * 0.4 - 0.2);
  return Math.min(maxMs, Math.round(exp * jitter));
}

export function useWebSocket<T = unknown>({
  url,
  onMessage,
  reconnectDelayMs = 3000,
  maxReconnectDelayMs = 30000,
  enabled = true,
}: UseWebSocketOptions<T>) {
  const [status, setStatus] = useState<WsStatus>("disconnected");
  const [lastMessage, setLastMessage] = useState<T | null>(null);
  const onMessageRef = useRef(onMessage);
  const closeRef = useRef<() => void>(() => {});

  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    if (!enabled) return;

    let closedByUs = false;
    let attempt = 0;
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (closedByUs) return;
      setStatus("connecting");
      const socket = new WebSocket(url);
      ws = socket;

      socket.onopen = () => {
        if (closedByUs || ws !== socket) return;
        attempt = 0;
        setStatus("connected");
      };

      socket.onmessage = (evt) => {
        if (closedByUs || ws !== socket) return;
        try {
          const data = JSON.parse(evt.data) as T;
          setLastMessage(data);
          onMessageRef.current?.(data);
        } catch {
          // ignore malformed frames
        }
      };

      // onerror is always followed by onclose, which owns the reconnect.
      socket.onerror = () => {};

      socket.onclose = () => {
        // A stale socket (superseded or closed by us) must never touch state or reconnect.
        if (closedByUs || ws !== socket) return;
        ws = null;
        setStatus("disconnected");
        timer = setTimeout(connect, backoffDelay(attempt, reconnectDelayMs, maxReconnectDelayMs));
        attempt += 1;
      };
    };

    const close = () => {
      closedByUs = true;
      if (timer) clearTimeout(timer);
      timer = null;
      const socket = ws;
      ws = null;
      socket?.close();
    };
    closeRef.current = close;

    connect();
    return () => {
      close();
    };
  }, [url, enabled, reconnectDelayMs, maxReconnectDelayMs]);

  /** Closes the socket and stays closed (no reconnect) until the URL / enabled changes. */
  const disconnect = useCallback(() => {
    closeRef.current();
    setStatus("disconnected");
  }, []);

  return { status, lastMessage, disconnect };
}
