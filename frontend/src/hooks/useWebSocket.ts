/**
 * useWebSocket — Reusable WebSocket hook with auto-reconnect.
 */
import { useEffect, useRef, useState, useCallback } from "react";

export type WsStatus = "connecting" | "connected" | "disconnected";

interface UseWebSocketOptions<T> {
  url: string;
  onMessage?: (data: T) => void;
  reconnectDelayMs?: number;
  enabled?: boolean;
}

export function useWebSocket<T = unknown>({
  url,
  onMessage,
  reconnectDelayMs = 3000,
  enabled = true,
}: UseWebSocketOptions<T>) {
  const [status, setStatus] = useState<WsStatus>("disconnected");
  const [lastMessage, setLastMessage] = useState<T | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current || !enabled) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (mountedRef.current) setStatus("connected");
    };

    ws.onmessage = (evt) => {
      if (!mountedRef.current) return;
      try {
        const data = JSON.parse(evt.data) as T;
        setLastMessage(data);
        onMessage?.(data);
      } catch {
        // ignore parse errors
      }
    };

    ws.onerror = () => {
      if (mountedRef.current) setStatus("disconnected");
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setStatus("disconnected");
      wsRef.current = null;
      // Auto-reconnect
      reconnectTimeout.current = setTimeout(connect, reconnectDelayMs);
    };
  }, [url, onMessage, reconnectDelayMs, enabled]);

  useEffect(() => {
    mountedRef.current = true;
    if (enabled) connect();
    return () => {
      mountedRef.current = false;
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
      wsRef.current?.close();
    };
  }, [connect, enabled]);

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
    wsRef.current?.close();
    setStatus("disconnected");
  }, []);

  return { status, lastMessage, disconnect };
}
