/**
 * Non-component half of SimulationSocketContext, split out so that file can
 * export only the Provider component (oxlint's react/only-export-components
 * — a file mixing component and non-component exports breaks Fast Refresh).
 */
import { createContext, useContext } from "react";
import type { WsStatus } from "../hooks/useWebSocket";
import type { WsPayload } from "../api/client";

export interface SimulationSocketValue {
  status: WsStatus;
  lastMessage: WsPayload | null;
}

export const SimulationSocketContext = createContext<SimulationSocketValue>({
  status: "disconnected",
  lastMessage: null,
});

export function useSimulationSocket(): SimulationSocketValue {
  return useContext(SimulationSocketContext);
}
