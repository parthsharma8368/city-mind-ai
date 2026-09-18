/**
 * SimulationSocketContext — one real WebSocket connection to /ws/simulation,
 * shared by every component that needs live status (the header's connection
 * badge and the SimulationControl panel both read from here instead of each
 * opening their own socket / faking their own status).
 */
import React, { createContext, useContext } from "react";
import { useWebSocket, type WsStatus } from "../hooks/useWebSocket";
import { WS_SIMULATION } from "../api/client";
import type { WsPayload } from "../api/client";

interface SimulationSocketValue {
  status: WsStatus;
  lastMessage: WsPayload | null;
}

const SimulationSocketContext = createContext<SimulationSocketValue>({
  status: "disconnected",
  lastMessage: null,
});

export const SimulationSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { status, lastMessage } = useWebSocket<WsPayload>({ url: WS_SIMULATION });
  return (
    <SimulationSocketContext.Provider value={{ status, lastMessage }}>
      {children}
    </SimulationSocketContext.Provider>
  );
};

export function useSimulationSocket(): SimulationSocketValue {
  return useContext(SimulationSocketContext);
}
