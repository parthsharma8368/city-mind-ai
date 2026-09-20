/**
 * SimulationSocketContext — one real WebSocket connection to /ws/simulation,
 * shared by every component that needs live status (the header's connection
 * badge and the SimulationControl panel both read from here instead of each
 * opening their own socket / faking their own status).
 */
import React from "react";
import { useWebSocket } from "../hooks/useWebSocket";
import { WS_SIMULATION } from "../api/client";
import type { WsPayload } from "../api/client";
import { SimulationSocketContext } from "./simulationSocketStore";

export const SimulationSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { status, lastMessage } = useWebSocket<WsPayload>({ url: WS_SIMULATION });
  return (
    <SimulationSocketContext.Provider value={{ status, lastMessage }}>
      {children}
    </SimulationSocketContext.Provider>
  );
};
