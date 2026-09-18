import React, { useState, useEffect } from "react";
import { Play, Square, Cpu, Monitor, RefreshCw } from "lucide-react";
import { api, WS_SIMULATION } from "../api/client";
import type { SimulationStatus, WsPayload } from "../api/client";
import { useWebSocket } from "../hooks/useWebSocket";

interface SimulationControlProps {
  onWsMessage?: (payload: WsPayload) => void;
}

export const SimulationControl: React.FC<SimulationControlProps> = ({ onWsMessage }) => {
  const [simStatus, setSimStatus] = useState<SimulationStatus | null>(null);
  const [launching, setLaunching] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [launchMsg, setLaunchMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { status: wsStatus, lastMessage: wsPayload } = useWebSocket<WsPayload>({
    url: WS_SIMULATION,
    onMessage: (data) => {
      onWsMessage?.(data);
      setSimStatus((prev) => ({
        gui_running: data.gui_running,
        gui_pid: data.gui_pid ?? null,
        latest_model: prev?.latest_model ?? null,
        xquartz_display: prev?.xquartz_display ?? null,
      }));
    },
  });

  useEffect(() => {
    api.getSimulationStatus()
      .then(setSimStatus)
      .catch(() => null);
  }, []);

  const handleLaunch = async () => {
    setLaunching(true);
    setError(null);
    setLaunchMsg(null);
    try {
      const res = await api.launchSimulationGui(undefined, 80);
      setLaunchMsg(res.message);
      setTimeout(() => api.getSimulationStatus().then(setSimStatus).catch(() => null), 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Launch failed");
    } finally {
      setLaunching(false);
    }
  };

  const handleStop = async () => {
    setStopping(true);
    setError(null);
    try {
      await api.stopSimulationGui();
      setSimStatus((prev) => prev ? { ...prev, gui_running: false, gui_pid: null } : null);
      setLaunchMsg(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Stop failed");
    } finally {
      setStopping(false);
    }
  };

  const isRunning = wsPayload?.gui_running ?? simStatus?.gui_running ?? false;
  const display = simStatus?.xquartz_display ?? "N/A";
  const modelPath = simStatus?.latest_model;
  const modelShort = modelPath ? modelPath.split("/").slice(-2).join("/") : "—";

  return (
    <div className="glass-card" style={{ padding: "1.25rem 1.5rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
          <Monitor size={18} color="var(--accent-cyan)" />
          <span style={{ fontWeight: 600, fontSize: "1rem" }}>SUMO-GUI Simulation Control</span>
        </div>
        <div className="ws-indicator">
          {wsStatus === "connected"
            ? <><span className="ws-dot" />Live</>
            : <><span className="ws-dot disconnected" />Connecting…</>
          }
        </div>
      </div>

      {/* Status Row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "0.75rem", marginBottom: "1rem" }}>
        <div style={{ background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: "0.75rem 1rem", border: "1px solid var(--border-subtle)" }}>
          <div className="metric-label">GUI Process</div>
          <div style={{ marginTop: "0.25rem", display: "flex", alignItems: "center", gap: 6 }}>
            {isRunning
              ? <><span className="tls-dot green" /><span style={{ color: "var(--signal-green)", fontWeight: 600, fontSize: "0.875rem" }}>Running</span></>
              : <><span className="tls-dot" style={{ background: "var(--text-muted)", color: "var(--text-muted)" }} /><span style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>Idle</span></>
            }
          </div>
        </div>
        <div style={{ background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: "0.75rem 1rem", border: "1px solid var(--border-subtle)" }}>
          <div className="metric-label">XQuartz Display</div>
          <div style={{ marginTop: "0.25rem", fontFamily: "var(--font-mono)", fontSize: "0.8rem", color: "var(--accent-cyan)" }}>
            {display}
          </div>
        </div>
        <div style={{ background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: "0.75rem 1rem", border: "1px solid var(--border-subtle)" }}>
          <div className="metric-label">Model</div>
          <div style={{ marginTop: "0.25rem", fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--text-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {modelShort}
          </div>
        </div>
      </div>

      {/* PID badge */}
      {isRunning && (wsPayload?.gui_pid ?? simStatus?.gui_pid) && (
        <div style={{ marginBottom: "0.75rem" }}>
          <span className="badge-blue">
            <Cpu size={10} /> PID {wsPayload?.gui_pid ?? simStatus?.gui_pid}
          </span>
        </div>
      )}

      {/* Buttons */}
      <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
        {!isRunning ? (
          <button
            className="btn btn-success btn-lg"
            onClick={handleLaunch}
            disabled={launching}
          >
            {launching ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Launching…</> : <><Play size={18} /> Launch SUMO-GUI (XQuartz)</>}
          </button>
        ) : (
          <button
            className="btn btn-danger btn-lg"
            onClick={handleStop}
            disabled={stopping}
          >
            {stopping ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Stopping…</> : <><Square size={18} /> Stop Simulation</>}
          </button>
        )}
        <button
          className="btn btn-sm"
          onClick={() => api.getSimulationStatus().then(setSimStatus).catch(() => null)}
          title="Refresh status"
        >
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* Messages */}
      {launchMsg && (
        <div style={{ marginTop: "0.75rem", padding: "0.6rem 0.9rem", background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.25)", borderRadius: 6, fontSize: "0.8rem", color: "var(--signal-green)" }}>
          ✓ {launchMsg}
        </div>
      )}
      {error && (
        <div style={{ marginTop: "0.75rem", padding: "0.6rem 0.9rem", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)", borderRadius: 6, fontSize: "0.8rem", color: "var(--signal-red)" }}>
          ✗ {error}
        </div>
      )}

      {/* Live metrics from WS */}
      {wsPayload?.metrics && Object.keys(wsPayload.metrics).length > 0 && (
        <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid var(--border-subtle)" }}>
          <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600, marginBottom: "0.5rem" }}>
            Live Metrics
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px,1fr))", gap: "0.5rem" }}>
            {Object.entries(wsPayload.metrics).map(([ctrl, m]) => (
              <div key={ctrl} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--border-subtle)", borderRadius: 6, padding: "0.6rem 0.75rem" }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: 2, fontWeight: 600, textTransform: "uppercase" }}>{ctrl}</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "var(--accent-cyan)" }}>{m.mean_waiting_time.toFixed(1)}s wait</div>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{m.total_arrived} arrived</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
