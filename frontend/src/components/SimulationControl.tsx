import React, { useState, useEffect } from "react";
import { Play, Square, Cpu, Monitor, RefreshCw } from "lucide-react";
import { api } from "../api/client";
import type { SimulationStatus, SimControllerId } from "../api/client";
import { useSimulationSocket } from "../context/simulationSocketStore";
import { CONTROLLER_LABELS } from "../constants/controllers";
import { fmt, fmtInt } from "../utils/format";

const CONTROLLER_IDS: SimControllerId[] = ["fixed_time", "sumo_actuated", "max_pressure", "ppo", "ippo"];

export const SimulationControl: React.FC = () => {
  const [simStatus, setSimStatus] = useState<SimulationStatus | null>(null);
  const [pending, setPending] = useState<SimControllerId | null>(null);
  const [launchMsg, setLaunchMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { status: wsStatus, lastMessage: wsPayload } = useSimulationSocket();

  const refreshStatus = () => api.getSimulationStatus().then(setSimStatus).catch(() => null);

  useEffect(() => {
    const ctrl = new AbortController();
    api.getSimulationStatus(ctrl.signal)
      .then((st) => { if (!ctrl.signal.aborted) setSimStatus(st); })
      .catch(() => null);
    return () => ctrl.abort();
  }, []);

  // running_controllers/gui_running/gui_pid come live from the websocket
  // when connected; simStatus (from the one-time REST fetch above) is only
  // needed as a fallback before the socket connects, and for the REST-only
  // fields (latest_model, xquartz_display) the websocket never carries.
  const runningControllers = wsPayload?.running_controllers ?? simStatus?.running_controllers ?? [];
  const display = simStatus?.xquartz_display ?? "N/A";

  const handleLaunch = async (controllerId: SimControllerId) => {
    setPending(controllerId);
    setError(null);
    setLaunchMsg(null);
    try {
      const res = await api.launchSimulationGui(controllerId, undefined, 80);
      setLaunchMsg(res.message);
      setTimeout(refreshStatus, 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : `Launch failed for ${controllerId}`);
    } finally {
      setPending(null);
    }
  };

  const handleStop = async (controllerId: SimControllerId) => {
    setPending(controllerId);
    setError(null);
    try {
      await api.stopSimulationGui(controllerId);
      setLaunchMsg(null);
      setTimeout(refreshStatus, 500);
    } catch (err) {
      setError(err instanceof Error ? err.message : `Stop failed for ${controllerId}`);
    } finally {
      setPending(null);
    }
  };

  return (
    <div className="glass-card" style={{ padding: "1.25rem 1.5rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
          <Monitor size={18} color="var(--accent-cyan)" aria-hidden="true" />
          <span style={{ fontWeight: 600, fontSize: "1rem" }}>SUMO-GUI Simulation Control</span>
        </div>
        <div className="ws-indicator" role="status" aria-live="polite">
          {wsStatus === "connected"
            ? <><span className="ws-dot" aria-hidden="true" />Live</>
            : <><span className="ws-dot disconnected" aria-hidden="true" />Connecting…</>
          }
        </div>
      </div>

      <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "0.9rem" }}>
        Launch more than one controller to compare live in separate XQuartz windows — same seed and
        scenario each time, so the same intersection can be watched side by side.
      </p>

      {/* Per-controller launch/stop grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.6rem", marginBottom: "1rem" }}>
        {CONTROLLER_IDS.map((id) => {
          const label = CONTROLLER_LABELS[id] ?? id;
          const isRunning = runningControllers.includes(id);
          const isPending = pending === id;
          return (
            <div key={id} style={{ background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: "0.75rem", border: "1px solid var(--border-subtle)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: "0.5rem" }}>
                <span
                  className="tls-dot"
                  style={{ background: isRunning ? "var(--signal-green)" : "var(--text-muted)" }}
                  aria-hidden="true"
                />
                <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>{label}</span>
              </div>
              <button
                className={`btn btn-sm ${isRunning ? "btn-danger" : "btn-success"}`}
                style={{ width: "100%", justifyContent: "center" }}
                onClick={() => (isRunning ? handleStop(id) : handleLaunch(id))}
                disabled={isPending}
                aria-pressed={isRunning}
                aria-busy={isPending}
                aria-label={isPending ? `${isRunning ? "Stopping" : "Launching"} ${label}…` : undefined}
              >
                {isPending
                  ? <span className="spinner" aria-hidden="true" style={{ width: 12, height: 12 }} />
                  : isRunning
                    ? <><Square size={12} aria-hidden="true" /> Stop</>
                    : <><Play size={12} aria-hidden="true" /> Launch</>
                }
              </button>
            </div>
          );
        })}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap", marginBottom: "0.75rem" }}>
        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
          XQuartz Display: <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent-cyan)" }}>{display}</span>
        </span>
        <button className="btn btn-sm" onClick={refreshStatus} title="Refresh status">
          <RefreshCw size={14} aria-hidden="true" /> Refresh
        </button>
        {runningControllers.length > 0 && (
          <span className="badge-blue">
            <Cpu size={10} aria-hidden="true" /> {runningControllers.length} window{runningControllers.length > 1 ? "s" : ""} running
          </span>
        )}
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
            Live Metrics (from saved evaluation results)
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px,1fr))", gap: "0.5rem" }}>
            {Object.entries(wsPayload.metrics).map(([ctrl, m]) => (
              <div key={ctrl} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--border-subtle)", borderRadius: 6, padding: "0.6rem 0.75rem" }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: 2, fontWeight: 600, textTransform: "uppercase" }}>{ctrl}</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "var(--accent-cyan)" }}>{fmt(m?.mean_waiting_time, 1)}s wait</div>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{fmtInt(m?.total_arrived)} arrived</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
