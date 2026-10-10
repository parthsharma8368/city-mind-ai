import type React from "react";
import { Clock, TrendingDown, Gauge, Car, Hourglass } from "lucide-react";
import { msToKmh } from "../utils/trajectory";
import { fmt, fmtInt, isNum } from "../utils/format";

interface MetricsPanelProps {
  /** Episode means from REST /metrics/summary. undefined = not loaded (renders "—", never a fake 0). */
  avgWait?: number | null;
  avgQueue?: number | null;
  avgSpeed?: number | null;
  /** Whole-episode arrived total (same source as the Episode Statistics card). */
  arrivedCount?: number | null;
  /** Instantaneous values at the current replay frame (not episode means). */
  queuedNow?: number | null;
  vehiclesNow?: number | null;
  speedNowMs?: number | null;
  loading?: boolean;
}

const dash = <span aria-label="no data">—</span>;

export const MetricsPanel: React.FC<MetricsPanelProps> = ({
  avgWait, avgQueue, avgSpeed, arrivedCount, queuedNow, vehiclesNow, speedNowMs, loading = false,
}) => {
  const val = (v: number | null | undefined, render: (n: number) => string, unit: string) =>
    isNum(v) ? `${render(v)}${unit}` : loading ? <span className="skeleton-text" aria-busy="true">…</span> : dash;

  return (
    <div className="metric-grid">
      <div className="metric-card glass-card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span className="metric-label">Avg Waiting Time</span>
          <Clock size={16} color="var(--accent-cyan)" aria-hidden="true" />
        </div>
        <div className="metric-value">{val(avgWait, (n) => fmt(n, 1), " s")}</div>
        <span className="metric-subtext">Episode mean per vehicle (read with trips completed)</span>
      </div>

      <div className="metric-card glass-card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span className="metric-label">Avg Queue Length</span>
          <TrendingDown size={16} color="var(--signal-yellow)" aria-hidden="true" />
        </div>
        <div className="metric-value">{val(avgQueue, (n) => fmt(n, 2), " veh")}</div>
        <span className="metric-subtext">Episode mean, across controlled lanes</span>
      </div>

      <div className="metric-card glass-card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span className="metric-label">Network Speed</span>
          <Gauge size={16} color="var(--signal-green)" aria-hidden="true" />
        </div>
        <div className="metric-value">{val(avgSpeed, (n) => fmt(msToKmh(n), 1), " km/h")}</div>
        <span className="metric-subtext">
          Episode mean{isNum(speedNowMs) ? ` · ${fmt(msToKmh(speedNowMs), 1)} km/h at this frame` : ""}
        </span>
      </div>

      <div className="metric-card glass-card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span className="metric-label">Episode Arrived</span>
          <Car size={16} color="var(--accent-blue)" aria-hidden="true" />
        </div>
        <div className="metric-value">{val(arrivedCount, (n) => fmtInt(n), "")}</div>
        <span className="metric-subtext">Trips completed over the whole episode</span>
      </div>

      <div className="metric-card glass-card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span className="metric-label">Queued now (&lt;2 m/s)</span>
          <Hourglass size={16} color="var(--signal-red)" aria-hidden="true" />
        </div>
        <div className="metric-value">
          {val(queuedNow, (n) => fmtInt(n), "")}
          {isNum(vehiclesNow) && isNum(queuedNow) && (
            <span style={{ fontSize: "1rem", color: "var(--text-muted)" }}> / {fmtInt(vehiclesNow)}</span>
          )}
        </div>
        <span className="metric-subtext">Slow vehicles / vehicles in network at this replay frame</span>
      </div>
    </div>
  );
};
