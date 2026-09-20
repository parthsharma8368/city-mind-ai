import type React from "react";
import { Clock, TrendingDown, Gauge, Car } from "lucide-react";
import { msToKmh } from "../utils/trajectory";

interface MetricsPanelProps {
  avgWait: number;
  avgQueue: number;
  avgSpeed: number;
  vehiclesCount: number;
  arrivedCount?: number;
  reward?: number;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({
  avgWait,
  avgQueue,
  avgSpeed,
  vehiclesCount,
  arrivedCount = 0,
}) => {
  return (
    <div className="metric-grid">
      <div className="metric-card glass-card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span className="metric-label">Avg Waiting Time</span>
          <Clock size={16} color="var(--accent-cyan)" />
        </div>
        <div className="metric-value">{avgWait.toFixed(1)} s</div>
        <span className="metric-subtext">Per vehicle delay</span>
      </div>

      <div className="metric-card glass-card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span className="metric-label">Avg Queue Length</span>
          <TrendingDown size={16} color="var(--signal-yellow)" />
        </div>
        <div className="metric-value">{avgQueue.toFixed(1)} veh</div>
        <span className="metric-subtext">Across controlled lanes</span>
      </div>

      <div className="metric-card glass-card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span className="metric-label">Network Speed</span>
          <Gauge size={16} color="var(--signal-green)" />
        </div>
        <div className="metric-value">{msToKmh(avgSpeed).toFixed(1)} km/h</div>
        <span className="metric-subtext">Mean traffic velocity</span>
      </div>

      <div className="metric-card glass-card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span className="metric-label">Throughput / Active</span>
          <Car size={16} color="var(--accent-blue)" />
        </div>
        <div className="metric-value">
          {arrivedCount} <span style={{ fontSize: "1rem", color: "var(--text-muted)" }}>/ {vehiclesCount}</span>
        </div>
        <span className="metric-subtext">Arrived vs active trips</span>
      </div>
    </div>
  );
};
