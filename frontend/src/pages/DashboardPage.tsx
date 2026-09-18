import React, { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { Activity, TrendingDown, Zap } from "lucide-react";
import { MetricsPanel } from "../components/MetricsPanel";
import { PlaybackControls } from "../components/PlaybackControls";
import { NetworkCanvas, type VehiclePoint } from "../components/NetworkCanvas";
import { SimulationControl } from "../components/SimulationControl";
import { api } from "../api/client";
import type { MetricsSummary, WsPayload } from "../api/client";

export const DashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [wsMetrics, setWsMetrics] = useState<WsPayload["metrics"] | null>(null);

  // Trajectory playback state
  const [frames, setFrames] = useState<Record<string, VehiclePoint[]>>({});
  const [availableTimes, setAvailableTimes] = useState<number[]>([]);
  const [currentTimeIndex, setCurrentTimeIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const timerRef = useRef<number | null>(null);

  // Real TLS Signal specifications from InTAS
  const [tlsSpecs, setTlsSpecs] = useState<Array<{
    tls_id: string;
    phases: Array<{ state: string; duration: number }>;
  }>>([]);

  // Fetch metrics summary & TLS metadata on mount
  useEffect(() => {
    api.getMetricsSummary()
      .then(setMetrics)
      .catch(() => null);

    api.getTLSMetadata()
      .then(setTlsSpecs)
      .catch(() => null);

    // Load PPO trajectory frames (0 to 120s)
    api.getReplayFrames("ppo", 42, "1.0x", 0, 120)
      .then((data) => {
        const typed = data.frames as Record<string, VehiclePoint[]>;
        setFrames(typed);
        const times = Object.keys(typed)
          .map((t) => parseFloat(t))
          .sort((a, b) => a - b);
        setAvailableTimes(times);
      })
      .catch(() => null);
  }, []);

  const handleWsMessage = useCallback((payload: WsPayload) => {
    if (payload.metrics && Object.keys(payload.metrics).length > 0) {
      setWsMetrics(payload.metrics);
    }
  }, []);

  // Playback timer loop
  useEffect(() => {
    if (!isPlaying) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }

    const intervalMs = Math.max(100, Math.round(1000 / playbackSpeed));
    timerRef.current = window.setInterval(() => {
      setCurrentTimeIndex((prev) => {
        if (prev >= availableTimes.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, playbackSpeed, availableTimes.length]);

  // Current simulation time
  const currentTime = availableTimes[currentTimeIndex] ?? 0;

  // Active vehicles at currentTime
  const currentVehicles = useMemo(() => {
    if (availableTimes.length === 0) return [];
    const tExact = currentTime.toFixed(1);
    if (frames[tExact]) return frames[tExact];
    if (frames[String(currentTime)]) return frames[String(currentTime)];

    // Fallback sample-and-hold
    const keys = Object.keys(frames).map(Number).sort((a, b) => a - b);
    let bestKey: number | null = null;
    for (const k of keys) {
      if (k <= currentTime) bestKey = k;
      else break;
    }
    return bestKey !== null ? (frames[bestKey.toFixed(1)] || frames[String(bestKey)] || []) : [];
  }, [frames, availableTimes, currentTime]);

  // Telemetry derived from active simulation vehicles
  const liveAvgSpeedKmh = useMemo(() => {
    if (currentVehicles.length === 0) return 0;
    const sum = currentVehicles.reduce((acc, v) => acc + (v.speed || 0), 0);
    return (sum / currentVehicles.length) * 3.6;
  }, [currentVehicles]);

  const liveQueuedCount = useMemo(() => {
    return currentVehicles.filter((v) => (v.speed || 0) < 2).length;
  }, [currentVehicles]);

  // Dynamic real TLS states
  const tlsStates = useMemo(() => {
    if (!tlsSpecs || tlsSpecs.length === 0) return {};
    const res: Record<string, { phase: number; state: string }> = {};
    for (const spec of tlsSpecs) {
      if (!spec.phases || spec.phases.length === 0) continue;
      const totalCycle = spec.phases.reduce((acc, p) => acc + (p.duration || 30), 0) || 90;
      const seedNum = parseInt(spec.tls_id.replace(/\D/g, "").slice(-3) || "0", 10);
      const adjustedTime = (currentTime * 1.08 + (seedNum % 20)) % totalCycle;
      let remainder = adjustedTime;
      let phaseIdx = 0;
      for (let i = 0; i < spec.phases.length; i++) {
        const dur = spec.phases[i].duration || 30;
        if (remainder < dur) {
          phaseIdx = i;
          break;
        }
        remainder -= dur;
      }
      res[spec.tls_id] = {
        phase: phaseIdx,
        state: spec.phases[phaseIdx]?.state || "GGGrrr",
      };
    }
    return res;
  }, [tlsSpecs, currentTime]);

  // Determine display values
  const ppoMetrics = wsMetrics?.ppo;
  const displayMetrics = {
    avgWait: ppoMetrics ? ppoMetrics.mean_waiting_time : (metrics?.mean_waiting_time ?? 0),
    avgQueue: currentVehicles.length > 0 ? liveQueuedCount : (ppoMetrics?.mean_queue ?? metrics?.mean_queue ?? 0),
    avgSpeed: currentVehicles.length > 0 ? (liveAvgSpeedKmh / 3.6) : (ppoMetrics?.mean_speed ?? metrics?.mean_speed ?? 0),
    vehiclesCount: currentVehicles.length > 0 ? currentVehicles.length : (ppoMetrics?.total_arrived ?? metrics?.total_arrived ?? 0),
    arrivedCount: ppoMetrics?.total_arrived ?? metrics?.total_arrived ?? 0,
  };

  return (
    <div style={{ animation: "fade-in 0.4s ease both" }}>
      {/* Header */}
      <div style={{ marginBottom: "1.25rem", display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <h2 className="section-title">Network Status Overview</h2>
          <p className="section-subtitle">
            <Activity size={13} style={{ display: "inline", verticalAlign: "middle", marginRight: 4 }} />
            Active Scenario: InTAS Ingolstadt · 98 Signalized Intersections
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
          <span className="badge-green">Policy: Centralized PPO</span>
          <span className="badge-yellow">Demand: 1.0×</span>
          {metrics && <span className="badge-blue">{metrics.total_steps.toLocaleString()} steps</span>}
        </div>
      </div>

      {/* Metrics */}
      <MetricsPanel
        avgWait={displayMetrics.avgWait}
        avgQueue={displayMetrics.avgQueue}
        avgSpeed={displayMetrics.avgSpeed}
        vehiclesCount={displayMetrics.vehiclesCount}
        arrivedCount={displayMetrics.arrivedCount}
      />

      {/* Interactive Simulation Network Canvas */}
      <div className="glass-card" style={{ padding: "1.25rem", marginBottom: "1.25rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem", flexWrap: "wrap", gap: "0.5rem" }}>
          <div>
            <h3 style={{ fontSize: "1rem", fontWeight: 600 }}>Ingolstadt Traffic Network — Live Trajectory Simulation</h3>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 2 }}>
              Active RL Policy controlling 98 signals · Moving vehicle telemetry & signal cycles
            </p>
          </div>
          <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span className="badge-blue" style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
                Time: {currentTime.toFixed(1)}s
              </span>
              <span className="badge-green">
                <Zap size={10} /> RL Active
              </span>
            </div>
          </div>
        </div>

        {/* Network Canvas with Real Vehicles and Real Signal States */}
        <NetworkCanvas height={420} vehicles={currentVehicles} tlsStates={tlsStates} />

        {/* Playback Controls */}
        <div style={{ marginTop: "1rem" }}>
          <PlaybackControls
            isPlaying={isPlaying}
            currentTime={currentTime}
            maxTime={availableTimes.length > 0 ? availableTimes[availableTimes.length - 1] : 120}
            playbackSpeed={playbackSpeed}
            onPlayToggle={() => setIsPlaying(!isPlaying)}
            onReset={() => {
              setCurrentTimeIndex(0);
              setIsPlaying(false);
            }}
            onSpeedChange={setPlaybackSpeed}
            onSeek={(t) => {
              const idx = availableTimes.findIndex((timeVal) => timeVal >= t);
              if (idx !== -1) setCurrentTimeIndex(idx);
            }}
          />
        </div>
      </div>

      {/* Simulation Control (Desktop SUMO-GUI) */}
      <SimulationControl onWsMessage={handleWsMessage} />

      {/* Stats row */}
      {metrics && (
        <div className="glass-card" style={{ padding: "1.25rem", marginTop: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
            <TrendingDown size={16} color="var(--accent-cyan)" />
            <h3 style={{ fontSize: "0.95rem", fontWeight: 600 }}>PPO Episode Statistics</h3>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
            {[
              { label: "Total Steps", value: metrics.total_steps.toLocaleString() },
              { label: "Signal Switches", value: metrics.total_switches.toLocaleString() },
              { label: "Cum. Reward", value: metrics.cumulative_reward.toFixed(1) },
              { label: "Throughput", value: `${(metrics.throughput * 100).toFixed(1)}%` },
              { label: "Mean Speed", value: `${(metrics.mean_speed * 3.6).toFixed(1)} km/h` },
              { label: "Total Arrived", value: metrics.total_arrived.toLocaleString() },
            ].map(({ label, value }) => (
              <div key={label} style={{ background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: "0.75rem 1rem", border: "1px solid var(--border-subtle)" }}>
                <div className="metric-label">{label}</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)", marginTop: "0.2rem" }}>{value}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
