import React, { useEffect, useState, useMemo, useRef } from "react";
import { Activity, TrendingDown, Zap, AlertTriangle } from "lucide-react";
import { MetricsPanel } from "../components/MetricsPanel";
import { PlaybackControls } from "../components/PlaybackControls";
import { NetworkCanvas, type VehiclePoint } from "../components/NetworkCanvas";
import { SimulationControl } from "../components/SimulationControl";
import { api } from "../api/client";
import type { MetricsSummary, TLSFrame } from "../api/client";
import { useSimulationSocket } from "../context/simulationSocketStore";
import { sampleAndHold, tlsFramesToStateMap, meanSpeedMs, countQueued, msToKmh } from "../utils/trajectory";
import { controllerLabel } from "../constants/controllers";

export const DashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const { lastMessage: wsPayload } = useSimulationSocket();

  // Trajectory playback state — this is the RL policy's *actual* recorded
  // run (vehicle positions and real signal phases), not a live feed: training
  // and evaluation are offline batch jobs, so "real-time" here means "true to
  // what really happened," not "streaming while it happens."
  const [frames, setFrames] = useState<Record<string, VehiclePoint[]>>({});
  const [tlsFrames, setTlsFrames] = useState<Record<string, TLSFrame[]>>({});
  const [availableTimes, setAvailableTimes] = useState<number[]>([]);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [currentTimeIndex, setCurrentTimeIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const timerRef = useRef<number | null>(null);

  // Fetch the headline metrics first — /api/metrics/summary picks whichever
  // RL controller (ppo or ippo) currently has the lower waiting time, so the
  // Dashboard doesn't keep showing a stale "ppo" result once ippo overtakes
  // it. The replay below follows whatever that resolves to.
  useEffect(() => {
    api.getMetricsSummary()
      .then((m) => { setMetrics(m); setMetricsError(null); })
      .catch((e: Error) => setMetricsError(e.message));
  }, []);

  const headlineController = metrics?.controller;

  useEffect(() => {
    if (!headlineController) return; // wait for /summary to resolve which controller "wins" first
    api.getReplayFrames(headlineController, 42, "1.0x", 0, 120)
      .then((data) => {
        setFrames(data.frames);
        setTlsFrames(data.tls_frames ?? {});
        const times = Object.keys(data.frames)
          .map((t) => parseFloat(t))
          .sort((a, b) => a - b);
        setAvailableTimes(times);
        setReplayError(null);
      })
      .catch((e: Error) => setReplayError(e.message));
  }, [headlineController]);

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

  const currentVehicles = useMemo(
    () => (availableTimes.length === 0 ? [] : sampleAndHold(frames, currentTime)),
    [frames, availableTimes.length, currentTime]
  );

  // Real per-TLS phase state at the current replay time, straight from the
  // controller's recorded trajectory — no client-side approximation.
  const tlsStates = useMemo(
    () => (availableTimes.length === 0 ? {} : tlsFramesToStateMap(sampleAndHold(tlsFrames, currentTime))),
    [tlsFrames, availableTimes.length, currentTime]
  );

  // Telemetry derived from active simulation vehicles
  const liveAvgSpeedMs = useMemo(() => meanSpeedMs(currentVehicles), [currentVehicles]);
  const liveQueuedCount = useMemo(() => countQueued(currentVehicles), [currentVehicles]);

  // Determine display values — follow whichever controller /summary picked as the headline.
  const headlineWsMetrics = headlineController ? wsPayload?.metrics?.[headlineController] : undefined;
  const displayMetrics = {
    avgWait: headlineWsMetrics ? headlineWsMetrics.mean_waiting_time : (metrics?.mean_waiting_time ?? 0),
    avgQueue: currentVehicles.length > 0 ? liveQueuedCount : (headlineWsMetrics?.mean_queue ?? metrics?.mean_queue ?? 0),
    avgSpeed: currentVehicles.length > 0 ? liveAvgSpeedMs : (headlineWsMetrics?.mean_speed ?? metrics?.mean_speed ?? 0),
    vehiclesCount: currentVehicles.length > 0 ? currentVehicles.length : (headlineWsMetrics?.total_arrived ?? metrics?.total_arrived ?? 0),
    arrivedCount: headlineWsMetrics?.total_arrived ?? metrics?.total_arrived ?? 0,
  };

  // The replay fetch only ever runs once /summary has already succeeded (see
  // the effect above), so metricsError and replayError can never both be set
  // at once — "can't reach the API at all" is exactly what a failed /summary
  // means. A replay-only failure gets its own inline message below instead.
  const backendUnreachable = Boolean(metricsError);

  return (
    <div style={{ animation: "fade-in 0.4s ease both" }}>
      {backendUnreachable && (
        <div className="connectivity-banner" role="alert">
          <AlertTriangle size={16} aria-hidden="true" />
          <span>
            Can't reach the CityMind API — is <code>uvicorn api.main:app</code> running?
            Showing whatever was last loaded.
          </span>
        </div>
      )}

      {/* Header */}
      <div style={{ marginBottom: "1.25rem", display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <h2 className="section-title">Network Status Overview</h2>
          <p className="section-subtitle">
            <Activity size={13} style={{ display: "inline", verticalAlign: "middle", marginRight: 4 }} aria-hidden="true" />
            Active Scenario: InTAS Ingolstadt · 98 Signalized Intersections
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
          <span className="badge-green">Policy: {metrics ? controllerLabel(metrics.controller) : "…"}</span>
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
            <h3 style={{ fontSize: "1rem", fontWeight: 600 }}>Ingolstadt Traffic Network — Recorded Trajectory Replay</h3>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 2 }}>
              Replaying {metrics ? controllerLabel(metrics.controller) : "the headline policy"}'s actual recorded run · vehicle positions and signal phases as they really occurred
            </p>
          </div>
          <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span className="badge-blue" style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
                Time: {currentTime.toFixed(1)}s
              </span>
              <span className="badge-green">
                <Zap size={10} aria-hidden="true" /> Recorded Run
              </span>
            </div>
          </div>
        </div>

        {replayError && availableTimes.length === 0 ? (
          <div className="network-canvas-wrapper" style={{ height: 420, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div style={{ textAlign: "center", color: "var(--text-muted)" }}>
              <p style={{ color: "var(--signal-red)", marginBottom: "0.5rem" }}>⚠ No recorded trajectory available</p>
              <p style={{ fontSize: "0.8rem" }}>Run <code>scripts/run_evaluation.py</code> to generate one.</p>
            </div>
          </div>
        ) : (
          <>
            <NetworkCanvas height={420} vehicles={currentVehicles} tlsStates={tlsStates} currentTime={currentTime} />
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
          </>
        )}
      </div>

      {/* Simulation Control (Desktop SUMO-GUI) */}
      <SimulationControl />

      {/* Stats row */}
      {metrics && (
        <div className="glass-card" style={{ padding: "1.25rem", marginTop: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
            <TrendingDown size={16} color="var(--accent-cyan)" aria-hidden="true" />
            <h3 style={{ fontSize: "0.95rem", fontWeight: 600 }}>{controllerLabel(metrics.controller)} Episode Statistics</h3>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
            {[
              { label: "Total Steps", value: metrics.total_steps.toLocaleString() },
              { label: "Signal Switches", value: metrics.total_switches.toLocaleString() },
              { label: "Cum. Reward", value: metrics.cumulative_reward.toFixed(1) },
              { label: "Throughput", value: `${(metrics.throughput * 100).toFixed(1)}%` },
              { label: "Mean Speed", value: `${msToKmh(metrics.mean_speed).toFixed(1)} km/h` },
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
