import React, { useEffect, useState, useMemo, useRef } from "react";
import { Link } from "react-router-dom";
import { Activity, TrendingDown, Zap, AlertTriangle } from "lucide-react";
import { MetricsPanel } from "../components/MetricsPanel";
import { PlaybackControls } from "../components/PlaybackControls";
import { NetworkCanvas, type VehiclePoint } from "../components/NetworkCanvas";
import { TenSeedTable } from "../components/TenSeedTable";
import { QueueTrendSparkline } from "../components/QueueTrendSparkline";
import { api, isAbortError } from "../api/client";
import type { ComparisonResult, MetricsSummary, TLSFrame } from "../api/client";
import { fmt, fmtInt } from "../utils/format";
import { sampleAndHold, tlsFramesToStateMap, meanSpeedMs, countQueued, msToKmh, buildQueueTrend } from "../utils/trajectory";
import { controllerLabel, REPLAY_WINDOW_START, REPLAY_WINDOW_SECONDS, CONTROLLER_HEX } from "../constants/controllers";

export const DashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);

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

  // Fetch the headline metrics first — /api/metrics/summary picks the final model (A or B) that completes the most trips
  // over all evaluation seeds; the replay below animates that model's seed-42 run.
  useEffect(() => {
    const ctrl = new AbortController();
    api.getMetricsSummary(ctrl.signal)
      .then((m) => { if (!ctrl.signal.aborted) { setMetrics(m); setMetricsError(null); } })
      .catch((e: Error) => { if (!ctrl.signal.aborted && !isAbortError(e)) setMetricsError(e.message); });
    // Wait-time comparison for the starvation banner (best effort; the banner
    // stays neutral about wait time if this is unavailable).
    api.getComparisonMetrics(ctrl.signal)
      .then((c) => { if (!ctrl.signal.aborted) setComparison(c); })
      .catch(() => undefined);
    return () => ctrl.abort();
  }, []);

  const headlineController = metrics?.controller;

  useEffect(() => {
    if (!headlineController) return; // wait for /summary to resolve which controller "wins" first
    const windowStart = REPLAY_WINDOW_START;
    const ctrl = new AbortController();
    api.getReplayFrames(headlineController, 42, "1.0x", windowStart, windowStart + REPLAY_WINDOW_SECONDS, ctrl.signal)
      .then((data) => {
        if (ctrl.signal.aborted) return;
        setFrames(data.frames);
        setTlsFrames(data.tls_frames ?? {});
        const times = Object.keys(data.frames)
          .map((t) => parseFloat(t))
          .sort((a, b) => a - b);
        setAvailableTimes(times);
        setCurrentTimeIndex(0);
        setReplayError(null);
      })
      .catch((e: Error) => { if (!ctrl.signal.aborted && !isAbortError(e)) setReplayError(e.message); });
    return () => ctrl.abort();
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

  // Current simulation time (absolute sim-time, matches the raw keys in
  // `frames`/`tlsFrames` — needed as-is for sampleAndHold lookups below).
  const currentTime = availableTimes[currentTimeIndex] ?? 0;
  // Display-only: the fetched window's own first frame, so the on-screen
  // clock reads from 00:00 regardless of which absolute sim-time the
  // controller's trajectory actually starts at (peak-window controllers
  // start at 25200s, 07:00).
  const displayWindowStart = availableTimes.length > 0 ? availableTimes[0] : 0;

  const currentVehicles = useMemo(
    () => (availableTimes.length === 0 ? [] : sampleAndHold(frames, currentTime)),
    [frames, availableTimes.length, currentTime]
  );

  // Queued-vehicle count across the WHOLE fetched window, not just this
  // instant — this is what actually shows whether the policy is clearing
  // traffic or just holding a queue steady (see QueueTrendSparkline).
  const queueTrend = useMemo(() => buildQueueTrend(frames), [frames]);

  // Real per-TLS phase state at the current replay time, straight from the
  // controller's recorded trajectory — no client-side approximation.
  const tlsStates = useMemo(
    () => (availableTimes.length === 0 ? {} : tlsFramesToStateMap(sampleAndHold(tlsFrames, currentTime))),
    [tlsFrames, availableTimes.length, currentTime]
  );

  // Telemetry derived from active simulation vehicles
  const liveAvgSpeedMs = useMemo(() => meanSpeedMs(currentVehicles), [currentVehicles]);
  const liveQueuedCount = useMemo(() => countQueued(currentVehicles), [currentVehicles]);

  // One source of truth for episode figures: REST /metrics/summary (the same
  // seed-42 run the replay animates). The WebSocket payload is intentionally NOT
  // mixed in here -- it can aggregate other seeds and produced a different
  // "arrived" total than the Episode Statistics card. Instantaneous replay
  // values are shown separately and labelled as such.
  const hasFrame = currentVehicles.length > 0;
  const queuedNow = hasFrame ? liveQueuedCount : null;
  const vehiclesNow = hasFrame ? currentVehicles.length : null;
  const speedNowMs = hasFrame ? liveAvgSpeedMs : null;

  // Wait-time direction vs the SUMO-Actuated baseline (the same baseline the
  // summary's trips_completed_gap_pct is measured against), so the starvation
  // banner never claims a "lower wait time" the controller doesn't have.
  const waitPctVsBaseline = comparison?.improvement_vs_actuated?.controller === metrics?.controller
    ? comparison?.improvement_vs_actuated?.waiting_time_improvement_pct
    : undefined;

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

      {metrics?.starvation_flag && (
        <div
          className="connectivity-banner"
          role="alert"
          style={{
            background: "rgba(245, 158, 11, 0.1)",
            borderColor: "rgba(245, 158, 11, 0.35)",
            color: "var(--signal-yellow)",
            justifyContent: "space-between",
            flexWrap: "wrap",
          }}
        >
          <span style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
            <AlertTriangle size={16} aria-hidden="true" />
            <span>
              <strong>{controllerLabel(metrics.controller)}</strong> completes {fmt(Math.abs(metrics.trips_completed_gap_pct ?? 0), 1)}% fewer
              trips than baseline, leaving more vehicles stuck in the network.{" "}
              {waitPctVsBaseline === undefined
                ? "Read its wait time together with trips completed."
                : waitPctVsBaseline > 0
                  ? "Its lower wait time reflects fewer completed trips, not faster service."
                  : `Its wait time is also ${fmt(Math.abs(waitPctVsBaseline), 1)}% higher than baseline, so it is worse on both counts.`}
            </span>
          </span>
          <Link
            to="/comparison"
            style={{
              color: "var(--signal-yellow)",
              border: "1px solid rgba(245, 158, 11, 0.4)",
              borderRadius: "var(--radius-sm)",
              padding: "0.3rem 0.7rem",
              fontSize: "0.78rem",
              fontWeight: 600,
              whiteSpace: "nowrap",
              textDecoration: "none",
            }}
          >
            See full picture →
          </Link>
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
          <span className={metrics?.starvation_flag ? "badge-yellow" : "badge-green"}>Policy: {metrics ? controllerLabel(metrics.controller) : "…"}</span>
          <span className="badge-yellow">Demand: 1.0×</span>
          {metrics && <span className="badge-blue">{fmtInt(metrics.total_steps)} steps</span>}
        </div>
      </div>

      {/* Metrics */}
      <MetricsPanel
        loading={!metrics && !metricsError}
        avgWait={metrics?.mean_waiting_time}
        avgQueue={metrics?.mean_queue}
        avgSpeed={metrics?.mean_speed}
        arrivedCount={metrics?.total_arrived}
        queuedNow={queuedNow}
        vehiclesNow={vehiclesNow}
        speedNowMs={speedNowMs}
      />

      {/* Interactive Simulation Network Canvas */}
      <div className="glass-card" style={{ padding: "1.25rem", marginBottom: "1.25rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem", flexWrap: "wrap", gap: "0.5rem" }}>
          <div>
            <h3 style={{ fontSize: "1rem", fontWeight: 600 }}>Ingolstadt Traffic Network — Recorded Trajectory Replay</h3>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 2 }}>
              Replaying {metrics ? controllerLabel(metrics.controller) : "the best final model"}'s actual recorded run (07:00-07:05, seed 42) · vehicle positions and signal phases as they really occurred · <Link to="/replay" style={{ color: "var(--accent-cyan)" }}>watch it next to SUMO actuated →</Link>
            </p>
          </div>
          <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span className="badge-blue" style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
                Time: {(currentTime - displayWindowStart).toFixed(1)}s
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
              <p style={{ fontSize: "0.8rem" }}>Run <code>scripts/evaluate.py</code> to generate one.</p>
            </div>
          </div>
        ) : (
          <>
            <NetworkCanvas height={420} vehicles={currentVehicles} tlsStates={tlsStates} currentTime={currentTime} />

            {queueTrend.length > 1 && (
              <div style={{ marginTop: "0.85rem", padding: "0.6rem 0.85rem", background: "rgba(255,255,255,0.02)", border: "1px solid var(--border-subtle)", borderRadius: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                    Queued vehicles over this {REPLAY_WINDOW_SECONDS / 60}-minute window
                  </span>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                    watch whether this <strong style={{ color: "var(--text-secondary)" }}>falls</strong> or stays <strong style={{ color: "var(--text-secondary)" }}>flat</strong> — flat means it's not clearing
                  </span>
                </div>
                <QueueTrendSparkline
                  data={queueTrend}
                  currentTime={currentTime}
                  color={metrics ? CONTROLLER_HEX[metrics.controller] ?? "#22d3ee" : "#22d3ee"}
                />
              </div>
            )}

            <div style={{ marginTop: "1rem" }}>
              <PlaybackControls
                isPlaying={isPlaying}
                currentTime={currentTime - displayWindowStart}
                maxTime={availableTimes.length > 0 ? availableTimes[availableTimes.length - 1] - displayWindowStart : 120}
                playbackSpeed={playbackSpeed}
                onPlayToggle={() => setIsPlaying(!isPlaying)}
                onReset={() => {
                  setCurrentTimeIndex(0);
                  setIsPlaying(false);
                }}
                onSpeedChange={setPlaybackSpeed}
                onSeek={(t) => {
                  const idx = availableTimes.findIndex((timeVal) => timeVal - displayWindowStart >= t);
                  if (idx !== -1) setCurrentTimeIndex(idx);
                }}
              />
            </div>
          </>
        )}
      </div>

      {/* The paired 10-seed test against SUMO actuated */}
      <div className="glass-card" style={{ padding: "1.25rem", marginTop: "1.25rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.35rem" }}>
          <TrendingDown size={16} color="var(--signal-green)" aria-hidden="true" />
          <h3 style={{ fontSize: "0.95rem", fontWeight: 600 }}>Final models vs SUMO actuated — 10 random traffic seeds</h3>
        </div>
        <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "0.9rem" }}>
          The replay above is one run (seed 42); this table is the evidence over all seeds. Every measure is on the <Link to="/comparison" style={{ color: "var(--accent-cyan)" }}>Comparison page</Link>.
        </p>
        <TenSeedTable keys={["lag_per_scheduled_car_s", "mean_waiting_time", "mean_queue", "mean_speed", "trips_completed", "backlog", "never_inserted"]} />
      </div>

      {/* Stats row */}
      {metrics && (
        <div className="glass-card" style={{ padding: "1.25rem", marginTop: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
            <TrendingDown size={16} color="var(--accent-cyan)" aria-hidden="true" />
            <h3 style={{ fontSize: "0.95rem", fontWeight: 600 }}>{controllerLabel(metrics.controller)} · seed {metrics.seed ?? 42} · Episode Statistics</h3>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
            {[
              { label: "Total Steps", value: fmtInt(metrics.total_steps) },
              { label: "Signal Switches", value: fmtInt(metrics.total_switches) },
              { label: "Cum. Reward", value: fmt(metrics.cumulative_reward, 1) },
              { label: "Throughput", value: Number.isFinite(metrics.throughput) ? `${fmt(metrics.throughput * 100, 1)}%` : "—" },
              { label: "Mean Speed", value: Number.isFinite(metrics.mean_speed) ? `${fmt(msToKmh(metrics.mean_speed), 1)} km/h` : "—" },
              { label: "Total Arrived", value: fmtInt(metrics.total_arrived) },
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
