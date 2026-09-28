import React, { useState, useEffect, useRef, useMemo } from "react";
import { Link } from "react-router-dom";
import {
  Play, Pause, RotateCcw, Car, Gauge,
  Activity, AlertTriangle, Film, Columns, Eye
} from "lucide-react";

import { api } from "../api/client";
import type { TLSFrame, ComparisonEntry } from "../api/client";
import { NetworkCanvas, type VehiclePoint } from "../components/NetworkCanvas";
import { QueueTrendSparkline } from "../components/QueueTrendSparkline";
import { sampleAndHold, tlsFramesToStateMap, meanSpeedKmh, countQueued, buildQueueTrend } from "../utils/trajectory";
import { CONTROLLER_LABELS, CONTROLLER_COLORS, CONTROLLER_HEX, replayWindowStart, REPLAY_WINDOW_SECONDS } from "../constants/controllers";

const CONTROLLER_IDS = ["ppo", "ippo", "fixed_time", "max_pressure", "sumo_actuated"] as const;
const RL_CONTROLLER_IDS = CONTROLLER_IDS.filter((id) => id === "ppo" || id === "ippo");
// Split view's left-hand panel: which baseline to compare the RL controller against.
const BASELINE_CONTROLLER_IDS = CONTROLLER_IDS.filter((id) => id === "fixed_time" || id === "sumo_actuated");

export const ReplayPage: React.FC = () => {
  const [isSplitMode, setIsSplitMode] = useState(true);
  const [controller, setController] = useState("ppo");
  // Split view's right-hand panel: which RL controller to compare against the baseline.
  const [splitRlController, setSplitRlController] = useState("ppo");
  // Split view's left-hand panel: which baseline to compare the RL controller against.
  const [splitBaselineController, setSplitBaselineController] = useState("fixed_time");
  const [seed] = useState(42);
  const [demandId] = useState("1.0x");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Single-view frames
  const [frames, setFrames] = useState<Record<string, VehiclePoint[]>>({});
  const [availableTimes, setAvailableTimes] = useState<number[]>([]);

  // Split-view frames: Fixed-Time vs the selected RL controller
  const [fixedFrames, setFixedFrames] = useState<Record<string, VehiclePoint[]>>({});
  const [ppoFrames, setPpoFrames] = useState<Record<string, VehiclePoint[]>>({});

  // Real recorded TLS phase per controller, keyed the same way as the frame dicts.
  const [tlsFrames, setTlsFrames] = useState<Record<string, TLSFrame[]>>({});
  const [fixedTlsFrames, setFixedTlsFrames] = useState<Record<string, TLSFrame[]>>({});
  const [ppoTlsFrames, setPpoTlsFrames] = useState<Record<string, TLSFrame[]>>({});

  // Playback state
  const [currentTimeIndex, setCurrentTimeIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const timerRef = useRef<number | null>(null);

  // Full-episode metrics for the table under the replay — same source of
  // truth as the Comparison page (not recomputed from the visible 120s
  // window), so a number here never contradicts the one shown there.
  const [comparisonRows, setComparisonRows] = useState<ComparisonEntry[]>([]);
  useEffect(() => {
    api.getComparisonMetrics()
      .then((res) => setComparisonRows(res.results))
      .catch(() => setComparisonRows([]));
  }, []);

  // Load trajectory data
  useEffect(() => {
    let cancelled = false;
    async function loadTrajectory() {
      setLoading(true);
      setError(null);
      setIsPlaying(false);
      try {
        if (isSplitMode) {
          // Fetch both the selected baseline and the selected RL controller in
          // parallel, each at ITS OWN recorded window — most controllers were
          // regenerated at the peak window, but "ippo" still only has data at
          // the original midnight window (see replayWindowStart), so a single
          // shared window would silently return zero frames for whichever
          // side doesn't match it.
          const baselineStart = replayWindowStart(splitBaselineController);
          const rlStart = replayWindowStart(splitRlController);
          const [ftData, ppoData] = await Promise.all([
            api.getReplayFrames(splitBaselineController, seed, demandId, baselineStart, baselineStart + REPLAY_WINDOW_SECONDS),
            api.getReplayFrames(splitRlController, seed, demandId, rlStart, rlStart + REPLAY_WINDOW_SECONDS),
          ]);
          if (cancelled) return;

          const typedFt = ftData.frames as Record<string, VehiclePoint[]>;
          const typedPpo = ppoData.frames as Record<string, VehiclePoint[]>;
          setFixedFrames(typedFt);
          setPpoFrames(typedPpo);
          setFixedTlsFrames(ftData.tls_frames ?? {});
          setPpoTlsFrames(ppoData.tls_frames ?? {});

          // Get union of sorted timestamps
          const timeSet = new Set([
            ...Object.keys(typedFt).map((t) => parseFloat(t)),
            ...Object.keys(typedPpo).map((t) => parseFloat(t)),
          ]);
          const times = Array.from(timeSet).sort((a, b) => a - b);
          setAvailableTimes(times);
          setCurrentTimeIndex(0);
        } else {
          // Fetch single controller, at its own recorded window (see above).
          const windowStart = replayWindowStart(controller);
          const data = await api.getReplayFrames(controller, seed, demandId, windowStart, windowStart + REPLAY_WINDOW_SECONDS);
          if (cancelled) return;

          const typedFrames = data.frames as Record<string, VehiclePoint[]>;
          setFrames(typedFrames);
          setTlsFrames(data.tls_frames ?? {});

          const times = Object.keys(typedFrames)
            .map((t) => parseFloat(t))
            .sort((a, b) => a - b);
          setAvailableTimes(times);
          setCurrentTimeIndex(0);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load trajectory frames");
          setFrames({});
          setFixedFrames({});
          setPpoFrames({});
          setTlsFrames({});
          setFixedTlsFrames({});
          setPpoTlsFrames({});
          setAvailableTimes([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadTrajectory();
    return () => {
      cancelled = true;
    };
  }, [isSplitMode, controller, splitRlController, splitBaselineController, seed, demandId]);

  // Current simulation time (absolute — matches the raw keys in the frame
  // dicts, needed as-is for sampleAndHold lookups throughout this page).
  const currentTime = availableTimes[currentTimeIndex] ?? 0;
  // Display-only offset so the on-screen clock reads from 00:00 regardless
  // of which absolute sim-time the fetched window actually starts at.
  const displayWindowStart = availableTimes.length > 0 ? availableTimes[0] : 0;

  // Active vehicles in Single Mode
  const currentVehicles = useMemo(() => {
    if (availableTimes.length === 0) return [];
    return sampleAndHold(frames, currentTime);
  }, [frames, availableTimes, currentTime]);

  // Active vehicles in Split Mode
  const currentFixedVehicles = useMemo(() => {
    if (availableTimes.length === 0) return [];
    return sampleAndHold(fixedFrames, currentTime);
  }, [fixedFrames, availableTimes, currentTime]);

  const currentPpoVehicles = useMemo(() => {
    if (availableTimes.length === 0) return [];
    return sampleAndHold(ppoFrames, currentTime);
  }, [ppoFrames, availableTimes, currentTime]);

  // Queued-vehicle count across the whole fetched window (not just this
  // instant) — the visual answer to "does this policy's queue actually
  // come down, or does it just sit there while the average looks fine?"
  const queueTrend = useMemo(() => buildQueueTrend(frames), [frames]);
  const fixedQueueTrend = useMemo(() => buildQueueTrend(fixedFrames), [fixedFrames]);
  const ppoQueueTrend = useMemo(() => buildQueueTrend(ppoFrames), [ppoFrames]);

  // Real recorded TLS phase state per controller — not a client-side guess.
  const tlsCurrentStates = useMemo(
    () => (availableTimes.length === 0 ? {} : tlsFramesToStateMap(sampleAndHold(tlsFrames, currentTime))),
    [tlsFrames, availableTimes.length, currentTime]
  );
  const tlsFixedStates = useMemo(
    () => (availableTimes.length === 0 ? {} : tlsFramesToStateMap(sampleAndHold(fixedTlsFrames, currentTime))),
    [fixedTlsFrames, availableTimes.length, currentTime]
  );
  const tlsPpoStates = useMemo(
    () => (availableTimes.length === 0 ? {} : tlsFramesToStateMap(sampleAndHold(ppoTlsFrames, currentTime))),
    [ppoTlsFrames, availableTimes.length, currentTime]
  );

  // Single Mode Telemetry
  const avgSpeedKmh = useMemo(() => meanSpeedKmh(currentVehicles), [currentVehicles]);
  const queuedCount = useMemo(() => countQueued(currentVehicles), [currentVehicles]);

  // Split Mode Telemetry: Fixed
  const fixedSpeedKmh = useMemo(() => meanSpeedKmh(currentFixedVehicles), [currentFixedVehicles]);
  const fixedQueued = useMemo(() => countQueued(currentFixedVehicles), [currentFixedVehicles]);

  // Split Mode Telemetry: PPO
  const ppoSpeedKmh = useMemo(() => meanSpeedKmh(currentPpoVehicles), [currentPpoVehicles]);
  const ppoQueued = useMemo(() => countQueued(currentPpoVehicles), [currentPpoVehicles]);

  // Playback loop
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

  return (
    <div style={{ animation: "fade-in 0.4s ease both" }}>
      {/* Header */}
      <div style={{
        marginBottom: "1.25rem",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-end",
        flexWrap: "wrap",
        gap: "0.75rem"
      }}>
        <div>
          <h2 className="section-title">Trajectory Replay Studio</h2>
          <p className="section-subtitle">
            <Film size={13} style={{ display: "inline", verticalAlign: "middle", marginRight: 4 }} />
            Synchronized frame-by-frame vehicle playback on InTAS Ingolstadt road network
          </p>
        </div>

        {/* View Mode Toggle & Controller Selection */}
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
          <div style={{
            display: "flex",
            background: "rgba(255,255,255,0.05)",
            borderRadius: 20,
            padding: 3,
            border: "1px solid var(--border-subtle)"
          }}>
            <button
              className={`btn btn-sm ${isSplitMode ? "btn-primary" : ""}`}
              style={{
                borderRadius: 18,
                padding: "0.35rem 0.9rem",
                fontSize: "0.8rem",
                background: isSplitMode ? "var(--accent-cyan)" : "transparent",
                color: isSplitMode ? "#060910" : "var(--text-secondary)",
                fontWeight: isSplitMode ? 700 : 500,
                display: "flex",
                alignItems: "center",
                gap: 5,
              }}
              onClick={() => setIsSplitMode(true)}
            >
              <Columns size={13} /> Side-by-Side Dual View
            </button>
            <button
              className={`btn btn-sm ${!isSplitMode ? "btn-primary" : ""}`}
              style={{
                borderRadius: 18,
                padding: "0.35rem 0.9rem",
                fontSize: "0.8rem",
                background: !isSplitMode ? "var(--accent-cyan)" : "transparent",
                color: !isSplitMode ? "#060910" : "var(--text-secondary)",
                fontWeight: !isSplitMode ? 700 : 500,
                display: "flex",
                alignItems: "center",
                gap: 5,
              }}
              onClick={() => setIsSplitMode(false)}
            >
              <Eye size={13} /> Single View
            </button>
          </div>

          {!isSplitMode && (
            <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
              {CONTROLLER_IDS.map((id) => (
                <button
                  key={id}
                  className={`btn btn-sm ${controller === id ? "btn-primary" : ""}`}
                  style={{
                    borderRadius: 20,
                    padding: "0.35rem 0.8rem",
                    fontSize: "0.78rem",
                    border: controller === id ? "1px solid var(--accent-cyan)" : "1px solid var(--border-subtle)",
                    background: controller === id ? "rgba(34,211,238,0.15)" : "rgba(255,255,255,0.03)",
                    color: controller === id ? "var(--accent-cyan)" : "var(--text-secondary)",
                  }}
                  onClick={() => setController(id)}
                >
                  {CONTROLLER_LABELS[id] ?? id}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Synchronized Simulation Time Header */}
      <div className="glass-card" style={{
        padding: "0.75rem 1.25rem",
        marginBottom: "1rem",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: "0.75rem"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Activity size={18} color="var(--accent-cyan)" />
          <div>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Synchronized Simulation Clock
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.25rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
              {(currentTime - displayWindowStart).toFixed(1)} <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>sec</span>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: "0.6rem", alignItems: "center" }}>
          <span className="badge-blue" style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
            Seed: {seed}
          </span>
          <span className="badge-yellow" style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
            Demand: {demandId} (InTAS Scenario)
          </span>
          {loading && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.78rem", color: "var(--accent-cyan)", marginLeft: 8 }}>
              <span className="spinner" style={{ width: 14, height: 14 }} /> Synchronizing trajectories…
            </div>
          )}
        </div>
      </div>

      {isSplitMode && !error && (
        <div style={{
          fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "1rem",
          padding: "0.6rem 0.9rem", background: "rgba(255,255,255,0.02)",
          border: "1px solid var(--border-subtle)", borderRadius: 8,
        }}>
          <strong style={{ color: "var(--text-primary)" }}>What to actually watch:</strong> don't just compare how
          busy each side looks right now — watch the queue trend line under each map over the full {REPLAY_WINDOW_SECONDS / 60} minutes.
          A side whose queue keeps falling is clearing traffic; a side whose queue stays flat near its peak is
          backing up even if its average wait number looks good (see the <Link to="/story" style={{ color: "var(--accent-cyan)" }}>Story page</Link> for why that distinction matters here).
        </div>
      )}

      {error ? (
        <div style={{
          padding: "2.5rem",
          textAlign: "center",
          background: "rgba(239,68,68,0.06)",
          borderRadius: 8,
          border: "1px solid rgba(239,68,68,0.2)",
          marginBottom: "1rem"
        }}>
          <AlertTriangle size={28} color="var(--signal-red)" style={{ margin: "0 auto 0.5rem" }} />
          <p style={{ color: "var(--signal-red)", fontWeight: 600 }}>Could not load trajectory</p>
          <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", marginTop: "0.3rem" }}>{error}</p>
        </div>
      ) : isSplitMode ? (
        /* ── DUAL SIDE-BY-SIDE VIEW ────────────────────────────────────────── */
        <div className="replay-split-grid" style={{ marginBottom: "1.25rem" }}>
          {/* LEFT: Baseline — switchable between Fixed-Time and SUMO-Actuated */}
          <div className="glass-card" style={{ padding: "1rem", border: "1px solid rgba(245,158,11,0.25)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem", flexWrap: "wrap", gap: 6 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{
                  width: 8, height: 8, borderRadius: "50%",
                  background: CONTROLLER_COLORS[splitBaselineController] ?? "var(--signal-yellow)",
                }} />
                <span style={{
                  fontWeight: 700, fontSize: "0.95rem",
                  color: CONTROLLER_COLORS[splitBaselineController] ?? "var(--signal-yellow)",
                }}>
                  {CONTROLLER_LABELS[splitBaselineController] ?? splitBaselineController}
                </span>
              </div>
              <div role="group" aria-label="Baseline controller to compare" style={{ display: "flex", gap: 4 }}>
                {BASELINE_CONTROLLER_IDS.map((id) => (
                  <button
                    key={id}
                    className={`btn btn-sm ${splitBaselineController === id ? "btn-primary" : ""}`}
                    style={{ padding: "0.2rem 0.55rem", fontSize: "0.7rem" }}
                    onClick={() => setSplitBaselineController(id)}
                    aria-pressed={splitBaselineController === id}
                  >
                    {CONTROLLER_LABELS[id] ?? id}
                  </button>
                ))}
              </div>
            </div>

            {/* Left Telemetry */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.5rem", marginBottom: "0.75rem" }}>
              <div style={{ background: "rgba(255,255,255,0.02)", padding: "0.5rem 0.65rem", borderRadius: 6 }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
                  <Car size={11} /> Vehicles
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, marginTop: 2 }}>
                  {currentFixedVehicles.length}
                </div>
              </div>
              <div style={{ background: "rgba(255,255,255,0.02)", padding: "0.5rem 0.65rem", borderRadius: 6 }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
                  <Gauge size={11} /> Mean Speed
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, marginTop: 2, color: "var(--text-primary)" }}>
                  {fixedSpeedKmh.toFixed(1)} <span style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>km/h</span>
                </div>
              </div>
              <div style={{ background: "rgba(255,255,255,0.02)", padding: "0.5rem 0.65rem", borderRadius: 6 }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
                  <AlertTriangle size={11} color="var(--signal-red)" /> Queued
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, marginTop: 2, color: "var(--signal-red)" }}>
                  {fixedQueued}
                </div>
              </div>
            </div>

            {/* Canvas */}
            <NetworkCanvas height={390} vehicles={currentFixedVehicles} tlsStates={tlsFixedStates} currentTime={currentTime} />
            {fixedQueueTrend.length > 1 && (
              <div style={{ marginTop: "0.6rem" }}>
                <QueueTrendSparkline data={fixedQueueTrend} currentTime={currentTime} color={CONTROLLER_HEX[splitBaselineController] ?? "#f59e0b"} height={36} />
              </div>
            )}
          </div>

          {/* RIGHT: RL Agent — switchable between the centralized and parameter-shared policies */}
          <div className="glass-card" style={{ padding: "1rem", border: "1px solid rgba(34,211,238,0.3)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem", flexWrap: "wrap", gap: 6 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent-cyan)" }} />
                <span style={{ fontWeight: 700, fontSize: "0.95rem", color: "var(--accent-cyan)" }}>
                  {CONTROLLER_LABELS[splitRlController] ?? splitRlController}
                </span>
              </div>
              <div role="group" aria-label="RL controller to compare" style={{ display: "flex", gap: 4 }}>
                {RL_CONTROLLER_IDS.map((id) => (
                  <button
                    key={id}
                    className={`btn btn-sm ${splitRlController === id ? "btn-primary" : ""}`}
                    style={{ padding: "0.2rem 0.55rem", fontSize: "0.7rem" }}
                    onClick={() => setSplitRlController(id)}
                    aria-pressed={splitRlController === id}
                  >
                    {id.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* Right Telemetry */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.5rem", marginBottom: "0.75rem" }}>
              <div style={{ background: "rgba(255,255,255,0.02)", padding: "0.5rem 0.65rem", borderRadius: 6 }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
                  <Car size={11} /> Vehicles
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, marginTop: 2 }}>
                  {currentPpoVehicles.length}
                </div>
              </div>
              <div style={{ background: "rgba(255,255,255,0.02)", padding: "0.5rem 0.65rem", borderRadius: 6 }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
                  <Gauge size={11} /> Mean Speed
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, marginTop: 2, color: "var(--signal-green)" }}>
                  {ppoSpeedKmh.toFixed(1)} <span style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>km/h</span>
                </div>
              </div>
              <div style={{ background: "rgba(255,255,255,0.02)", padding: "0.5rem 0.65rem", borderRadius: 6 }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
                  <AlertTriangle size={11} color="var(--signal-green)" /> Queued
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, marginTop: 2, color: "var(--signal-green)" }}>
                  {ppoQueued}
                </div>
              </div>
            </div>

            {/* Canvas */}
            <NetworkCanvas height={390} vehicles={currentPpoVehicles} tlsStates={tlsPpoStates} currentTime={currentTime} />
            {ppoQueueTrend.length > 1 && (
              <div style={{ marginTop: "0.6rem" }}>
                <QueueTrendSparkline data={ppoQueueTrend} currentTime={currentTime} color={CONTROLLER_HEX[splitRlController] ?? "#22d3ee"} height={36} />
              </div>
            )}
          </div>
        </div>
      ) : (
        /* ── SINGLE CONTROLLER VIEW ───────────────────────────────────────── */
        <>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "0.75rem",
            marginBottom: "1.25rem"
          }}>
            <div className="glass-card" style={{ padding: "0.85rem 1.1rem" }}>
              <div className="metric-label" style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <Car size={12} color="var(--signal-green)" /> Vehicles in Network
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.3rem", fontWeight: 700, color: "var(--text-primary)", marginTop: "0.2rem" }}>
                {currentVehicles.length.toLocaleString()}
              </div>
            </div>

            <div className="glass-card" style={{ padding: "0.85rem 1.1rem" }}>
              <div className="metric-label" style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <Gauge size={12} color="var(--accent-indigo)" /> Mean Speed
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.3rem", fontWeight: 700, color: "var(--text-primary)", marginTop: "0.2rem" }}>
                {avgSpeedKmh.toFixed(1)} <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>km/h</span>
              </div>
            </div>

            <div className="glass-card" style={{ padding: "0.85rem 1.1rem" }}>
              <div className="metric-label" style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <AlertTriangle size={12} color={queuedCount > 10 ? "var(--signal-red)" : "var(--signal-green)"} /> Queued (&lt;2 m/s)
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.3rem", fontWeight: 700, color: queuedCount > 10 ? "var(--signal-red)" : "var(--signal-green)", marginTop: "0.2rem" }}>
                {queuedCount}
              </div>
            </div>
          </div>

          <div className="glass-card" style={{ padding: "1.25rem", marginBottom: "1.25rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
              <span style={{ fontWeight: 600, fontSize: "0.95rem" }}>Trajectory Canvas</span>
              <span className="badge-blue" style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem" }}>
                {controller.toUpperCase()} · seed {seed}
              </span>
            </div>
            <NetworkCanvas height={420} vehicles={currentVehicles} tlsStates={tlsCurrentStates} currentTime={currentTime} />
            {queueTrend.length > 1 && (
              <div style={{ marginTop: "0.75rem" }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: 4 }}>
                  Queued vehicles over this {REPLAY_WINDOW_SECONDS / 60}-minute window — watch for a fall, not just a low instant reading
                </div>
                <QueueTrendSparkline data={queueTrend} currentTime={currentTime} color={CONTROLLER_HEX[controller] ?? "#22d3ee"} height={40} />
              </div>
            )}
          </div>
        </>
      )}

      {/* Full-episode comparison table — grounds the animation in the real numbers */}
      {comparisonRows.length > 0 && (
        <div className="glass-card" style={{ padding: "1.25rem", marginBottom: "1.25rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "0.75rem", flexWrap: "wrap", gap: 4 }}>
            <span style={{ fontWeight: 600, fontSize: "0.95rem" }}>Comparison Metrics</span>
            <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
              Full 3600s episode (seed {seed}) — not just the {" "}
              {((availableTimes[availableTimes.length - 1] ?? 120) - displayWindowStart).toFixed(0)}s replay window above
            </span>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Controller</th>
                  <th>Avg Wait (s)</th>
                  <th>Avg Queue</th>
                  <th>Throughput</th>
                  <th>Mean Speed</th>
                  <th>Total Arrived</th>
                </tr>
              </thead>
              <tbody>
                {comparisonRows.map((r) => {
                  const isActive = isSplitMode
                    ? r.controller === splitBaselineController || r.controller === splitRlController
                    : r.controller === controller;
                  return (
                    <tr key={r.controller} style={isActive ? { background: "rgba(34,211,238,0.06)" } : undefined}>
                      <td>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <span style={{
                            width: 8, height: 8, borderRadius: "50%",
                            background: CONTROLLER_COLORS[r.controller] ?? "#64748b",
                            boxShadow: isActive ? `0 0 6px ${CONTROLLER_COLORS[r.controller] ?? "#64748b"}` : undefined,
                          }} />
                          <span style={{ fontWeight: isActive ? 700 : 500 }}>
                            {CONTROLLER_LABELS[r.controller] ?? r.controller}
                          </span>
                          {isActive && <span className="badge-blue" style={{ fontSize: "0.65rem" }}>shown above</span>}
                        </div>
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{r.mean_waiting_time.toFixed(2)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{r.mean_queue.toFixed(3)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{(r.throughput * 100).toFixed(1)}%</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{r.mean_speed.toFixed(2)} m/s</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{r.total_arrived.toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Shared Synchronized Playback Controls Toolbar */}
      <div className="glass-card" style={{
        padding: "1rem 1.25rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.75rem"
      }}>
        {/* Scrubber slider */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--text-muted)", minWidth: 40 }}>
            {(currentTime - displayWindowStart).toFixed(0)}s
          </span>
          <input
            type="range"
            min={0}
            max={Math.max(0, availableTimes.length - 1)}
            value={currentTimeIndex}
            onChange={(e) => {
              setCurrentTimeIndex(parseInt(e.target.value, 10));
            }}
            style={{
              flex: 1,
              accentColor: "var(--accent-cyan)",
              cursor: "pointer",
            }}
          />
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--text-muted)", minWidth: 40 }}>
            {((availableTimes[availableTimes.length - 1] ?? 0) - displayWindowStart).toFixed(0)}s
          </span>
        </div>

        {/* Buttons row */}
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "0.75rem"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <button
              className="btn btn-primary btn-sm"
              style={{ padding: "0.45rem 1.1rem", display: "flex", alignItems: "center", gap: 6 }}
              onClick={() => setIsPlaying(!isPlaying)}
              disabled={availableTimes.length === 0}
            >
              {isPlaying ? <><Pause size={14} /> Pause</> : <><Play size={14} /> Play Synchronized</>}
            </button>

            <button
              className="btn btn-sm"
              onClick={() => {
                setIsPlaying(false);
                setCurrentTimeIndex(0);
              }}
              disabled={availableTimes.length === 0}
              title="Reset to 0s"
            >
              <RotateCcw size={14} /> Reset
            </button>
          </div>

          {/* Speed buttons */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginRight: 4 }}>Playback Speed:</span>
            {[1, 2, 5, 10].map((s) => (
              <button
                key={s}
                className={`btn btn-sm ${playbackSpeed === s ? "btn-primary" : ""}`}
                style={{
                  padding: "0.2rem 0.5rem",
                  fontSize: "0.72rem",
                  fontFamily: "var(--font-mono)",
                  minWidth: 32,
                  background: playbackSpeed === s ? "var(--accent-cyan)" : "transparent",
                  color: playbackSpeed === s ? "#060910" : "var(--text-secondary)",
                  fontWeight: playbackSpeed === s ? 700 : 400,
                }}
                onClick={() => setPlaybackSpeed(s)}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
