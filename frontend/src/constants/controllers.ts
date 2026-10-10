/**
 * Shared controller display metadata. Single source of truth so labels/colors
 * can't drift between the Dashboard, Comparison, and Replay pages the way
 * three separate copies of this map already had started to.
 */
export const CONTROLLER_LABELS: Record<string, string> = {
  ppo: "PPO (Centralized)",
  ippo: "IPPO (Parameter-Shared)",
  fixed_time: "Fixed-Time",
  max_pressure: "Max-Pressure",
  sumo_actuated: "SUMO-Actuated (Calibrated)",
  ippo_phase_select: "IPPO Phase-Select",
};

/** Short axis labels so six bars fit on a phone-width chart; the tooltip shows the full label. */
export const CONTROLLER_SHORT_LABELS: Record<string, string> = {
  ppo: "PPO",
  ippo: "IPPO",
  fixed_time: "Fixed",
  max_pressure: "Max-Pr.",
  sumo_actuated: "Actuated",
  ippo_phase_select: "Phase-Sel.",
};

export const CONTROLLER_COLORS: Record<string, string> = {
  ppo: "var(--accent-cyan)",
  ippo: "var(--accent-purple)",
  fixed_time: "var(--signal-yellow)",
  max_pressure: "var(--accent-indigo)",
  sumo_actuated: "var(--signal-green)",
  ippo_phase_select: "var(--text-muted)",
};

// Resolved hex twins of CONTROLLER_COLORS (must match styles/index.css's
// --accent-cyan/--accent-purple/--signal-yellow/--accent-indigo exactly).
// SVG `fill` attributes — e.g. recharts' <Cell fill>  — don't reliably
// resolve CSS custom properties the way DOM `style` does, so chart code
// needs a resolved value instead of the var(--...) token.
export const CONTROLLER_HEX: Record<string, string> = {
  ppo: "#22d3ee",
  ippo: "#a855f7",
  fixed_time: "#f59e0b",
  max_pressure: "#818cf8",
  sumo_actuated: "#22c55e",
  ippo_phase_select: "#64748b",
};

export function controllerLabel(id: string): string {
  return CONTROLLER_LABELS[id] ?? id;
}

export function controllerShortLabel(id: string): string {
  return CONTROLLER_SHORT_LABELS[id] ?? id;
}

// Absolute sim-time (seconds) each controller's recorded trajectory parquet
// actually starts at. Every controller except "ippo" was regenerated at the
// 07:00-08:00 peak window (25200 = 07:00:00) once the mechanics/metrics fix
// moved evaluation there; "ippo" (the original binary-action model) can no
// longer be replayed at all under the current DecentralizedSumoEnv (it now
// expects an 18-dim phase-selection observation, not that model's 12-dim
// one), so its trajectory file is still the original midnight-window
// recording and must be requested at time 0, not 25200 — requesting the
// wrong window for either case returns zero frames and silently breaks
// playback (no error, just nothing to animate).
const PEAK_WINDOW_START = 25200;
const MIDNIGHT_WINDOW_START = 0;

export function replayWindowStart(controllerId: string): number {
  return controllerId === "ippo" ? MIDNIGHT_WINDOW_START : PEAK_WINDOW_START;
}

// How much of the recorded episode to fetch for a replay. 120s (one or two
// signal cycles) isn't enough to see whether a policy is actually clearing
// a queue or just holding it steady — 300s (5 min) shows several full
// cycles' worth of buildup/clearing, which is what the comparison is
// actually about. Checked against real recorded data before picking this:
// at peak, a 300s fetch is on the order of a few thousand rows per
// timestamp x ~60 timestamps — still a sub-second local fetch, not the
// 10x-larger jump a naive "just fetch the whole episode" would be.
export const REPLAY_WINDOW_SECONDS = 300;

/**
 * Controllers whose recorded trajectory is NOT at the current peak window and
 * cannot be regenerated (the legacy binary-action IPPO model expects a 12-dim
 * observation the current env no longer produces). They must never be paired
 * side by side with peak-window controllers: the two recordings cover disjoint
 * sim-time ranges and any merged timeline is meaningless.
 */
export function isPeakAligned(controllerId: string): boolean {
  return replayWindowStart(controllerId) === PEAK_WINDOW_START;
}

export const LEGACY_REPLAY_NOTICE =
  "Not replayable under the current environment: this model was recorded at midnight with the old observation space, " +
  "so it is not comparable with the peak-hour (07:00) recordings of the other controllers. It is shown on its own only.";
