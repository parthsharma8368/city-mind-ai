/**
 * Shared controller display metadata. Single source of truth so labels/colors
 * can't drift between the Dashboard, Comparison, and Replay pages.
 *
 * Ids are the result folders of the evaluation (api/results.py): the SUMO-actuated baseline, a fixed-time plan for context
 * and the two independently trained final models.
 */
export const CONTROLLER_LABELS: Record<string, string> = {
  ippo_A: "Model A (final)",
  ippo_B: "Model B (final)",
  sumo_actuated: "SUMO actuated (baseline)",
  fixed_time: "Fixed time",
};

/** Short axis labels so the bars fit on a phone-width chart; the tooltip shows the full label. */
export const CONTROLLER_SHORT_LABELS: Record<string, string> = {
  ippo_A: "Model A",
  ippo_B: "Model B",
  sumo_actuated: "Actuated",
  fixed_time: "Fixed",
};

export const CONTROLLER_COLORS: Record<string, string> = {
  ippo_A: "var(--accent-cyan)",
  ippo_B: "var(--accent-purple)",
  sumo_actuated: "var(--signal-green)",
  fixed_time: "var(--signal-yellow)",
};

// Resolved hex twins of CONTROLLER_COLORS (must match styles/index.css's
// --accent-cyan/--accent-purple/--signal-green/--signal-yellow exactly).
// SVG `fill` attributes — e.g. recharts' <Cell fill>  — don't reliably
// resolve CSS custom properties the way DOM `style` does, so chart code
// needs a resolved value instead of the var(--...) token.
export const CONTROLLER_HEX: Record<string, string> = {
  ippo_A: "#22d3ee",
  ippo_B: "#a855f7",
  sumo_actuated: "#22c55e",
  fixed_time: "#f59e0b",
};

export function controllerLabel(id: string): string {
  return CONTROLLER_LABELS[id] ?? id;
}

export function controllerShortLabel(id: string): string {
  return CONTROLLER_SHORT_LABELS[id] ?? id;
}

/** Every recorded run covers the 07:00-08:00 peak: absolute sim-time (s) at which a replay starts. */
export const REPLAY_WINDOW_START = 25200;

// How much of the recorded episode to fetch for a replay. 120s (one or two
// signal cycles) isn't enough to see whether a policy is actually clearing
// a queue or just holding it steady — 300s (5 min) shows several full
// cycles' worth of buildup/clearing, which is what the comparison is
// actually about. Checked against real recorded data before picking this:
// at peak, a 300s fetch is on the order of a few thousand rows per
// timestamp x ~60 timestamps — still a sub-second local fetch, not the
// 10x-larger jump a naive "just fetch the whole episode" would be.
export const REPLAY_WINDOW_SECONDS = 300;
