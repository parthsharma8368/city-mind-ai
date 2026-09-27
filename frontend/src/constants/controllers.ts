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
  ippo_phase_select: "IPPO Phase-Select (training)",
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
