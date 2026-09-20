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
};

export const CONTROLLER_COLORS: Record<string, string> = {
  ppo: "var(--accent-cyan)",
  ippo: "var(--accent-purple)",
  fixed_time: "var(--signal-yellow)",
  max_pressure: "var(--accent-indigo)",
};

export function controllerLabel(id: string): string {
  return CONTROLLER_LABELS[id] ?? id;
}
