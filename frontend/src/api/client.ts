/**
 * CityMind AI — API client for FastAPI backend.
 */

export const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api";
export const WS_SIMULATION = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/simulation";
const API_KEY = import.meta.env.VITE_API_KEY;

// ── Types ─────────────────────────────────────────────────────────────────────

export interface NetworkGeoJSON {
  type: string;
  features: Array<{
    type: string;
    id: string;
    properties: {
      type: "lane" | "junction";
      edge_id?: string;
      speed?: number;
      is_tls?: boolean;
      tls_id?: string;
      node_type?: string;
    };
    geometry: {
      type: "LineString" | "Point";
      coordinates: number[][] | number[];
    };
  }>;
}

export interface ExperimentSummary {
  controller: string;
  seed: number;
  demand_id: number | string;
  total_steps: number;
  final_sim_time: number;
  average_waiting_time: number;
  total_waiting_time: number;
  average_queue_length: number;
  maximum_queue_length: number;
  average_speed: number;
  throughput: number;
  total_switches: number;
  cumulative_reward: number;
}

export interface ExperimentRun {
  run_id: string;
  algorithm: string;
  seed: number;
  total_timesteps: number;
  model_path?: string | null;
  has_model?: boolean;
  config?: Record<string, unknown>;
}

export interface ComparisonEntry {
  controller: string;
  label: string;
  mean_waiting_time: number;
  mean_queue: number;
  throughput: number;
  mean_speed: number;
  total_arrived: number;
  cumulative_reward: number;
  backlog: number;
  total_teleports: number;
  trips_completed: number;
  mean_trip_timeloss: number;
}

export interface ComparisonImprovement {
  controller?: string;
  baseline?: string;
  waiting_time_improvement_pct?: number;
  throughput_improvement_pct?: number;
  trips_completed_gap_pct?: number;
  starvation_flag?: boolean;
}

export interface ComparisonResult {
  results: ComparisonEntry[];
  improvement?: ComparisonImprovement;
  improvement_vs_actuated?: ComparisonImprovement;
}

export interface MetricsSummary {
  controller: string;
  mean_waiting_time: number;
  mean_queue: number;
  throughput: number;
  mean_speed: number;
  total_arrived: number;
  total_steps: number;
  cumulative_reward: number;
  total_switches: number;
  backlog: number;
  total_teleports: number;
  trips_completed: number;
  mean_trip_timeloss: number;
  trips_completed_gap_pct?: number;
  starvation_flag?: boolean;
}

export type SimControllerId = "fixed_time" | "sumo_actuated" | "max_pressure" | "ppo" | "ippo" | "ippo_phase_select";

export interface SimulationStatus {
  gui_running: boolean;
  gui_pid?: number | null;
  latest_model?: string | null;
  xquartz_display?: string | null;
  running_controllers?: SimControllerId[];
}

export interface LaunchResponse {
  status: string;
  pid?: number | null;
  model?: string | null;
  message: string;
}

export interface XAIJunctionExplanation {
  tls_id: string;
  tls_index: number;
  action: number;
  switch_probability: number;
  narrative: string;
  local_influence_pct: number;
  neighbor_influence_pct: number;
  group_breakdown: Record<string, number>;
  top_positive_features: Array<{
    feature_index: number;
    name: string;
    group: string;
    tls_id: string;
    lane_id?: string;
    value: number;
    attribution: number;
  }>;
  top_negative_features: Array<{
    feature_index: number;
    name: string;
    group: string;
    tls_id: string;
    lane_id?: string;
    value: number;
    attribution: number;
  }>;
}

export interface XAIMeta {
  controller: string;
  explains: string;
  coordination_evidence_type: "model_attribution" | "empirical_correlation";
  coordination_evidence_label: string;
  coordination_evidence_description: string;
}

export interface WsPayload {
  type: string;
  timestamp: number;
  gui_running: boolean;
  gui_pid?: number | null;
  running_controllers?: SimControllerId[];
  metrics: Record<string, {
    mean_waiting_time: number;
    mean_queue: number;
    throughput: number;
    mean_speed: number;
    total_arrived: number;
    cumulative_reward: number;
  }>;
}

export interface TLSFrame {
  tls_id: string;
  phase: number;
  state: string;
}

export interface VehiclePointDTO {
  vehicle_id: string;
  x: number;
  y: number;
  speed: number;
}

// ── API Client ────────────────────────────────────────────────────────────────

async function _fetch<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers);
  if (API_KEY) headers.set("X-API-Key", API_KEY);
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${path} → ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  // ── Network ────────────────────────────────────────────────────────────
  getNetworkGeoJSON: () => _fetch<NetworkGeoJSON>("/network/geojson"),
  getTLSMetadata: () => _fetch<Array<{
    tls_id: string;
    program_id: string;
    phases: Array<{ state: string; duration: number }>;
  }>>("/network/tls_metadata"),

  // ── Experiments ────────────────────────────────────────────────────────
  getExperiments: () => _fetch<ExperimentRun[]>("/experiments/"),
  getExperiment: (runId: string) => _fetch<ExperimentRun>(`/experiments/${runId}`),
  getExperimentSummary: (runId: string) => _fetch<ExperimentSummary>(`/experiments/${runId}/summary`),

  // ── Metrics ────────────────────────────────────────────────────────────
  getComparisonMetrics: () => _fetch<ComparisonResult>("/metrics/comparison"),
  getMetricsSummary: () => _fetch<MetricsSummary>("/metrics/summary"),

  // ── Simulation Control (XQuartz) ───────────────────────────────────────
  getSimulationStatus: () => _fetch<SimulationStatus>("/simulation/status"),
  // FastAPI binds these as query params (no Pydantic body on the endpoint),
  // so they're sent on the URL, not as a JSON body — the previous version of
  // this call sent a JSON body FastAPI was never reading.
  launchSimulationGui: (controllerId: SimControllerId, model?: string, delay = 80, duration = 3600) => {
    const params = new URLSearchParams({ controller: controllerId, delay: String(delay), duration: String(duration) });
    if (model) params.set("model", model);
    return _fetch<LaunchResponse>(`/simulation/launch-gui?${params.toString()}`, { method: "POST" });
  },
  stopSimulationGui: (controllerId?: SimControllerId) => {
    const params = controllerId ? `?${new URLSearchParams({ controller: controllerId }).toString()}` : "";
    return _fetch<{ status: string; message: string }>(`/simulation/stop-gui${params}`, { method: "POST" });
  },

  // ── XAI ───────────────────────────────────────────────────────────────
  getXAISummary: (controller: string = "ppo") => _fetch<XAIJunctionExplanation[]>(`/xai/summary?controller=${controller}`),
  getXAIJunction: (tlsId: string, controller: string = "ppo") => _fetch<XAIJunctionExplanation>(`/xai/explain/${tlsId}?controller=${controller}`),
  getXAIMeta: (controller: string = "ppo") => _fetch<XAIMeta>(`/xai/meta?controller=${controller}`),

  // ── Replay ────────────────────────────────────────────────────────────
  getReplayFrames: (controller: string, seed = 42, demandId = "1.0x", startTime = 0, endTime = 60) => {
    const params = new URLSearchParams({
      controller,
      seed: String(seed),
      demand_id: demandId,
      start_time: String(startTime),
      end_time: String(endTime),
    });
    return _fetch<{
      frames: Record<string, VehiclePointDTO[]>;
      tls_frames: Record<string, TLSFrame[]>;
      frame_count: number;
    }>(`/replay/frames?${params.toString()}`);
  },
};
