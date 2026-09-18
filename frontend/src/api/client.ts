/**
 * CityMind AI — API client for FastAPI backend.
 */

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api";
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
  average_travel_time: number;
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
}

export interface ComparisonResult {
  results: ComparisonEntry[];
  improvement?: {
    waiting_time_improvement_pct?: number;
    throughput_improvement_pct?: number;
  };
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
}

export interface SimulationStatus {
  gui_running: boolean;
  gui_pid?: number | null;
  latest_model?: string | null;
  xquartz_display?: string | null;
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

export interface WsPayload {
  type: string;
  timestamp: number;
  gui_running: boolean;
  gui_pid?: number | null;
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
  getNetworkMetadata: () => _fetch<Record<string, unknown>>("/network/network_metadata"),
  getScenarioSummary: () => _fetch<Record<string, unknown>>("/network/summary"),
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
  launchSimulationGui: (model?: string, delay?: number) =>
    _fetch<LaunchResponse>("/simulation/launch-gui", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, delay: delay ?? 80 }),
    }),
  stopSimulationGui: () =>
    _fetch<{ status: string; message: string }>("/simulation/stop-gui", { method: "POST" }),
  getLatestModel: () => _fetch<{ model_path: string; run_id: string; config: Record<string, unknown> }>("/simulation/latest-model"),

  // ── XAI ───────────────────────────────────────────────────────────────
  getXAISummary: () => _fetch<XAIJunctionExplanation[]>("/xai/summary"),
  getXAIJunction: (tlsId: string) => _fetch<XAIJunctionExplanation>(`/xai/explain/${tlsId}`),

  // ── Replay ────────────────────────────────────────────────────────────
  getAvailableTrajectories: () => _fetch<{ trajectories: string[] }>("/replay/available"),
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

  // ── Comparison ─────────────────────────────────────────────────────────
  getComparisonSummary: (demandId = "1.0x") =>
    _fetch<{ results: ComparisonEntry[] }>(`/comparison/summary?demand_id=${demandId}`),
};
