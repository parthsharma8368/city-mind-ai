/**
 * CityMind AI — API client for FastAPI backend.
 */

// Defaults target the loopback-only dev backend. Override with VITE_API_URL /
// VITE_WS_URL (see frontend/.env.example). NOTE: never put secrets in VITE_*
// variables -- Vite inlines them into the public bundle, so no API key is
// read here; the backend control endpoints are loopback-only by default.
const DEFAULT_API_BASE = "http://127.0.0.1:8000/api";

export const API_BASE: string = (import.meta.env.VITE_API_URL || DEFAULT_API_BASE).replace(/\/+$/, "");
/** Backend origin without the /api suffix (used for the docs link). */
export const API_ORIGIN: string = API_BASE.replace(/\/api$/, "");
export const API_DOCS_URL = `${API_ORIGIN}/docs`;

function deriveWsUrl(apiBase: string): string {
  return `${apiBase.replace(/\/api$/, "").replace(/^http/, "ws")}/ws/simulation`;
}
export const WS_SIMULATION: string = import.meta.env.VITE_WS_URL || deriveWsUrl(API_BASE);

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
  cumulative_reward: number | null;
  backlog: number;
  total_teleports: number;
  trips_completed: number;
  mean_trip_timeloss: number;
  /** Mean over the evaluation seeds (the comparison endpoint); absent in older payloads. */
  n_seeds?: number;
  lag_per_scheduled_car_s?: number;
  never_inserted?: number;
  due_completion_ratio?: number;
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
  seed?: number;
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

export type SimControllerId = "fixed_time" | "sumo_actuated" | "ippo_A" | "ippo_B";

export interface TenSeedMeasure {
  key: string;
  label: string;
  better: "lower" | "higher";
  baseline_mean: number;
  model_mean: number;
  pct_diff: number;
  ci_lo: number;
  ci_hi: number;
  seeds_won: number;
  n_seeds: number;
  sign_test_p: number;
  verdict: string;
}

export interface TenSeedResult {
  baseline: string;
  seeds: number[];
  models: Record<string, { label: string; measures: TenSeedMeasure[] }>;
}

export interface SimulationControllerInfo {
  id: SimControllerId;
  label: string;
  kind: "baseline" | "model";
  available: boolean;
}

export interface SimulationStatus {
  gui_running: boolean;
  gui_pid?: number | null;
  xquartz_display?: string | null;
  running_controllers?: SimControllerId[];
  controllers?: SimulationControllerInfo[];
  /** Why a window stopped by itself (exit code and the last log line), by controller id. */
  last_errors?: Record<string, string>;
}

export interface LaunchOptions {
  delay?: number;
  duration?: number;
  /** Window position "X,Y" and size "W,H" (to tile several SUMO windows). */
  pos?: string;
  size?: string;
}

// ── Showcase (compact recorded frames + stored metrics) ──────────────────────
export interface ShowcaseControllerInfo {
  id: SimControllerId;
  label: string;
  kind: "baseline" | "model";
  has_recording: boolean;
  has_metrics: boolean;
}

export interface ShowcaseConfig {
  seed: number;
  start: number;
  end: number;
  step: number;
  speed: number;
  hour: [number, number];
  demand_id: string;
  n_seeds: number;
  controllers: ShowcaseControllerInfo[];
}

/** Parallel arrays: car r of frame k is (vid[r], x[r], y[r], v[r]) for r in [sum(counts[:k]), sum(counts[:k+1])). */
export interface ShowcaseFramesDTO {
  controller: string;
  seed: number;
  start: number;
  end: number;
  step: number;
  origin: [number, number];
  times: number[];
  counts: number[];
  vid: number[];
  x: number[];
  y: number[];
  v: number[];
  vehicle_count: number;
  tls_ids: string[];
  /** One g/y/r/n character per junction (in tls_ids order) for every frame. */
  tls: string[];
}

export interface ShowcaseMeasure {
  key: string;
  label: string;
  better: "lower" | "higher";
  format: string;
  mean: number;
  min: number;
  max: number;
  per_seed: number[];
}

export interface ShowcaseMetrics {
  controller: string;
  label: string;
  kind: "baseline" | "model";
  seeds: number[];
  n_runs: number;
  window: string;
  source: string;
  measures: ShowcaseMeasure[];
}

export interface PolicyDenseDTO {
  /** [outputs, inputs]; `w` is row-major (output-major) and `b` has one bias per output. */
  shape: [number, number];
  w: number[];
  b: number[];
}

export interface PolicyNetDTO {
  l1: PolicyDenseDTO;
  l2: PolicyDenseDTO;
  out: PolicyDenseDTO;
}

/** The trained weights of a final model: actor 22 -> 256 -> 256 -> 2 and critic 22 -> 256 -> 256 -> 1 (tanh hidden layers). */
export interface PolicyDTO {
  controller: string;
  label: string;
  obs_names: string[];
  actions: string[];
  activation: "tanh";
  actor: PolicyNetDTO;
  critic: PolicyNetDTO;
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

export async function _fetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${path} → ${res.status}: ${text.slice(0, 200)}`);
  }
  try {
    return (await res.json()) as T;
  } catch {
    throw new Error(`API ${path} → ${res.status}: response was not valid JSON`);
  }
}

/** True for the error thrown when a fetch is cancelled via AbortController. */
export function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

// Module-level promise caches: the 4.9 MB network GeoJSON and TLS metadata are
// static for a session, so every NetworkCanvas shares one download. A failed
// fetch clears the cache so the next caller can retry.
function cachedOnce<T>(load: () => Promise<T>): () => Promise<T> {
  let promise: Promise<T> | null = null;
  return () => {
    if (!promise) {
      promise = load().catch((err) => {
        promise = null;
        throw err;
      });
    }
    return promise;
  };
}

type TLSMetadata = Array<{
  tls_id: string;
  program_id: string;
  phases: Array<{ state: string; duration: number }>;
}>;

const loadNetworkGeoJSON = cachedOnce(() => _fetch<NetworkGeoJSON>("/network/geojson"));
const loadTLSMetadata = cachedOnce(() => _fetch<TLSMetadata>("/network/tls_metadata"));

export const api = {
  // ── Network ────────────────────────────────────────────────────────────
  getNetworkGeoJSON: loadNetworkGeoJSON,
  getTLSMetadata: loadTLSMetadata,

  // ── Experiments ────────────────────────────────────────────────────────
  getExperiments: (signal?: AbortSignal) => _fetch<ExperimentRun[]>("/experiments/", { signal }),
  getExperiment: (runId: string, signal?: AbortSignal) => _fetch<ExperimentRun>(`/experiments/${encodeURIComponent(runId)}`, { signal }),
  getExperimentSummary: (runId: string, signal?: AbortSignal) =>
    _fetch<ExperimentSummary>(`/experiments/${encodeURIComponent(runId)}/summary`, { signal }),

  // ── Metrics ────────────────────────────────────────────────────────────
  getComparisonMetrics: (signal?: AbortSignal) => _fetch<ComparisonResult>("/metrics/comparison", { signal }),
  getMetricsSummary: (signal?: AbortSignal) => _fetch<MetricsSummary>("/metrics/summary", { signal }),
  getTenSeed: (signal?: AbortSignal) => _fetch<TenSeedResult>("/metrics/ten-seed", { signal }),

  // ── Simulation Control (XQuartz) ───────────────────────────────────────
  getSimulationStatus: (signal?: AbortSignal) => _fetch<SimulationStatus>("/simulation/status", { signal }),
  // FastAPI binds these as query params (no Pydantic body on the endpoint),
  // so they're sent on the URL, not as a JSON body — the previous version of
  // this call sent a JSON body FastAPI was never reading.
  launchSimulationGui: (controllerId: SimControllerId, options: LaunchOptions = {}) => {
    const params = new URLSearchParams({
      controller: controllerId,
      delay: String(options.delay ?? 80),
      duration: String(options.duration ?? 3600),
    });
    if (options.pos) params.set("pos", options.pos);
    if (options.size) params.set("size", options.size);
    return _fetch<LaunchResponse>(`/simulation/launch-gui?${params.toString()}`, { method: "POST" });
  },
  stopSimulationGui: (controllerId?: SimControllerId) => {
    const params = controllerId ? `?${new URLSearchParams({ controller: controllerId }).toString()}` : "";
    return _fetch<{ status: string; message: string }>(`/simulation/stop-gui${params}`, { method: "POST" });
  },

  // ── Showcase ───────────────────────────────────────────────────────────
  getShowcaseConfig: (signal?: AbortSignal) => _fetch<ShowcaseConfig>("/showcase/config", { signal }),
  getShowcaseFrames: (controller: string, start: number, end: number, seed: number, signal?: AbortSignal) =>
    _fetch<ShowcaseFramesDTO>(
      `/showcase/frames?${new URLSearchParams({ controller, start: String(start), end: String(end), seed: String(seed) }).toString()}`,
      { signal },
    ),
  getShowcaseMetrics: (controller: string, signal?: AbortSignal) =>
    _fetch<ShowcaseMetrics>(`/showcase/metrics/${encodeURIComponent(controller)}`, { signal }),
  getPolicy: (controller: string, signal?: AbortSignal) =>
    _fetch<PolicyDTO>(`/showcase/policy/${encodeURIComponent(controller)}`, { signal }),

  // ── XAI ───────────────────────────────────────────────────────────────
  getXAISummary: (controller: string = "ppo", signal?: AbortSignal) =>
    _fetch<XAIJunctionExplanation[]>(`/xai/summary?controller=${encodeURIComponent(controller)}`, { signal }),
  getXAIJunction: (tlsId: string, controller: string = "ppo", signal?: AbortSignal) =>
    _fetch<XAIJunctionExplanation>(`/xai/explain/${encodeURIComponent(tlsId)}?controller=${encodeURIComponent(controller)}`, { signal }),
  getXAIMeta: (controller: string = "ppo", signal?: AbortSignal) =>
    _fetch<XAIMeta>(`/xai/meta?controller=${encodeURIComponent(controller)}`, { signal }),

  // ── Replay ────────────────────────────────────────────────────────────
  getReplayFrames: (controller: string, seed = 42, demandId = "1.0x", startTime = 0, endTime = 60, signal?: AbortSignal) => {
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
    }>(`/replay/frames?${params.toString()}`, { signal });
  },
};
