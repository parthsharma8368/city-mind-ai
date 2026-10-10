import React, { useEffect, useState } from "react";
import { api, isAbortError, type ExperimentRun, type ExperimentSummary } from "../api/client";
import { AlertCircle, CheckCircle2, Clock, FlaskConical } from "lucide-react";
import { fmt, fmtInt } from "../utils/format";

interface RunWithSummary extends ExperimentRun {
  summary?: ExperimentSummary;
}

export const ExperimentsPage: React.FC = () => {
  const [runs, setRuns] = useState<RunWithSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    async function loadData() {
      try {
        const list = await api.getExperiments(ctrl.signal);
        const runsWithSummary = await Promise.all(
          list.map(async (item) => {
            try {
              const summary = await api.getExperimentSummary(item.run_id, ctrl.signal);
              return { ...item, summary };
            } catch {
              return { ...item };
            }
          })
        );
        if (ctrl.signal.aborted) return;
        setRuns(runsWithSummary);
        setError(null);
      } catch (err) {
        if (ctrl.signal.aborted || isAbortError(err)) return;
        setError(err instanceof Error ? err.message : "Failed to load experiments");
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    }
    loadData();
    return () => ctrl.abort();
  }, []);

  return (
    <div style={{ animation: "fade-in 0.4s ease both" }}>
      <div style={{ marginBottom: "1.5rem", display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <h2 className="section-title">Experiment Tracking</h2>
          <p className="section-subtitle">Training runs under artifacts/experiments/ (PPO, IPPO and MaskablePPO phase-selection) — metrics shown only once evaluated against that run's own checkpoint. Wait time alone can reflect starvation: compare it with trips completed on the Comparison page.</p>
        </div>
        {!loading && <span className="badge-blue">{runs.length} runs</span>}
      </div>

      {loading ? (
        <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--text-muted)", padding: "2rem" }}>
          <span className="spinner" /> Loading experiments…
        </div>
      ) : error ? (
        <div className="glass-card" style={{ padding: "3rem", textAlign: "center" }}>
          <AlertCircle size={32} style={{ color: "var(--signal-red)", margin: "0 auto 1rem" }} />
          <h3 style={{ color: "var(--text-secondary)" }}>Couldn't Load Experiments</h3>
          <p style={{ marginTop: "0.5rem", fontSize: "0.875rem", color: "var(--text-muted)" }}>
            {error}
          </p>
        </div>
      ) : runs.length === 0 ? (
        <div className="glass-card" style={{ padding: "3rem", textAlign: "center", color: "var(--text-secondary)" }}>
          <FlaskConical size={32} style={{ margin: "0 auto 1rem", opacity: 0.6 }} />
          <h3>No Completed Runs Found</h3>
          <p style={{ marginTop: "0.5rem", fontSize: "0.875rem", color: "var(--text-muted)" }}>
            Run training to populate benchmark artifacts:
          </p>
          <code style={{
            display: "block", marginTop: "1rem",
            padding: "0.75rem 1.25rem",
            background: "rgba(255,255,255,0.05)", borderRadius: 8,
            color: "var(--accent-cyan)", fontFamily: "var(--font-mono)",
            fontSize: "0.8rem", textAlign: "left", whiteSpace: "pre-wrap",
          }}>
            {"python scripts/run_training.py --ppo-config configs/ppo.yaml\npython scripts/run_ippo_training.py --ippo-config configs/ippo.yaml"}
          </code>
        </div>
      ) : (
        <div className="glass-card" style={{ overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Algorithm</th>
                <th>Seed</th>
                <th>Timesteps</th>
                <th title="Mean wait per vehicle. Can look good when a controller starves traffic: check trips completed on the Comparison page.">Avg Wait</th>
                <th>Throughput</th>
                <th>Cum. Reward</th>
                <th>Model</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id}>
                  <td>
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                      {r.run_id}
                    </span>
                  </td>
                  <td><span className="badge-blue">{r.algorithm}</span></td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                    {r.summary?.seed ?? r.seed ?? "—"}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                    {fmtInt(r.total_timesteps)}
                  </td>
                  <td style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                    {r.summary && Number.isFinite(r.summary.average_waiting_time) ? `${fmt(r.summary.average_waiting_time, 1)} s` : "—"}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>
                    {r.summary && Number.isFinite(r.summary.throughput) ? `${fmt(r.summary.throughput * 100, 2)}%` : "—"}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", color: r.summary?.cumulative_reward !== undefined && r.summary.cumulative_reward >= 0 ? "var(--signal-green)" : "var(--signal-red)" }}>
                    {fmt(r.summary?.cumulative_reward, 1)}
                  </td>
                  <td>
                    {r.has_model || r.model_path
                      ? <span className="badge-green"><CheckCircle2 size={10} /> model_final.zip</span>
                      : <span className="badge-yellow"><Clock size={10} /> No model</span>
                    }
                  </td>
                  <td>
                    {r.summary && (r.has_model || r.model_path)
                      ? <span className="badge-green"><CheckCircle2 size={10} /> Evaluated</span>
                      : <span className="badge-yellow"><Clock size={10} /> Incomplete</span>
                    }
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
