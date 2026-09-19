import React, { useEffect, useState } from "react";
import { api, type ExperimentRun, type ExperimentSummary } from "../api/client";
import { CheckCircle2, Clock, FlaskConical } from "lucide-react";

interface RunWithSummary extends ExperimentRun {
  summary?: ExperimentSummary;
}

export const ExperimentsPage: React.FC = () => {
  const [runs, setRuns] = useState<RunWithSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const list = await api.getExperiments();
        const runsWithSummary = await Promise.all(
          list.map(async (item) => {
            try {
              const summary = await api.getExperimentSummary(item.run_id);
              return { ...item, summary };
            } catch {
              return { ...item };
            }
          })
        );
        setRuns(runsWithSummary);
      } catch (err) {
        console.error("Failed to load experiments", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  return (
    <div style={{ animation: "fade-in 0.4s ease both" }}>
      <div style={{ marginBottom: "1.5rem", display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <h2 className="section-title">Experiment Tracking</h2>
          <p className="section-subtitle">Training runs from artifacts/experiments/{"{ppo,ippo}"}/ — metrics shown only once evaluated against that run's own checkpoint</p>
        </div>
        {!loading && <span className="badge-blue">{runs.length} runs</span>}
      </div>

      {loading ? (
        <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--text-muted)", padding: "2rem" }}>
          <span className="spinner" /> Loading experiments…
        </div>
      ) : runs.length === 0 ? (
        <div className="glass-card" style={{ padding: "3rem", textAlign: "center", color: "var(--text-secondary)" }}>
          <FlaskConical size={32} style={{ margin: "0 auto 1rem", opacity: 0.6 }} />
          <h3>No Completed Runs Found</h3>
          <p style={{ marginTop: "0.5rem", fontSize: "0.875rem", color: "var(--text-muted)" }}>
            Run PPO training to populate benchmark artifacts:
          </p>
          <code style={{
            display: "block", marginTop: "1rem",
            padding: "0.75rem 1.25rem",
            background: "rgba(255,255,255,0.05)", borderRadius: 8,
            color: "var(--accent-cyan)", fontFamily: "var(--font-mono)",
            fontSize: "0.8rem", textAlign: "left",
          }}>
            python -m src.agents.ppo_agent --timesteps 1000000
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
                <th>Avg Wait</th>
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
                    {r.total_timesteps?.toLocaleString() ?? "—"}
                  </td>
                  <td style={{ color: "var(--accent-cyan)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                    {r.summary ? `${r.summary.average_waiting_time.toFixed(1)} s` : "—"}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>
                    {r.summary ? `${(r.summary.throughput * 100).toFixed(2)}%` : "—"}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", color: r.summary?.cumulative_reward !== undefined && r.summary.cumulative_reward >= 0 ? "var(--signal-green)" : "var(--signal-red)" }}>
                    {r.summary?.cumulative_reward?.toFixed(1) ?? "—"}
                  </td>
                  <td>
                    {r.has_model || r.model_path
                      ? <span className="badge-green"><CheckCircle2 size={10} /> model_final.zip</span>
                      : <span className="badge-yellow"><Clock size={10} /> No model</span>
                    }
                  </td>
                  <td>
                    <span className="badge-green">
                      <CheckCircle2 size={10} /> Done
                    </span>
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
