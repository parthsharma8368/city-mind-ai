import React, { useEffect, useState } from "react";
import { BarChart3, TrendingDown, TrendingUp, Award } from "lucide-react";
import { ComparisonBarChart } from "../components/MetricsChart";
import { api } from "../api/client";
import type { ComparisonResult, ComparisonEntry } from "../api/client";
import { CONTROLLER_LABELS as CTRL_LABELS, CONTROLLER_COLORS as CTRL_COLORS } from "../constants/controllers";

type MetricKey = "mean_waiting_time" | "mean_queue" | "throughput" | "mean_speed";

function MetricRow({ label, entry, baseline }: { label: MetricKey; entry: ComparisonEntry; baseline?: ComparisonEntry }) {
  const baseVal = baseline ? baseline[label] as number : 0;
  const myVal = entry[label] as number;
  const diff = baseVal > 0 ? (myVal - baseVal) / baseVal * 100 : 0;
  const lowerIsBetter = label !== "throughput" && label !== "mean_speed";
  const improved = lowerIsBetter ? diff < 0 : diff > 0;

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0.6rem 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
      <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>{label.replace(/_/g, " ")}</span>
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.9rem", color: "var(--text-primary)", fontWeight: 600 }}>
          {typeof myVal === "number" ? myVal.toFixed(3) : "—"}
        </span>
        {baseline && diff !== 0 && (
          <span className={`improvement-tag ${improved ? "positive" : "negative"}`}>
            {improved ? <TrendingDown size={10} /> : <TrendingUp size={10} />}
            {Math.abs(diff).toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  );
}

export const ComparisonPage: React.FC = () => {
  const [data, setData] = useState<ComparisonResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getComparisonMetrics()
      .then(setData)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "3rem", color: "var(--text-muted)" }}>
      <span className="spinner" /> Loading comparison data…
    </div>
  );

  if (error || !data || data.results.length === 0) {
    return (
      <div className="glass-card" style={{ padding: "3rem", textAlign: "center" }}>
        <BarChart3 size={36} style={{ color: "var(--text-muted)", margin: "0 auto 1rem" }} />
        <h3 style={{ color: "var(--text-secondary)", fontWeight: 600 }}>No Comparison Data</h3>
        <p style={{ fontSize: "0.875rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
          Run evaluation to generate comparison metrics.
        </p>
        <code style={{ display: "block", marginTop: "1rem", padding: "0.75rem 1rem", background: "rgba(255,255,255,0.05)", borderRadius: 8, fontSize: "0.8rem", color: "var(--accent-cyan)", fontFamily: "var(--font-mono)", textAlign: "left" }}>
          python scripts/run_evaluation.py --sumo-cfg data/raw/intas/scenario/InTAS_buildings.sumocfg --plot
        </code>
      </div>
    );
  }

  const { results, improvement } = data;
  const baseline = results.find((r) => r.controller === "fixed_time");

  return (
    <div style={{ animation: "fade-in 0.4s ease both" }}>
      {/* Header */}
      <div style={{ marginBottom: "1.5rem", display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <h2 className="section-title">Controller Comparison</h2>
          <p className="section-subtitle">PPO · IPPO · Fixed-Time · Max-Pressure · InTAS Ingolstadt · 1.0× demand</p>
        </div>
        {improvement?.waiting_time_improvement_pct !== undefined && (
          <div style={{ display: "flex", gap: "0.5rem" }}>
            {(() => {
              const pct = improvement.waiting_time_improvement_pct;
              const isImprovement = pct > 0;
              const label = improvement.controller
                ? CTRL_LABELS[improvement.controller] ?? improvement.controller
                : "RL controller";
              return (
                <span className={`improvement-tag ${isImprovement ? "positive" : "negative"}`}>
                  {isImprovement ? <Award size={12} /> : <TrendingUp size={12} />}
                  {label}: {Math.abs(pct).toFixed(1)}% wait time {isImprovement ? "reduction" : "increase"} vs Fixed-Time
                </span>
              );
            })()}
          </div>
        )}
      </div>

      {/* Charts */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem", marginBottom: "1.5rem" }}>
        <div className="glass-card" style={{ padding: "1.25rem" }}>
          <ComparisonBarChart data={results} metric="mean_waiting_time" title="Avg Waiting Time" unit="s" />
        </div>
        <div className="glass-card" style={{ padding: "1.25rem" }}>
          <ComparisonBarChart data={results} metric="throughput" title="Throughput" />
        </div>
        <div className="glass-card" style={{ padding: "1.25rem" }}>
          <ComparisonBarChart data={results} metric="mean_queue" title="Avg Queue Length" unit=" veh" />
        </div>
        <div className="glass-card" style={{ padding: "1.25rem" }}>
          <ComparisonBarChart data={results} metric="mean_speed" title="Mean Speed" unit=" m/s" />
        </div>
      </div>

      {/* Detailed Table */}
      <div className="glass-card" style={{ padding: "1.5rem", marginBottom: "1.5rem" }}>
        <h3 style={{ fontSize: "0.95rem", fontWeight: 600, marginBottom: "1.25rem" }}>Detailed Metrics Table</h3>
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
                <th>Cum. Reward</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.controller}>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ width: 10, height: 10, borderRadius: "50%", background: CTRL_COLORS[r.controller] ?? "#64748b", display: "inline-block" }} />
                      <span style={{ fontWeight: 600 }}>{CTRL_LABELS[r.controller] ?? r.controller}</span>
                    </div>
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--accent-cyan)" }}>
                    {r.mean_waiting_time.toFixed(2)}
                    {baseline && r.controller !== "fixed_time" && (
                      <span className={`improvement-tag ${r.mean_waiting_time < baseline.mean_waiting_time ? "positive" : "negative"}`} style={{ marginLeft: 6, fontSize: "0.68rem" }}>
                        {((baseline.mean_waiting_time - r.mean_waiting_time) / baseline.mean_waiting_time * 100).toFixed(1)}%
                      </span>
                    )}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{r.mean_queue.toFixed(4)}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{(r.throughput * 100).toFixed(2)}%</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{r.mean_speed.toFixed(3)}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{r.total_arrived.toLocaleString()}</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: r.cumulative_reward >= 0 ? "var(--signal-green)" : "var(--signal-red)" }}>
                    {r.cumulative_reward.toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Individual controller cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px,1fr))", gap: "1rem" }}>
        {results.map((r) => (
          <div key={r.controller} className="glass-card" style={{ padding: "1.25rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "1rem" }}>
              <span style={{ width: 12, height: 12, borderRadius: "50%", background: CTRL_COLORS[r.controller] ?? "#64748b", display: "inline-block", boxShadow: `0 0 8px ${CTRL_COLORS[r.controller] ?? "#64748b"}` }} />
              <span style={{ fontWeight: 700 }}>{CTRL_LABELS[r.controller] ?? r.controller}</span>
            </div>
            {(["mean_waiting_time", "mean_queue", "throughput", "mean_speed"] as MetricKey[]).map((key) => (
              <MetricRow key={key} label={key} entry={r} baseline={baseline?.controller !== r.controller ? baseline : undefined} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};
