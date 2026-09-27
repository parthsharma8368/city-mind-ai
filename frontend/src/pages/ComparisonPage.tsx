import React, { useEffect, useState } from "react";
import { BarChart3, TrendingDown, TrendingUp, Award, AlertTriangle } from "lucide-react";
import { ComparisonBarChart } from "../components/MetricsChart";
import { api } from "../api/client";
import type { ComparisonResult, ComparisonEntry, ComparisonImprovement } from "../api/client";
import { CONTROLLER_LABELS as CTRL_LABELS, CONTROLLER_COLORS as CTRL_COLORS } from "../constants/controllers";

type MetricKey = "mean_waiting_time" | "mean_queue" | "throughput" | "mean_speed" | "trips_completed" | "backlog";

function MetricRow({ label, entry, baseline }: { label: MetricKey; entry: ComparisonEntry; baseline?: ComparisonEntry }) {
  const baseVal = baseline ? baseline[label] as number : 0;
  const myVal = entry[label] as number;
  const diff = baseVal > 0 ? (myVal - baseVal) / baseVal * 100 : 0;
  // Higher-is-better metrics: throughput, mean_speed, trips_completed.
  // Everything else (including backlog) is lower-is-better.
  const lowerIsBetter = label !== "throughput" && label !== "mean_speed" && label !== "trips_completed";
  const improved = lowerIsBetter ? diff < 0 : diff > 0;
  const isCount = label === "trips_completed" || label === "backlog";

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0.6rem 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
      <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>{label.replace(/_/g, " ")}</span>
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.9rem", color: "var(--text-primary)", fontWeight: 600 }}>
          {typeof myVal !== "number" ? "—" : isCount ? myVal.toLocaleString() : myVal.toFixed(3)}
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
  const improvementVsActuated = data.improvement_vs_actuated;
  // fixed_time stays the reference baseline for the table/cards below — but the
  // starvation check right under the header looks at BOTH comparisons, because
  // sumo_actuated (the calibrated, already-adaptive baseline) is the harder and
  // more honest bar to clear.
  const baseline = results.find((r) => r.controller === "fixed_time");

  const starvationSources: ComparisonImprovement[] = [improvement, improvementVsActuated].filter(
    (s): s is ComparisonImprovement => Boolean(s?.starvation_flag)
  );

  // Only claim a real win when the backend has explicitly ruled out starvation
  // (trips_completed_gap_pct >= -2.0) — an unqualified wait-time badge is exactly
  // the kind of misleading headline this page exists to avoid.
  const showsRealImprovement = improvement?.starvation_flag === false && improvement?.waiting_time_improvement_pct !== undefined;

  return (
    <div style={{ animation: "fade-in 0.4s ease both" }}>
      {/* Header */}
      <div style={{ marginBottom: "1.5rem", display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <h2 className="section-title">Controller Comparison</h2>
          <p className="section-subtitle">PPO · IPPO · Fixed-Time · Max-Pressure · InTAS Ingolstadt · 1.0× demand</p>
        </div>
        {showsRealImprovement && (
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            {(() => {
              const pct = improvement!.waiting_time_improvement_pct!;
              const isImprovement = pct > 0;
              const label = improvement!.controller
                ? CTRL_LABELS[improvement!.controller!] ?? improvement!.controller
                : "RL controller";
              const tripsGap = improvement!.trips_completed_gap_pct;
              return (
                <span className={`improvement-tag ${isImprovement ? "positive" : "negative"}`}>
                  {isImprovement ? <Award size={12} /> : <TrendingUp size={12} />}
                  {label}: {Math.abs(pct).toFixed(1)}% wait time {isImprovement ? "reduction" : "increase"} vs Fixed-Time
                  {tripsGap !== undefined && ` · ${tripsGap >= 0 ? "+" : ""}${tripsGap.toFixed(1)}% trips completed`}
                </span>
              );
            })()}
          </div>
        )}
      </div>

      {/* Starvation warning — a lower average wait time next to fewer completed
          trips is an artifact of vehicles never finishing, not a real win. */}
      {starvationSources.length > 0 && (
        <div
          className="connectivity-banner"
          role="alert"
          style={{
            background: "rgba(245, 158, 11, 0.1)",
            borderColor: "rgba(245, 158, 11, 0.35)",
            color: "var(--signal-yellow)",
            alignItems: "flex-start",
          }}
        >
          <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: 2 }} aria-hidden="true" />
          <span>
            <strong>Starvation, not improvement.</strong> A lower average wait time below does not mean the RL
            controller is faster — it means fewer vehicles are finishing their trips while more sit stuck in the
            network (backlog).
            <ul style={{ margin: "0.5rem 0 0", paddingLeft: "1.1rem" }}>
              {starvationSources.map((s) => (
                <li key={`${s.controller}-${s.baseline}`}>
                  <strong>{s.controller ? CTRL_LABELS[s.controller] ?? s.controller : "The RL controller"}</strong>{" "}
                  completes <strong>{Math.abs(s.trips_completed_gap_pct ?? 0).toFixed(1)}%</strong> fewer trips than{" "}
                  {s.baseline ? CTRL_LABELS[s.baseline] ?? s.baseline : "the baseline"}, despite a lower reported
                  wait time.
                </li>
              ))}
            </ul>
          </span>
        </div>
      )}

      {/* Charts — lead with the honest, trip-completion-grounded metrics.
          mean_waiting_time and throughput (which look best for the RL
          controllers but can be starvation artifacts) live in the table below,
          not the headline row. */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem", marginBottom: "1.5rem" }}>
        <div className="glass-card" style={{ padding: "1.25rem" }}>
          <ComparisonBarChart data={results} metric="trips_completed" title="Trips Completed" />
        </div>
        <div className="glass-card" style={{ padding: "1.25rem" }}>
          <ComparisonBarChart data={results} metric="mean_trip_timeloss" title="Mean Trip Timeloss" unit="s" />
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
                <th>Trips Completed</th>
                <th>Backlog</th>
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
                  <td style={{ fontFamily: "var(--font-mono)" }}>{r.trips_completed?.toLocaleString() ?? "—"}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{r.backlog?.toLocaleString() ?? "—"}</td>
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
            {(["mean_waiting_time", "mean_queue", "throughput", "mean_speed", "trips_completed", "backlog"] as MetricKey[]).map((key) => (
              <MetricRow key={key} label={key} entry={r} baseline={baseline?.controller !== r.controller ? baseline : undefined} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};
