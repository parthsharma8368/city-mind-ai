import React, { useEffect, useState } from "react";
import { BarChart3, TrendingDown, TrendingUp, Award, AlertTriangle } from "lucide-react";
import { TenSeedTable } from "../components/TenSeedTable";
import { ComparisonBarChart } from "../components/MetricsChart";
import { api, isAbortError } from "../api/client";
import { fmt, fmtInt, isNum, isStarving } from "../utils/format";
import { metricDelta, type MetricKey } from "../utils/compare";
import type { ComparisonResult, ComparisonEntry, ComparisonImprovement } from "../api/client";
import { CONTROLLER_LABELS as CTRL_LABELS, CONTROLLER_COLORS as CTRL_COLORS } from "../constants/controllers";


function MetricRow({ label, entry, baseline, starving = false }: { label: MetricKey; entry: ComparisonEntry; baseline?: ComparisonEntry; starving?: boolean }) {
  const myVal = entry[label] as number;
  const delta = baseline ? metricDelta(label, myVal, baseline[label] as number, starving) : null;
  const isCount = label === "trips_completed" || label === "backlog";

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0.6rem 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
      <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>{label.replace(/_/g, " ")}</span>
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.9rem", color: "var(--text-primary)", fontWeight: 600 }}>
          {isCount ? fmtInt(myVal) : fmt(myVal, 3)}
        </span>
        {delta && delta.diff !== 0 && (
          <span
            className={`improvement-tag ${delta.tone}`}
            title={delta.suspect ? "Looks better only because this controller completes fewer trips (starvation), not because it is faster." : undefined}
          >
            {/* arrow follows the sign of the change, not whether it is good */}
            {delta.diff < 0 ? <TrendingDown size={10} aria-hidden="true" /> : <TrendingUp size={10} aria-hidden="true" />}
            {delta.diff < 0 ? "-" : "+"}{Math.abs(delta.diff).toFixed(1)}%
            <span className="sr-only">{delta.suspect ? " (starvation artifact)" : delta.improved ? " (better)" : " (worse)"}</span>
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
    const ctrl = new AbortController();
    api.getComparisonMetrics(ctrl.signal)
      .then((d) => { if (!ctrl.signal.aborted) setData(d); })
      .catch((e: Error) => { if (!ctrl.signal.aborted && !isAbortError(e)) setError(e.message); })
      .finally(() => { if (!ctrl.signal.aborted) setLoading(false); });
    return () => ctrl.abort();
  }, []);

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "3rem", color: "var(--text-muted)" }}>
      <span className="spinner" /> Loading comparison data…
    </div>
  );

  if (error || !data || !Array.isArray(data.results) || data.results.length === 0) {
    return (
      <div className="glass-card" style={{ padding: "3rem", textAlign: "center" }}>
        <BarChart3 size={36} style={{ color: "var(--text-muted)", margin: "0 auto 1rem" }} />
        <h3 style={{ color: "var(--text-secondary)", fontWeight: 600 }}>No Comparison Data</h3>
        <p style={{ fontSize: "0.875rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
          Run evaluation to generate comparison metrics.
        </p>
        <code style={{ display: "block", marginTop: "1rem", padding: "0.75rem 1rem", background: "rgba(255,255,255,0.05)", borderRadius: 8, fontSize: "0.8rem", color: "var(--accent-cyan)", fontFamily: "var(--font-mono)", textAlign: "left" }}>
          python scripts/report_eval.py --root artifacts/eval10_run --baseline sumo_actuated --models ippo_A ippo_B --extra fixed_time
        </code>
      </div>
    );
  }

  const { results, improvement } = data;
  const improvementVsActuated = data.improvement_vs_actuated;
  // SUMO actuated (the calibrated, already-adaptive controller) is the bar every table and badge below is measured against;
  // the starvation check right under the header looks at both comparisons.
  const baseline = results.find((r) => r.controller === "sumo_actuated");

  const starvationSources: ComparisonImprovement[] = [improvement, improvementVsActuated].filter(
    (s): s is ComparisonImprovement => Boolean(s?.starvation_flag)
  );

  // Only claim a real win when the backend has explicitly ruled out starvation
  // (trips_completed_gap_pct >= -2.0) — an unqualified wait-time badge is exactly
  // the kind of misleading headline this page exists to avoid.
  const headline = improvementVsActuated;
  const showsRealImprovement = headline?.starvation_flag === false && headline?.waiting_time_improvement_pct !== undefined;

  return (
    <div style={{ animation: "fade-in 0.4s ease both" }}>
      {/* Header */}
      <div style={{ marginBottom: "1.5rem", display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <h2 className="section-title">Controller Comparison</h2>
          <p className="section-subtitle">Final models A and B · SUMO actuated (baseline) · fixed time · InTAS Ingolstadt 07:00-08:00 · mean of {results[0]?.n_seeds ?? 10} seeds</p>
        </div>
        {showsRealImprovement && (
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            {(() => {
              const pct = headline!.waiting_time_improvement_pct!;
              const isImprovement = pct > 0;
              const label = headline!.controller
                ? CTRL_LABELS[headline!.controller!] ?? headline!.controller
                : "RL controller";
              const tripsGap = headline!.trips_completed_gap_pct;
              return (
                <span className={`improvement-tag ${isImprovement ? "positive" : "negative"}`}>
                  {isImprovement ? <Award size={12} /> : <TrendingUp size={12} />}
                  {label}: {fmt(Math.abs(pct), 1)}% wait time {isImprovement ? "reduction" : "increase"} vs SUMO actuated
                  {isNum(tripsGap) && ` · ${tripsGap >= 0 ? "+" : ""}${fmt(tripsGap, 1)}% trips completed`}
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
            <strong>Starvation, not improvement.</strong> Average wait time only counts vehicles that are moving
            through the network: a controller that lets fewer vehicles finish their trips while more sit stuck
            (backlog) can post a good wait time without being faster. Read wait time together with trips completed.
            <ul style={{ margin: "0.5rem 0 0", paddingLeft: "1.1rem" }}>
              {starvationSources.map((s) => (
                <li key={`${s.controller}-${s.baseline}`}>
                  <strong>{s.controller ? CTRL_LABELS[s.controller] ?? s.controller : "The RL controller"}</strong>{" "}
                  completes <strong>{fmt(Math.abs(s.trips_completed_gap_pct ?? 0), 1)}%</strong> fewer trips than{" "}
                  {s.baseline ? CTRL_LABELS[s.baseline] ?? s.baseline : "the baseline"}
                  {isNum(s.waiting_time_improvement_pct)
                    ? s.waiting_time_improvement_pct > 0
                      ? `, and reports a ${fmt(s.waiting_time_improvement_pct, 1)}% lower wait time only as a result of that.`
                      : `, and its wait time is also ${fmt(Math.abs(s.waiting_time_improvement_pct), 1)}% higher.`
                    : "."}
                </li>
              ))}
            </ul>
          </span>
        </div>
      )}

      {/* Charts: the main score first (lag per scheduled car counts cars that never finish), then trips and waiting. */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem", marginBottom: "1.5rem" }}>
        <div className="glass-card" style={{ padding: "1.25rem" }}>
          <ComparisonBarChart data={results} metric="lag_per_scheduled_car_s" title="Lag per scheduled car" unit="s" />
        </div>
        <div className="glass-card" style={{ padding: "1.25rem" }}>
          <ComparisonBarChart data={results} metric="trips_completed" title="Trips completed per hour" isCount />
        </div>
        <div className="glass-card" style={{ padding: "1.25rem" }}>
          <ComparisonBarChart data={results} metric="mean_waiting_time" title="Mean waiting time" unit=" s" />
        </div>
        <div className="glass-card" style={{ padding: "1.25rem" }}>
          <ComparisonBarChart data={results} metric="mean_queue" title="Mean queue length" unit=" veh" />
        </div>
      </div>

      {/* The paired test over all seeds */}
      <div className="glass-card" style={{ padding: "1.5rem", marginBottom: "1.5rem" }}>
        <h3 style={{ fontSize: "0.95rem", fontWeight: 600, marginBottom: "0.35rem" }}>Final models vs SUMO actuated — every measure, paired by seed</h3>
        <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
          The bars above are means. This table shows whether the difference is real: it is computed seed by seed on identical traffic.
        </p>
        <TenSeedTable />
      </div>

      {/* Detailed Table */}
      <div className="glass-card" style={{ padding: "1.5rem", marginBottom: "1.5rem" }}>
        <h3 style={{ fontSize: "0.95rem", fontWeight: 600, marginBottom: "1.25rem" }}>Mean per controller</h3>
        <div style={{ overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Controller</th>
                <th>Lag (s)</th>
                <th>Trips Completed</th>
                <th>Backlog</th>
                <th>Never Inserted</th>
                <th>Avg Wait (s)</th>
                <th>Avg Queue</th>
                <th>Throughput</th>
                <th>Mean Speed</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => {
                const starving = baseline ? isStarving(r.trips_completed, baseline.trips_completed) : false;
                const waitDelta = baseline && r.controller !== baseline.controller
                  ? metricDelta("mean_waiting_time", r.mean_waiting_time, baseline.mean_waiting_time, starving)
                  : null;
                return (
                <tr key={r.controller}>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ width: 10, height: 10, borderRadius: "50%", background: CTRL_COLORS[r.controller] ?? "#64748b", display: "inline-block" }} />
                      <span style={{ fontWeight: 600 }}>{CTRL_LABELS[r.controller] ?? r.controller}</span>
                    </div>
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{fmt(r.lag_per_scheduled_car_s, 1)}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{fmtInt(r.trips_completed)}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{fmtInt(r.backlog)}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{fmtInt(r.never_inserted)}</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--accent-cyan)" }}>
                    {fmt(r.mean_waiting_time, 2)}
                    {waitDelta && waitDelta.diff !== 0 && (
                      <span
                        className={`improvement-tag ${waitDelta.tone}`}
                        title={waitDelta.suspect ? "Lower wait only because this controller completes fewer trips (starvation)." : undefined}
                        style={{ marginLeft: 6, fontSize: "0.68rem" }}
                      >
                        {waitDelta.improved ? "-" : "+"}{Math.abs(waitDelta.diff).toFixed(1)}%
                        <span className="sr-only">{waitDelta.suspect ? " (starvation artifact)" : waitDelta.improved ? " (better)" : " (worse)"}</span>
                      </span>
                    )}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{fmt(r.mean_queue, 3)}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{isNum(r.throughput) ? `${fmt(r.throughput * 100, 1)}%` : "—"}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{fmt(r.mean_speed, 2)}</td>
                </tr>
                );
              })}
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
              <MetricRow key={key} label={key} entry={r} baseline={baseline?.controller !== r.controller ? baseline : undefined} starving={baseline ? isStarving(r.trips_completed, baseline.trips_completed) : false} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};
