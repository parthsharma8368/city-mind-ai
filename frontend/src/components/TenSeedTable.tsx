import React, { useEffect, useState } from "react";
import { api, isAbortError } from "../api/client";
import type { TenSeedMeasure, TenSeedResult } from "../api/client";
import { fmt } from "../utils/format";

/** Per-measure formatting of the baseline / model means (the API sends raw numbers). */
function fmtMean(key: string, v: number): string {
  if (key === "backlog" || key === "trips_completed" || key === "never_inserted" || key === "overdue_cars") return Math.round(v).toLocaleString("en-US");
  if (key === "mean_queue" || key === "due_completion_ratio" || key === "throughput") return fmt(v, 3);
  if (key === "mean_waiting_time" || key === "mean_speed") return fmt(v, 2);
  return fmt(v, 1);
}

const TONE: Record<string, string> = {
  Beaten: "positive",
  "Better (interval only)": "positive",
  Worse: "negative",
  "Worse (interval only)": "negative",
};

function verdictText(m: TenSeedMeasure): string {
  return m.verdict === "Beaten" ? "Beaten" : m.verdict;
}

/**
 * The paired 10-seed test: each final model against SUMO actuated on the same traffic, every measure.
 * "Beaten" = the 95% interval of the per-seed difference excludes zero AND the sign test is significant.
 */
export const TenSeedTable: React.FC<{ keys?: string[]; models?: string[] }> = ({ keys, models = ["ippo_A", "ippo_B"] }) => {
  const [data, setData] = useState<TenSeedResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    api.getTenSeed(ctrl.signal)
      .then((d) => { if (!ctrl.signal.aborted) setData(d); })
      .catch((e: Error) => { if (!ctrl.signal.aborted && !isAbortError(e)) setError(e.message); });
    return () => ctrl.abort();
  }, []);

  if (error) return <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>10-seed report unavailable: {error}</p>;
  if (!data) return <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Loading the 10-seed test…</p>;

  const present = models.filter((m) => data.models[m]);
  if (present.length === 0) return <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>No 10-seed report found. Run scripts/report_eval.py.</p>;
  const rows = data.models[present[0]].measures.filter((m) => !keys || keys.includes(m.key));
  const byKey = (model: string, key: string) => data.models[model].measures.find((m) => m.key === key);

  return (
    <div style={{ overflowX: "auto" }}>
      <table className="data-table">
        <thead>
          <tr>
            <th>Measure</th>
            <th>SUMO actuated</th>
            {present.map((m) => <th key={m}>{data.models[m].label}</th>)}
            <th>Verdict ({data.seeds.length} seeds)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <td style={{ fontWeight: 600 }}>{r.label}</td>
              <td style={{ fontFamily: "var(--font-mono)" }}>{fmtMean(r.key, r.baseline_mean)}</td>
              {present.map((m) => {
                const x = byKey(m, r.key);
                return (
                  <td key={m} style={{ fontFamily: "var(--font-mono)" }}>
                    {x ? fmtMean(r.key, x.model_mean) : "—"}
                    {x && (
                      <span className={`improvement-tag ${(x.better === "lower" ? x.pct_diff < 0 : x.pct_diff > 0) ? "positive" : "negative"}`} style={{ marginLeft: 6, fontSize: "0.68rem" }}>
                        {x.pct_diff > 0 ? "+" : ""}{x.pct_diff.toFixed(1)}%
                      </span>
                    )}
                  </td>
                );
              })}
              <td style={{ fontSize: "0.78rem" }}>
                {present.map((m) => {
                  const x = byKey(m, r.key);
                  if (!x) return null;
                  return (
                    <div key={m} style={{ display: "flex", gap: 6, alignItems: "center" }}>
                      <span style={{ color: "var(--text-muted)", width: "3.4em" }}>{data.models[m].label.replace(" (final)", "")}</span>
                      <span className={`improvement-tag ${TONE[x.verdict] ?? "neutral"}`} style={{ fontSize: "0.68rem" }}>{verdictText(x)}</span>
                      <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{x.seeds_won}/{x.n_seeds}</span>
                    </div>
                  );
                })}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "0.6rem" }}>
        Same traffic for model and baseline on every seed. “Beaten” = the 95 % interval of the per-seed difference excludes zero and the
        model wins on at least 9 of 10 seeds (sign test). The small number is the seeds won.
      </p>
    </div>
  );
};
