import React, { useEffect, useMemo, useState } from "react";
import type { ShowcaseMetrics } from "../../api/client";
import { measureMean } from "../../showcase/measures";
import { COUNT_UP_MS, easeOutCubic } from "../../showcase/schedule";

interface KpiSpec {
  key: string;
  label: string;
  unit?: string;
  decimals: number;
  /** Shown as a percentage (value x 100). */
  percent?: boolean;
}

const KPIS: KpiSpec[] = [
  { key: "lag_per_scheduled_car_s", label: "Lag per scheduled car", unit: "s", decimals: 1 },
  { key: "trips_completed", label: "Trips completed", decimals: 0 },
  { key: "mean_waiting_time", label: "Mean wait", unit: "s", decimals: 2 },
  { key: "backlog", label: "Backlog", unit: "cars", decimals: 0 },
  { key: "never_inserted", label: "Never inserted", unit: "cars", decimals: 0 },
  { key: "due_completion_ratio", label: "Due completion", decimals: 1, percent: true },
];

function formatValue(spec: KpiSpec, v: number): string {
  const x = spec.percent ? v * 100 : v;
  return x.toLocaleString("en-US", { minimumFractionDigits: spec.decimals, maximumFractionDigits: spec.decimals });
}

/** Counts from 0 to `target` once `active` turns true (ease-out), then holds. */
function useCountUp(target: number | null, active: boolean): number {
  const [value, setValue] = useState(0);
  const reduce = useMemo(() => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches, []);
  useEffect(() => {
    if (!active || target === null || reduce) return;
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const p = (now - t0) / COUNT_UP_MS;
      setValue(target * easeOutCubic(p));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, active, reduce]);
  if (!active || target === null) return 0;
  return reduce ? target : value;
}

const Kpi: React.FC<{ spec: KpiSpec; metrics: ShowcaseMetrics | null; baseline: ShowcaseMetrics | null; active: boolean; isBaseline: boolean }> = ({
  spec, metrics, baseline, active, isBaseline,
}) => {
  const target = measureMean(metrics, spec.key);
  const value = useCountUp(target, active);
  const base = measureMean(baseline, spec.key);
  const better = metrics?.measures.find((x) => x.key === spec.key)?.better;
  let delta: { text: string; good: boolean } | null = null;
  if (!isBaseline && target !== null && base !== null && base !== 0 && better) {
    const pct = ((target - base) / Math.abs(base)) * 100;
    delta = { text: `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`, good: better === "lower" ? pct < 0 : pct > 0 };
  }
  return (
    <div className="sc-kpi">
      <dt>{spec.label}</dt>
      <dd>
        <span className="sc-kpi-value">{target === null ? "—" : formatValue(spec, value)}</span>
        {spec.unit && <span className="sc-kpi-unit">{spec.unit}</span>}
        {delta && <span className={`sc-delta ${delta.good ? "good" : "bad"}`}>{delta.text}</span>}
      </dd>
    </div>
  );
};

interface Props {
  label: string;
  kind: "baseline" | "model";
  color: string;
  /** The empty outline is visible. */
  slotVisible: boolean;
  /** The stored results are dissolving in. */
  shown: boolean;
  /** This card's results are being read right now. */
  reading: boolean;
  metrics: ShowcaseMetrics | null;
  baseline: ShowcaseMetrics | null;
  isBaseline: boolean;
}

export const ResultCard: React.FC<Props> = ({ label, kind, color, slotVisible, shown, reading, metrics, baseline, isBaseline }) => (
  <section className={`sc-card${slotVisible ? " slot" : ""}${shown ? " in" : ""}${reading ? " reading" : ""}`} aria-label={`${label}: stored results`} aria-busy={reading}>
    <div className="sc-card-skeleton" aria-hidden="true">
      <span className="sc-skel w40" /><span className="sc-skel w70" /><span className="sc-skel w55" /><span className="sc-skel w62" />
    </div>
    <div className="sc-card-body" aria-hidden={!shown}>
      <header>
        <span className="sc-dot" style={{ background: color }} aria-hidden="true" />
        <h3>{label}</h3>
        <span className={`sc-chip ${kind}`}>{isBaseline ? "Baseline" : kind === "model" ? "Final model" : "Context"}</span>
      </header>
      <dl className="sc-kpis">
        {KPIS.map((spec) => (
          <Kpi key={spec.key} spec={spec} metrics={metrics} baseline={baseline} active={shown} isBaseline={isBaseline} />
        ))}
      </dl>
      <footer>{metrics ? `mean of ${metrics.n_runs} simulation seeds` : "no stored results"}</footer>
    </div>
  </section>
);
