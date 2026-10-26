import React, { useEffect, useState } from "react";
import type { ShowcaseMetrics, TenSeedResult } from "../../api/client";
import { measureMean } from "../../showcase/measures";

export interface BenchController {
  id: string;
  label: string;
  short: string;
  color: string;
  kind: "baseline" | "model";
}

interface Props {
  controllers: BenchController[];
  metrics: Record<string, ShowcaseMetrics | null>;
  tenSeed: TenSeedResult | null;
  /** How many of the four sections have started to dissolve in. */
  shown: number;
}

const perSeed = (m: ShowcaseMetrics | null, key: string): number[] => m?.measures.find((x) => x.key === key)?.per_seed ?? [];

// ── 1 · Verdict line + lag strip ─────────────────────────────────────────────
const Verdict: React.FC<{ tenSeed: TenSeedResult | null; modelIds: string[] }> = ({ tenSeed, modelIds }) => {
  if (!tenSeed) return null;
  const models = Object.entries(tenSeed.models).filter(([id]) => modelIds.includes(id));
  return (
    <ul className="sc-verdicts">
      {models.map(([id, m]) => {
        const beaten = m.measures.filter((x) => x.verdict === "Beaten").length;
        const worse = m.measures.filter((x) => x.verdict.startsWith("Worse")).length;
        return (
          <li key={id}>
            <span className="sc-v-name">{m.label.replace(" (final)", "")} against SUMO actuated</span>
            <span className="sc-v-big"><b>{beaten}</b> of {m.measures.length} measures beaten</span>
            <span className="sc-verdict-sub">{worse === 0 ? "none significantly worse" : `worse on ${worse}`} · {tenSeed.seeds.length} paired seeds</span>
          </li>
        );
      })}
    </ul>
  );
};

const LagStrip: React.FC<{ controllers: BenchController[]; metrics: Record<string, ShowcaseMetrics | null> }> = ({ controllers, metrics }) => {
  const key = "lag_per_scheduled_car_s";
  const all = controllers.flatMap((c) => perSeed(metrics[c.id], key));
  if (all.length === 0) return null;
  const lo = Math.floor((Math.min(...all) - 8) / 10) * 10;
  const hi = Math.ceil((Math.max(...all) + 8) / 10) * 10;
  const L = 176, R = 700, rowH = 58, top = 14;
  const x = (v: number) => L + ((v - lo) / (hi - lo)) * (R - L);
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / 20) * 20; t <= hi; t += 20) ticks.push(t);
  const base = measureMean(metrics["sumo_actuated"], key);
  const H = top + controllers.length * rowH + 52;
  return (
    <svg viewBox={`0 0 760 ${H}`} className="sc-chart" role="img" aria-label="Lag per scheduled car for every seed and controller; lower is better">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top - 4} y2={H - 38} className="sc-grid" />
          <text x={x(t)} y={H - 24} textAnchor="middle" className="sc-axis">{t}</text>
        </g>
      ))}
      <text x={R} y={H - 3} textAnchor="end" className="sc-axis">seconds late per scheduled car (lower is better)</text>
      {controllers.map((c, i) => {
        const cy = top + i * rowH + rowH / 2 - 4;
        const values = perSeed(metrics[c.id], key);
        const mean = measureMean(metrics[c.id], key);
        const delta = base && mean !== null && c.id !== "sumo_actuated" ? ((mean - base) / base) * 100 : null;
        return (
          <g key={c.id}>
            <text x={0} y={cy + 4} className="sc-row-label">{c.short}</text>
            <line x1={L} x2={R} y1={cy} y2={cy} className="sc-track" />
            {values.map((v, j) => (
              <circle key={j} cx={x(v)} cy={cy} r={4.2} fill={c.color} opacity={0.82} className="sc-pop" style={{ animationDelay: `${i * 160 + j * 45}ms` }} />
            ))}
            {mean !== null && (
              <g className="sc-pop" style={{ animationDelay: `${i * 160 + 520}ms` }}>
                <line x1={x(mean)} x2={x(mean)} y1={cy - 13} y2={cy + 13} stroke={c.color} strokeWidth={2.4} />
                <text x={x(mean)} y={cy - 19} textAnchor="middle" className="sc-mean" fill={c.color}>{mean.toFixed(1)}</text>
              </g>
            )}
            {delta !== null && <text x={760} y={cy + 4} textAnchor="end" className={`sc-svg-delta ${delta < 0 ? "good" : "bad"}`}>{delta > 0 ? "+" : ""}{delta.toFixed(1)}%</text>}
          </g>
        );
      })}
    </svg>
  );
};

// ── 2 · Small multiples ──────────────────────────────────────────────────────
const MINI: Array<{ key: string; title: string; fmt: (v: number) => string }> = [
  { key: "trips_completed", title: "Trips completed", fmt: (v) => Math.round(v).toLocaleString("en-US") },
  { key: "mean_waiting_time", title: "Mean waiting time (s)", fmt: (v) => v.toFixed(2) },
  { key: "backlog", title: "Backlog at 08:00 (cars)", fmt: (v) => Math.round(v).toLocaleString("en-US") },
  { key: "mean_queue", title: "Mean queue", fmt: (v) => v.toFixed(3) },
];

const MiniBars: React.FC<{ controllers: BenchController[]; metrics: Record<string, ShowcaseMetrics | null> }> = ({ controllers, metrics }) => {
  // bars grow from zero once the section has mounted
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setVisible(true));
    return () => cancelAnimationFrame(id);
  }, []);
  return (
  <div className="sc-minis">
    {MINI.map((spec) => {
      const values = controllers.map((c) => measureMean(metrics[c.id], spec.key));
      const top = Math.max(...values.map((v) => v ?? 0), 1e-9);
      const base = measureMean(metrics["sumo_actuated"], spec.key);
      const better = metrics["sumo_actuated"]?.measures.find((m) => m.key === spec.key)?.better;
      return (
        <figure key={spec.key} className="sc-mini">
          <figcaption>{spec.title}</figcaption>
          {controllers.map((c, i) => {
            const v = values[i];
            const pct = base && v !== null && c.id !== "sumo_actuated" ? ((v - base) / base) * 100 : null;
            const good = pct !== null && better ? (better === "lower" ? pct < 0 : pct > 0) : false;
            return (
              <div className="sc-mini-row" key={c.id}>
                <span className="sc-mini-name">{c.short}</span>
                <span className="sc-mini-track">
                  <span className="sc-mini-bar" style={{ width: visible && v !== null ? `${(v / top) * 100}%` : "0%", background: c.color, transitionDelay: `${i * 90}ms` }} />
                </span>
                <span className="sc-mini-val">{v === null ? "—" : spec.fmt(v)}</span>
                <span className={`sc-mini-delta ${pct === null ? "" : good ? "good" : "bad"}`}>{pct === null ? "" : `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`}</span>
              </div>
            );
          })}
        </figure>
      );
    })}
  </div>
  );
};

// ── 3 · The same gap on every seed ───────────────────────────────────────────
const SeedLines: React.FC<{ controllers: BenchController[]; metrics: Record<string, ShowcaseMetrics | null> }> = ({ controllers, metrics }) => {
  const key = "lag_per_scheduled_car_s";
  const series = controllers.map((c) => ({ c, v: perSeed(metrics[c.id], key) })).filter((s) => s.v.length > 0);
  if (series.length === 0) return null;
  const seeds = metrics["sumo_actuated"]?.seeds ?? metrics[series[0].c.id]?.seeds ?? [];
  const n = Math.max(...series.map((s) => s.v.length));
  const all = series.flatMap((s) => s.v);
  const lo = Math.floor((Math.min(...all) - 6) / 10) * 10;
  const hi = Math.ceil((Math.max(...all) + 6) / 10) * 10;
  const L = 44, R = 740, T = 14, B = 232;
  const x = (i: number) => L + (n === 1 ? 0 : (i / (n - 1)) * (R - L));
  const y = (v: number) => B - ((v - lo) / (hi - lo)) * (B - T);
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / 20) * 20; t <= hi; t += 20) ticks.push(t);
  return (
    <svg viewBox="0 0 760 262" className="sc-chart" role="img" aria-label="Lag per scheduled car on each of the evaluation seeds">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={R} y1={y(t)} y2={y(t)} className="sc-grid" />
          <text x={L - 8} y={y(t) + 3} textAnchor="end" className="sc-axis">{t}</text>
        </g>
      ))}
      {seeds.map((s, i) => <text key={s} x={x(i)} y={254} textAnchor="middle" className="sc-axis">{s}</text>)}
      <text x={L - 8} y={8} textAnchor="end" className="sc-axis">s</text>
      {series.map(({ c, v }) => (
        <g key={c.id}>
          <polyline points={v.map((val, i) => `${x(i)},${y(val)}`).join(" ")} fill="none" stroke={c.color} strokeWidth={2} strokeLinejoin="round" className="sc-draw" />
          {v.map((val, i) => <circle key={i} cx={x(i)} cy={y(val)} r={3} fill={c.color} />)}
        </g>
      ))}
    </svg>
  );
};

// ── 4 · Paired test table (compact) ──────────────────────────────────────────
function fmtMeasure(key: string, v: number): string {
  if (["backlog", "trips_completed", "never_inserted", "overdue_cars"].includes(key)) return Math.round(v).toLocaleString("en-US");
  if (["mean_queue", "due_completion_ratio", "throughput"].includes(key)) return v.toFixed(3);
  if (["mean_waiting_time", "mean_speed"].includes(key)) return v.toFixed(2);
  return v.toFixed(1);
}

function verdictChip(verdict: string): { text: string; tone: "strong" | "soft" | "level" | "bad" } {
  if (verdict === "Beaten") return { text: "beaten", tone: "strong" };
  if (verdict.startsWith("Better (interval")) return { text: "better", tone: "soft" };
  if (verdict.startsWith("Worse")) return { text: "worse", tone: "bad" };
  return { text: "level", tone: "level" };
}

const PairedTable: React.FC<{ tenSeed: TenSeedResult | null; modelIds: string[] }> = ({ tenSeed, modelIds }) => {
  if (!tenSeed) return <p className="sc-note">The paired test is not available (run scripts/report_eval.py).</p>;
  const models = modelIds.filter((id) => tenSeed.models[id]);
  if (models.length === 0) return null;
  const rows = tenSeed.models[models[0]].measures;
  return (
    <div className="sc-tablewrap">
      <table className="sc-table">
        <thead>
          <tr>
            <th>Measure</th><th>SUMO actuated</th>
            {models.map((id) => <th key={id} colSpan={1}>{tenSeed.models[id].label.replace(" (final)", "")}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <td>{r.label}</td>
              <td className="num">{fmtMeasure(r.key, r.baseline_mean)}</td>
              {models.map((id) => {
                const x = tenSeed.models[id].measures.find((m) => m.key === r.key);
                if (!x) return <td key={id}>—</td>;
                const good = x.better === "lower" ? x.pct_diff < 0 : x.pct_diff > 0;
                const chip = verdictChip(x.verdict);
                return (
                  <td key={id}>
                    <span className="num">{fmtMeasure(r.key, x.model_mean)}</span>
                    <span className={`sc-delta ${good ? "good" : "bad"}`}>{x.pct_diff > 0 ? "+" : ""}{x.pct_diff.toFixed(1)}%</span>
                    <span className={`sc-vchip ${chip.tone}`} title={`${x.verdict} - won ${x.seeds_won} of ${x.n_seeds} seeds`}>{chip.text} {x.seeds_won}/{x.n_seeds}</span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="sc-note">Same traffic for model and baseline on every seed. "beaten" = the 95% interval of the per-seed difference excludes zero and the model wins on at least 9 of 10 seeds (exact sign test); "better" = interval only; "level" = not distinguishable. The small number is the seeds won.</p>
    </div>
  );
};

const WHY: Array<[string, string]> = [
  ["The learner now actually learns.", "The part that decides what to do was almost switched off by the part that judges the situation. Each now learns at its own pace."],
  ["A reward that cannot be gamed.", "A small cost every 5 s for each stopped car, each car waiting to enter the city, and each light change, instead of 'waiting time went down'."],
  ["Cleaner inputs.", "Two numbers stuck at their maximum were dropped; which phase is green, how long it has run against its limits, and cars waiting per phase were added."],
  ["It sees the whole queue.", "Stop lines hold about two cars per lane, but the real queue stands 100 to 200 m back; it now counts stopped cars up to 300 m behind."],
  ["It sees cars waiting outside the city.", "Counting cars not yet inserted, then 10 to 20 more training rounds, closed the last gap."],
];

export const Benchmarks: React.FC<Props> = ({ controllers, metrics, tenSeed, shown }) => (
  <div className="sc-bench">
    {shown >= 1 && (
      <section className="sc-bsec">
        <h2>Benchmark</h2>
        <p className="sc-lede">Ten complete simulation hours (07:00 to 08:00), each with its own random seed, replayed through all four controllers on identical traffic.</p>
        <Verdict tenSeed={tenSeed} modelIds={controllers.filter((c) => c.kind === "model").map((c) => c.id)} />
        <div className="sc-panelbox">
          <h3>Lag per scheduled car <small>every dot is one seed, the bar is the mean</small></h3>
          <LagStrip controllers={controllers} metrics={metrics} />
        </div>
      </section>
    )}
    {shown >= 2 && (
      <section className="sc-bsec">
        <div className="sc-panelbox">
          <h3>The same traffic, four ways to run the lights <small>mean over all seeds; the % is against SUMO actuated</small></h3>
          <MiniBars controllers={controllers} metrics={metrics} />
        </div>
      </section>
    )}
    {shown >= 3 && (
      <section className="sc-bsec">
        <div className="sc-panelbox">
          <h3>Not a lucky seed <small>lag per scheduled car, seed by seed</small></h3>
          <SeedLines controllers={controllers} metrics={metrics} />
          <ul className="sc-legend">
            {controllers.map((c) => <li key={c.id}><span className="sc-dot" style={{ background: c.color }} /> {c.label}</li>)}
          </ul>
        </div>
      </section>
    )}
    {shown >= 4 && (
      <section className="sc-bsec">
        <div className="sc-panelbox">
          <h3>Paired test against SUMO actuated <small>95% interval of the per-seed difference, and the exact sign test</small></h3>
          <PairedTable tenSeed={tenSeed} modelIds={controllers.filter((c) => c.kind === "model").map((c) => c.id)} />
        </div>
        <div className="sc-panelbox">
          <h3>What made the difference</h3>
          <ol className="sc-why">
            {WHY.map(([lead, rest], i) => (
              <li key={lead}><span className="sc-why-n">{i + 1}</span><div><b>{lead}</b> {rest}</div></li>
            ))}
          </ol>
        </div>
      </section>
    )}
  </div>
);
