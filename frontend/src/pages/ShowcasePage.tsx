import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Play, Pause, RotateCcw, FastForward, AlertTriangle } from "lucide-react";
import { api, isAbortError } from "../api/client";
import type { NetworkGeoJSON, ShowcaseConfig, ShowcaseMetrics, TenSeedResult } from "../api/client";
import { SimPanel, type PanelHandle } from "../components/showcase/SimPanel";
import { ResultCard } from "../components/showcase/ResultCard";
import { measureMean, pctDiff } from "../showcase/measures";
import { Benchmarks, type BenchController } from "../components/showcase/Benchmarks";
import { SumoLauncher } from "../components/showcase/SumoLauncher";
import { buildFrameSet, formatClock, type FrameSet } from "../showcase/frames";
import { computeBounds, type Bounds } from "../showcase/mapView";
import { buildSchedule, revealAt, simTimeAt, type RevealState } from "../showcase/schedule";
import { CONTROLLER_HEX, controllerShortLabel } from "../constants/controllers";
import "../styles/showcase.css";

type Phase = "idle" | "ready" | "running" | "error";

interface Loaded {
  config: ShowcaseConfig;
  geo: NetworkGeoJSON;
  bounds: Bounds;
  order: string[];
  frames: Record<string, FrameSet>;
  metrics: Record<string, ShowcaseMetrics | null>;
  tenSeed: TenSeedResult | null;
}

interface Progress {
  map: boolean;
  runs: number;
  results: number;
  total: number;
}

const NO_REVEAL: RevealState = { slotsVisible: false, cardsShown: 0, reading: null, metricsLoaded: false, deltasShown: false, benchShown: 0, complete: false };
const sameReveal = (a: RevealState, b: RevealState) =>
  a.slotsVisible === b.slotsVisible && a.cardsShown === b.cardsShown && a.reading === b.reading && a.metricsLoaded === b.metricsLoaded &&
  a.deltasShown === b.deltasShown && a.benchShown === b.benchShown && a.complete === b.complete;

/** Fetch everything the showcase needs; progress is reported as the pieces arrive (not simulated). */
async function loadShowcase(signal: AbortSignal, onProgress: (patch: Partial<Progress>) => void): Promise<Loaded> {
  const config = await api.getShowcaseConfig(signal);
  const order = config.controllers.filter((c) => c.has_recording).map((c) => c.id);
  if (order.length === 0) throw new Error("No recorded runs were found. Run scripts/evaluate.py first.");
  onProgress({ total: order.length });
  const geoPromise = api.getNetworkGeoJSON().then((g) => { onProgress({ map: true }); return g; });
  let runs = 0;
  let results = 0;
  const framePromises = order.map((id) =>
    api.getShowcaseFrames(id, config.start, config.end, config.seed, signal).then((dto) => {
      onProgress({ runs: ++runs });
      return [id, buildFrameSet(dto)] as const;
    }),
  );
  const metricPromises = order.map((id) =>
    api.getShowcaseMetrics(id, signal)
      .catch((e: unknown) => { if (isAbortError(e)) throw e; return null; })
      .then((m) => { onProgress({ results: ++results }); return [id, m] as const; }),
  );
  const tenSeedPromise = api.getTenSeed(signal).catch((e: unknown) => { if (isAbortError(e)) throw e; return null; });
  const [geo, frames, metrics, tenSeed] = await Promise.all([geoPromise, Promise.all(framePromises), Promise.all(metricPromises), tenSeedPromise]);
  return {
    config, geo, bounds: computeBounds(geo), order,
    frames: Object.fromEntries(frames), metrics: Object.fromEntries(metrics), tenSeed,
  };
}

const Delta: React.FC<{ value: number | null; lowerIsBetter: boolean; label: string }> = ({ value, lowerIsBetter, label }) => {
  if (value === null) return null;
  const good = lowerIsBetter ? value < 0 : value > 0;
  return <span className="sc-pill"><em>{label}</em><b className={good ? "good" : "bad"}>{value > 0 ? "+" : ""}{value.toFixed(1)}%</b></span>;
};

export const ShowcasePage: React.FC = () => {
  const [phase, setPhase] = useState<Phase>("idle");
  const [data, setData] = useState<Loaded | null>(null);
  const [progress, setProgress] = useState<Progress>({ map: false, runs: 0, results: 0, total: 4 });
  const [error, setError] = useState<string | null>(null);
  const [reveal, setReveal] = useState<RevealState>(NO_REVEAL);
  const [playing, setPlaying] = useState(true);
  const [attempt, setAttempt] = useState(0);
  const wantStart = useRef(false);
  const [startRequested, setStartRequested] = useState(false);

  // Fetch in the background as soon as the page opens, so the click starts the show immediately.
  useEffect(() => {
    const ctrl = new AbortController();
    loadShowcase(ctrl.signal, (patch) => { if (!ctrl.signal.aborted) setProgress((p) => ({ ...p, ...patch })); })
      .then((loaded) => {
        if (ctrl.signal.aborted) return;
        setData(loaded);
        if (wantStart.current) { wantStart.current = false; setStartRequested(false); setPhase("running"); } else setPhase("ready");
      })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted || isAbortError(e)) return;
        wantStart.current = false;
        setStartRequested(false);
        setError(e instanceof Error ? e.message.replace(/^API [^:]+: /, "") : "Could not load the showcase data.");
        setPhase("error");
      });
    return () => ctrl.abort();
  }, [attempt]);

  const retry = useCallback(() => {
    setError(null);
    setPhase("idle");
    setProgress({ map: false, runs: 0, results: 0, total: 4 });
    setAttempt((a) => a + 1);
  }, []);

  const schedule = useMemo(
    () => (data ? buildSchedule(data.config.end - data.config.start, data.config.speed, data.order.length) : null),
    [data],
  );

  // ── animation loop: one master clock drives every panel and the reveal ─────
  const panels = useRef(new Map<string, PanelHandle>());
  const register = useCallback((id: string, handle: PanelHandle | null) => {
    if (handle) panels.current.set(id, handle);
    else panels.current.delete(id);
  }, []);
  const elapsed = useRef(0);
  const playingRef = useRef(true);
  const revealRef = useRef<RevealState>(NO_REVEAL);
  const clockEl = useRef<HTMLSpanElement>(null);
  const barEl = useRef<HTMLSpanElement>(null);
  const resultsRef = useRef<HTMLElement>(null);
  const benchRef = useRef<HTMLDivElement>(null);
  const scrolled = useRef(false);

  useEffect(() => { playingRef.current = playing; }, [playing]);

  useEffect(() => {
    if (phase !== "running" || !data || !schedule) return;
    let raf = 0;
    let last = performance.now();
    let lastSim = -1;
    let lastDraw = 0;
    let lastDom = 0;
    const { start, end, speed } = data.config;
    const perf = ((window as unknown as { __showcasePerf?: { drawMs: number } }).__showcasePerf ??= { drawMs: 0 });
    const loop = (now: number) => {
      const dt = Math.min(now - last, 100); // a hidden tab must not make the show jump ahead
      last = now;
      if (playingRef.current) elapsed.current += dt;
      const sim = simTimeAt(elapsed.current, start, end, speed);
      if (sim !== lastSim || now - lastDraw > 400) {
        lastSim = sim;
        lastDraw = now;
        const t0 = performance.now();
        panels.current.forEach((p) => p.draw(sim));
        perf.drawMs = perf.drawMs * 0.9 + (performance.now() - t0) * 0.1; // smoothed, readable from the console: window.__showcasePerf
      }
      if (now - lastDom > 100) {
        lastDom = now;
        if (clockEl.current) clockEl.current.textContent = formatClock(sim);
        if (barEl.current) barEl.current.style.transform = `scaleX(${((sim - start) / Math.max(1, end - start)).toFixed(4)})`;
      }
      const next = revealAt(elapsed.current, schedule);
      if (!sameReveal(next, revealRef.current)) {
        revealRef.current = next;
        setReveal(next);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [phase, data, schedule]);

  const start = useCallback(() => {
    elapsed.current = 0;
    revealRef.current = NO_REVEAL;
    scrolled.current = false;
    setReveal(NO_REVEAL);
    setPlaying(true);
    playingRef.current = true;
    if (data) setPhase("running");
    else { wantStart.current = true; setStartRequested(true); }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [data]);

  const restart = useCallback(() => {
    elapsed.current = 0;
    revealRef.current = NO_REVEAL;
    scrolled.current = false;
    setReveal(NO_REVEAL);
    setPlaying(true);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const skip = useCallback(() => {
    if (schedule) elapsed.current = schedule.completeAtMs;
  }, [schedule]);

  // Space pauses / resumes while the show runs.
  useEffect(() => {
    if (phase !== "running") return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (e.code !== "Space" || (t && /^(BUTTON|INPUT|TEXTAREA|SELECT|A)$/.test(t.tagName))) return;
      e.preventDefault();
      setPlaying((p) => !p);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase]);

  // When the benchmarks begin, bring them into view once (the audience watches the dissolve).
  useEffect(() => {
    if (reveal.benchShown >= 1 && !scrolled.current) {
      scrolled.current = true;
      const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
      benchRef.current?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    }
  }, [reveal.benchShown]);

  const colorOf = (id: string) => CONTROLLER_HEX[id] ?? "#64748b";
  const controllers: BenchController[] = useMemo(
    () => (data?.order ?? []).map((id) => ({
      id, label: data!.config.controllers.find((c) => c.id === id)?.label ?? id, short: controllerShortLabel(id),
      color: colorOf(id), kind: data!.config.controllers.find((c) => c.id === id)?.kind ?? "baseline",
    })),
    [data],
  );
  const maxStopped = useMemo(() => Math.max(1, ...Object.values(data?.frames ?? {}).map((f) => Math.max(0, ...Array.from(f.stopped)))), [data]);
  const allColors = useMemo(() => Object.fromEntries(Object.keys(CONTROLLER_HEX).map((k) => [k, CONTROLLER_HEX[k]])), []);

  // ── idle / loading / error ──────────────────────────────────────────────
  if (phase !== "running" || !data || !schedule) {
    const loading = phase === "idle";
    const pct = Math.round(((progress.map ? 1 : 0) + progress.runs + progress.results) / Math.max(1, 1 + 2 * progress.total) * 100);
    return (
      <div className="sc-root">
        <section className="sc-hero">
          <p className="sc-eyebrow">CityMind AI · Ingolstadt, 98 traffic signals</p>
          <h1>Watch one shared policy<br />run a city.</h1>
          <p className="sc-sub">Four controllers, the same ten minutes of rush hour, replayed side by side at 10x, then every number measured over ten full simulation hours.</p>
          {phase === "error" ? (
            <div className="sc-fail" role="alert">
              <AlertTriangle size={18} aria-hidden="true" />
              <div><b>The showcase data could not be loaded.</b><span>{error}</span></div>
              <button type="button" className="sc-btn" onClick={retry}>Try again</button>
            </div>
          ) : (
            <>
              <button type="button" className="sc-cta" onClick={start} aria-busy={loading}>
                <span className="sc-cta-icon" aria-hidden="true"><Play size={20} fill="currentColor" /></span>
                {startRequested ? "Starting as soon as the runs are loaded…" : "Run the showcase"}
              </button>
              <div className="sc-loading" aria-live="polite">
                <span className="sc-loadbar"><span style={{ width: `${phase === "ready" ? 100 : pct}%` }} /></span>
                <span>
                  {phase === "ready"
                    ? `Ready · ${data?.order.length ?? 4} recorded runs · results of ${data?.config.n_seeds ?? 10} seeds`
                    : `${progress.map ? "Map ready" : "Loading the map"} · runs ${progress.runs}/${progress.total} · results ${progress.results}/${progress.total}`}
                </span>
              </div>
            </>
          )}
          <p className="sc-foot">Replays recorded evaluation runs (seed 42) and reads stored results; nothing is re-simulated for this demo. Use the SUMO buttons below to watch a live simulation. How does the model decide? <Link to="/model">See inside it →</Link></p>
        </section>
        <SumoLauncher colors={allColors} />
      </div>
    );
  }

  const baseline = data.metrics["sumo_actuated"] ?? null;
  const readingName = reveal.reading !== null ? controllers[reveal.reading]?.label : null;
  const statusLine = reveal.metricsLoaded
    ? `All stored results loaded · ${controllers.length} controllers × ${data.config.n_seeds} seeds`
    : readingName ? `Reading stored results · ${readingName}`
    : reveal.cardsShown > 0 && controllers[reveal.cardsShown] ? `Locating stored results · ${controllers[reveal.cardsShown].label}`
    : reveal.slotsVisible ? "Locating the stored evaluation results…" : "";
  const modelDeltas = controllers.filter((c) => c.kind === "model").map((c) => {
    const m = data.metrics[c.id];
    const pct = (key: string) => pctDiff(measureMean(m, key), measureMean(baseline, key));
    return { c, lag: pct("lag_per_scheduled_car_s"), trips: pct("trips_completed"), wait: pct("mean_waiting_time"), backlog: pct("backlog") };
  });

  return (
    <div className="sc-root running">
      <header className="sc-stagebar">
        <div className="sc-clock">
          <span className="sc-live-dot" aria-hidden="true" />
          <span className="sc-time" ref={clockEl} aria-label="Simulated time">{formatClock(data.config.start)}</span>
          <span className="sc-meta">{data.config.speed}x · recorded run · seed {data.config.seed} · 07:20 to 07:30</span>
        </div>
        <div className="sc-progress" aria-hidden="true"><span ref={barEl} /></div>
        <div className="sc-controls">
          <button type="button" className="sc-icon-btn" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause (space)" : "Resume (space)"} title="Space">
            {playing ? <Pause size={16} aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
          </button>
          <button type="button" className="sc-icon-btn" onClick={restart} aria-label="Restart the showcase" title="Restart"><RotateCcw size={16} aria-hidden="true" /></button>
          <button type="button" className="sc-icon-btn" onClick={skip} aria-label="Skip to the results" title="Skip to results"><FastForward size={16} aria-hidden="true" /></button>
        </div>
      </header>

      <div className="sc-grid4">
        {controllers.map((c) => (
          <SimPanel
            key={c.id} id={c.id} label={c.label} kind={c.kind} color={c.color}
            frames={data.frames[c.id]} geo={data.geo} bounds={data.bounds} maxStopped={maxStopped} register={register}
          />
        ))}
      </div>

      <section className="sc-results" ref={resultsRef} aria-label="Evaluation results">
        <header className={`sc-results-head${reveal.slotsVisible ? " in" : ""}`}>
          <h2>Evaluation results</h2>
          <p className="sc-status" role="status">
            {statusLine && <span className={`sc-led${reveal.metricsLoaded ? " done" : ""}`} aria-hidden="true" />}{statusLine}
          </p>
        </header>
        <div className="sc-grid4">
          {controllers.map((c, i) => (
            <ResultCard
              key={c.id} label={c.label} kind={c.kind} color={c.color}
              slotVisible={reveal.slotsVisible} shown={reveal.cardsShown > i} reading={reveal.reading === i}
              metrics={data.metrics[c.id] ?? null} baseline={baseline} isBaseline={c.id === "sumo_actuated"}
            />
          ))}
        </div>
        <div className={`sc-deltas${reveal.deltasShown ? " in" : ""}`} aria-hidden={!reveal.deltasShown}>
          {modelDeltas.map(({ c, lag, trips, wait, backlog }) => (
            <p key={c.id}>
              <span className="sc-dot" style={{ background: c.color }} aria-hidden="true" />
              <strong>{c.short}</strong> against SUMO actuated:
              <Delta value={lag} lowerIsBetter label="lag" />
              <Delta value={trips} lowerIsBetter={false} label="trips" />
              <Delta value={wait} lowerIsBetter label="wait" />
              <Delta value={backlog} lowerIsBetter label="backlog" />
            </p>
          ))}
        </div>
      </section>

      <div ref={benchRef} className="sc-bench-anchor">
        {reveal.metricsLoaded && <Benchmarks controllers={controllers} metrics={data.metrics} tenSeed={data.tenSeed} shown={reveal.benchShown} />}
      </div>

      {reveal.complete && (
        <div className="sc-done">
          <button type="button" className="sc-btn ghost" onClick={restart}><RotateCcw size={15} aria-hidden="true" /> Run the showcase again</button>
        </div>
      )}
      <SumoLauncher colors={allColors} />
    </div>
  );
};
