import React, { useMemo } from "react";
import type { DecideInfo, ObserveInfo } from "./NeuralScene";
import { MAX_GREEN, MIN_GREEN, PHASES } from "../../brain/junction";

const SIGNAL = { g: "#34d399", y: "#fbbf24", r: "#f87171" } as const;

/** What the policy is looking at: the made-up junction behind the 22 numbers. */
export const JunctionCard: React.FC<{ observed: ObserveInfo | null }> = ({ observed }) => {
  const s = observed?.state;
  return (
    <section className="br-card" aria-label="The junction the networks are looking at">
      <h3>Junction now</h3>
      <p className="br-card-sub">cars waiting for each green phase</p>
      <svg viewBox="0 0 240 118" className="br-junction" role="img" aria-label="Queue length and signal colour of each phase">
        {Array.from({ length: PHASES }, (_, i) => {
          const q = s?.queues[i] ?? 0;
          const active = !!s && s.phase === i;
          const colour = !s ? "#475569" : s.transition && active ? SIGNAL.y : active && !s.transition ? SIGNAL.g : SIGNAL.r;
          const h = Math.min(1, q / 16) * 64;
          const x = 22 + i * 76;
          return (
            <g key={i}>
              <rect x={x} y={14} width={44} height={64} rx={7} fill="rgba(255,255,255,.04)" stroke="rgba(255,255,255,.08)" />
              <rect x={x} y={78 - h} width={44} height={h} rx={7} fill={colour} opacity={0.85} style={{ transition: "all .6s ease" }} />
              <text x={x + 22} y={96} textAnchor="middle" className="br-svg-num">{Math.round(q)}</text>
              <text x={x + 22} y={111} textAnchor="middle" className="br-svg-label">phase {i + 1}</text>
              <circle cx={x + 22} cy={7} r={4.5} fill={colour} style={{ filter: `drop-shadow(0 0 5px ${colour})` }} />
            </g>
          );
        })}
      </svg>
      <dl className="br-facts">
        <div><dt>time in this green</dt><dd>{s ? `${s.transition ? "clearing" : `${s.elapsed} s`}` : "—"} <small>min {MIN_GREEN} · max {MAX_GREEN}</small></dd></div>
        <div><dt>waiting to enter</dt><dd>{s ? Math.round(s.pending) : "—"} <small>cars</small></dd></div>
      </dl>
    </section>
  );
};

export const ActorCard: React.FC<{ decision: DecideInfo | null; recent: DecideInfo[]; actions: string[]; parameters: number }> = ({ decision, recent, actions, parameters }) => (
  <section className="br-card actor" aria-label="What the actor says">
    <h3><span className="br-dot actor" aria-hidden="true" />Actor says</h3>
    <p className="br-card-sub">22 → 256 → 256 → 2 · {parameters.toLocaleString("en-US")} weights</p>
    <ul className="br-bars">
      {actions.map((a, i) => {
        const p = decision?.probs[i] ?? 0;
        const chosen = !!decision && decision.action === i;
        return (
          <li key={a} className={chosen ? "chosen" : ""}>
            <span className="br-bar-name">{i === 0 ? "keep" : "advance"}</span>
            <span className="br-bar"><span style={{ transform: `scaleX(${p})` }} /></span>
            <span className="br-bar-num">{decision ? `${(p * 100).toFixed(0)}%` : "—"}</span>
          </li>
        );
      })}
    </ul>
    <p className="br-verdict" aria-live="polite">
      {!decision ? "Waiting for the first decision…"
        : decision.blocked ? "Wants to advance, but the safety layer holds the green (minimum not reached)."
        : decision.forced ? "Safety layer ends this green (maximum reached)."
        : decision.action === 1 ? "Decision: advance to the next green." : "Decision: keep this green."}
    </p>
    <div className="br-recent" aria-label="Recent decisions, oldest first">
      {recent.map((d) => (
        <i key={d.step} className={d.blocked ? "held" : d.action === 1 || d.forced ? "adv" : "keep"} title={`step ${d.step}: ${d.action === 1 ? "advance" : "keep"}`} />
      ))}
    </div>
    <p className="br-card-sub br-legend"><i className="keep" />keep <i className="adv" />advance <i className="held" />advance refused by the safety layer</p>
  </section>
);

export const CriticCard: React.FC<{ history: number[]; parameters: number }> = ({ history, parameters }) => {
  const value = history.length ? history[history.length - 1] : null;
  const points = useMemo(() => {
    if (history.length < 2) return "";
    const lo = Math.min(...history);
    const hi = Math.max(...history);
    const span = hi - lo || 1;
    return history.map((v, i) => `${((i / (history.length - 1)) * 100).toFixed(1)},${(26 - ((v - lo) / span) * 22).toFixed(1)}`).join(" ");
  }, [history]);
  return (
    <section className="br-card critic" aria-label="What the critic says">
      <h3><span className="br-dot critic" aria-hidden="true" />Critic says</h3>
      <p className="br-card-sub">22 → 256 → 256 → 1 · {parameters.toLocaleString("en-US")} weights</p>
      <p className="br-value" aria-live="polite">{value === null ? "—" : value.toFixed(0)}</p>
      <p className="br-card-sub">expected future reward of this situation; more negative means more waiting ahead</p>
      <svg viewBox="0 0 100 30" preserveAspectRatio="none" className="br-spark" aria-hidden="true">
        <polyline points={points} fill="none" stroke="#fbbf24" strokeWidth="1.6" vectorEffect="non-scaling-stroke" />
      </svg>
    </section>
  );
};

/** The 22 numbers going in, with their names. */
export const InputBars: React.FC<{ names: string[]; obs: Float32Array | null }> = ({ names, obs }) => (
  <section className="br-card wide" aria-label="The 22 numbers the networks read">
    <h3>What goes in</h3>
    <p className="br-card-sub">the same 22 numbers feed the actor and the critic; slots for phases 4–6 stay 0 because this junction has three</p>
    <ul className="br-inputs">
      {names.map((n, i) => {
        const v = obs ? obs[i] : 0;
        return (
          <li key={n}>
            <span className="br-in-name">{n}</span>
            <span className="br-bar thin"><span style={{ transform: `scaleX(${Math.max(0, Math.min(1, v))})` }} /></span>
            <span className="br-bar-num">{v.toFixed(2)}</span>
          </li>
        );
      })}
    </ul>
  </section>
);
