import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";
import { api, isAbortError } from "../api/client";
import { buildMlp, parameterCount, type Mlp } from "../brain/forward";
import { NeuralScene, type DecideInfo, type ObserveInfo } from "../components/brain/NeuralScene";
import { ActorCard, CriticCard, InputBars, JunctionCard } from "../components/brain/BrainCards";
import "../styles/brain.css";

const MODELS = [
  { id: "ippo_A", label: "Model A" },
  { id: "ippo_B", label: "Model B" },
] as const;
const SPEEDS = [0.5, 1, 2] as const;
const HISTORY = 48;
const RECENT = 30;

export const BrainPage: React.FC = () => {
  const [modelId, setModelId] = useState<(typeof MODELS)[number]["id"]>("ippo_A");
  const [mlp, setMlp] = useState<Mlp | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);
  const [resetKey, setResetKey] = useState(0);
  const [observed, setObserved] = useState<ObserveInfo | null>(null);
  const [decision, setDecision] = useState<DecideInfo | null>(null);
  const [history, setHistory] = useState<number[]>([]);
  const [recent, setRecent] = useState<DecideInfo[]>([]);

  useEffect(() => {
    const ctrl = new AbortController();
    api.getPolicy(modelId, ctrl.signal)
      .then((dto) => {
        if (ctrl.signal.aborted) return;
        setMlp(buildMlp(dto));
        setError(null);
        setObserved(null);
        setDecision(null);
        setHistory([]);
        setRecent([]);
      })
      .catch((e: Error) => {
        if (!ctrl.signal.aborted && !isAbortError(e)) setError(e.message);
      });
    return () => ctrl.abort();
  }, [modelId, attempt]);

  const choose = (id: (typeof MODELS)[number]["id"]) => {
    if (id === modelId) return;
    setMlp(null);
    setError(null);
    setModelId(id);
  };
  const onObserve = useCallback((info: ObserveInfo) => setObserved(info), []);
  const onDecide = useCallback((info: DecideInfo) => {
    setDecision(info);
    setHistory((h) => [...h.slice(-(HISTORY - 1)), info.value]);
    setRecent((r) => [...r.slice(-(RECENT - 1)), info]);
  }, []);

  const params = useMemo(() => (mlp ? { actor: parameterCount(mlp.actor), critic: parameterCount(mlp.critic) } : null), [mlp]);

  return (
    <div className="br-root">
      <header className="br-head">
        <div>
          <p className="br-eyebrow">Inside the model</p>
          <h1>Two networks watch every junction.</h1>
          <p className="br-sub">
            The <b className="actor">actor</b> chooses whether to keep the green or advance; the <b className="critic">critic</b> scores how good the
            situation is. Both read the same 22 numbers every 5 seconds, with the real trained weights.
          </p>
        </div>
        <div className="br-toolbar" role="group" aria-label="Figure controls">
          <div className="br-seg" role="group" aria-label="Model">
            {MODELS.map((m) => (
              <button key={m.id} type="button" className={m.id === modelId ? "on" : ""} aria-pressed={m.id === modelId} onClick={() => choose(m.id)}>{m.label}</button>
            ))}
          </div>
          <div className="br-seg" role="group" aria-label="Speed">
            {SPEEDS.map((s) => (
              <button key={s} type="button" className={s === speed ? "on" : ""} aria-pressed={s === speed} onClick={() => setSpeed(s)}>{s}x</button>
            ))}
          </div>
          <button type="button" className="sc-btn ghost" onClick={() => setPaused((p) => !p)} aria-pressed={paused}>
            {paused ? <Play size={15} aria-hidden="true" /> : <Pause size={15} aria-hidden="true" />}{paused ? "Resume" : "Pause"}
          </button>
          <button type="button" className="sc-btn ghost" onClick={() => setResetKey((k) => k + 1)}>
            <RotateCcw size={15} aria-hidden="true" />Reset view
          </button>
        </div>
      </header>



      {error ? (
        <div className="br-fail" role="alert">
          <div><strong>The trained weights could not be loaded.</strong><span>{error}</span></div>
          <button type="button" className="sc-btn" onClick={() => { setError(null); setAttempt((a) => a + 1); }}>Try again</button>
        </div>
      ) : !mlp ? (
        <div className="br-stage br-loading" role="status" aria-live="polite"><span>Loading the weights of {MODELS.find((m) => m.id === modelId)?.label}…</span></div>
      ) : (
        <div className="br-stage-wrap">
          <NeuralScene mlp={mlp} paused={paused} speed={speed} resetKey={resetKey} onObserve={onObserve} onDecide={onDecide} />
          <span className="br-tag actor"><i aria-hidden="true" />Actor · chooses</span>
          <span className="br-tag critic"><i aria-hidden="true" />Critic · judges</span>
        </div>
      )}

      {mlp && params && (
        <>
          <div className="br-live">
            <JunctionCard observed={observed} />
            <ActorCard decision={decision} recent={recent} actions={mlp.actions} parameters={params.actor} />
            <CriticCard history={history} parameters={params.critic} />
          </div>

          <InputBars names={mlp.obsNames} obs={observed?.obs ?? null} />

          <div className="br-explain">
            <article>
              <h3>One brain, 98 junctions</h3>
              <p>
                The same weights run at every junction. Each one feeds in its own 22 numbers every 5 seconds and gets back keep or advance.
                A safety layer still enforces the minimum green ({`10 s here`}), the maximum green and the yellow time.
              </p>
            </article>
            <article>
              <h3>Why two networks?</h3>
              <p>
                Only the actor drives the signals. The critic exists for training: PPO compares what really happened with what the critic expected, and
                makes the actor's choice more likely when it turned out better. At run time the critic is not used.
              </p>
            </article>
            <article>
              <h3>How to read the figure</h3>
              <p>
                Glow is how strongly a neuron fires; cyan and amber are positive, violet and rose negative. A line is brighter the more a connection
                pushes right now (its weight times the firing of the neuron it leaves). A wave crosses the network on every decision.
              </p>
            </article>
            <article>
              <h3>What is real here</h3>
              <p>
                The weights are the trained ones of {MODELS.find((m) => m.id === modelId)?.label}, run in your browser. The traffic is a small made-up junction so the numbers are easy to follow;
                in the real runs the 22 numbers come from SUMO. Each hidden layer has 256 neurons: the 16 that vary most are drawn.
              </p>
            </article>
          </div>
        </>
      )}
      <p className="br-sr" aria-live="polite">
        {decision ? `Actor: ${(decision.probs[1] * 100).toFixed(0)} percent advance. Critic value ${decision.value.toFixed(0)}.` : ""}
      </p>
    </div>
  );
};
