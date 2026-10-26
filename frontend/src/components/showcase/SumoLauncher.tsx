import React, { useCallback, useEffect, useRef, useState } from "react";
import { MonitorPlay, Square, Loader2 } from "lucide-react";
import { api, isAbortError } from "../../api/client";
import type { SimControllerId, SimulationControllerInfo } from "../../api/client";
import { useSimulationSocket } from "../../context/simulationSocketStore";

const FALLBACK: SimulationControllerInfo[] = [
  { id: "sumo_actuated", label: "SUMO actuated (baseline)", kind: "baseline", available: true },
  { id: "fixed_time", label: "Fixed time", kind: "baseline", available: true },
  { id: "ippo_A", label: "Model A (final)", kind: "model", available: true },
  { id: "ippo_B", label: "Model B (final)", kind: "model", available: true },
];
const BLURB: Record<string, string> = {
  sumo_actuated: "The calibrated detector-based program the city runs today.",
  fixed_time: "A plain timer for every junction, for context.",
  ippo_A: "One shared policy, 98 junctions, fine-tuned 10 rounds.",
  ippo_B: "Second, independently trained policy (different seed).",
};
/** A started window needs this long to open (SUMO loads the whole city first). */
const STARTING_MS = 60_000;

export const SumoLauncher: React.FC<{ colors: Record<string, string> }> = ({ colors }) => {
  const { lastMessage, status } = useSimulationSocket();
  const [controllers, setControllers] = useState<SimulationControllerInfo[]>(FALLBACK);
  const [polled, setPolled] = useState<SimControllerId[]>([]);
  const [failures, setFailures] = useState<Record<string, string>>({});
  const [starting, setStarting] = useState<Record<string, true>>({});
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const mounted = useRef(true);

  const refresh = useCallback((signal?: AbortSignal) => {
    api.getSimulationStatus(signal)
      .then((s) => {
        if (!mounted.current) return;
        if (s.controllers?.length) setControllers(s.controllers);
        setPolled(s.running_controllers ?? []);
        const failed = s.last_errors ?? {};
        setFailures(failed);
        // a window that already stopped is no longer "starting"
        if (Object.keys(failed).length) setStarting((cur) => Object.fromEntries(Object.entries(cur).filter(([k]) => !(k in failed))));
      })
      .catch((e: unknown) => {
        if (!isAbortError(e) && mounted.current) setPolled([]);
      });
  }, []);

  useEffect(() => {
    mounted.current = true;
    const ctrl = new AbortController();
    refresh(ctrl.signal);
    const id = window.setInterval(() => refresh(), 4000);
    return () => {
      mounted.current = false;
      ctrl.abort();
      window.clearInterval(id);
    };
  }, [refresh]);

  // The websocket is authoritative while connected; polling covers the gaps.
  const running: SimControllerId[] = status === "connected" && lastMessage ? (lastMessage.running_controllers ?? []) : polled;

  // A window counts as "starting" until the backend reports it running (or the allowance runs out).
  const isStarting = (id: string) => starting[id] === true && !running.includes(id as SimControllerId) && !failures[id];

  const launch = async (id: SimControllerId, opts?: { pos?: string; size?: string }) => {
    setError(null);
    setStarting((s) => ({ ...s, [id]: true }));
    window.setTimeout(() => {
      if (mounted.current) setStarting((s) => Object.fromEntries(Object.entries(s).filter(([k]) => k !== id)));
    }, STARTING_MS);
    try {
      const r = await api.launchSimulationGui(id, opts);
      setNote(r.message);
      refresh();
    } catch (e) {
      setStarting((s) => {
        const { [id]: _drop, ...rest } = s;
        return rest;
      });
      setError(e instanceof Error ? e.message.replace(/^API [^:]+: /, "") : "Could not start SUMO.");
    }
  };

  const stop = async (id?: SimControllerId) => {
    setError(null);
    try {
      const r = await api.stopSimulationGui(id);
      setNote(r.message);
      setStarting((s) => (id ? Object.fromEntries(Object.entries(s).filter(([k]) => k !== id)) : {}));
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not stop SUMO.");
    }
  };

  const launchAll = async () => {
    const w = Math.floor((window.screen.availWidth || 1440) / 2);
    const h = Math.floor(((window.screen.availHeight || 900) - 40) / 2);
    const cells = ["0,30", `${w},30`, `0,${30 + h}`, `${w},${30 + h}`];
    const ids = controllers.filter((c) => c.available).map((c) => c.id);
    for (let i = 0; i < ids.length; i++) {
      await launch(ids[i], { pos: cells[i % 4], size: `${w},${h}` });
      await new Promise((r) => setTimeout(r, 900));
    }
  };

  const anyRunning = running.length > 0;
  return (
    <section className="sc-sumo" aria-label="Watch the simulation in SUMO">
      <header className="sc-sumo-head">
        <div>
          <h2>Watch it in SUMO</h2>
          <p className="sc-lede">Opens the real SUMO window on this computer and drives the same Ingolstadt peak hour live with the chosen controller. The window takes about half a minute to open.</p>
        </div>
        <div className="sc-sumo-actions">
          <button type="button" className="sc-btn ghost" onClick={() => void launchAll()} disabled={controllers.every((c) => !c.available)}>
            <MonitorPlay size={15} aria-hidden="true" /> Open all four, tiled
          </button>
          <button type="button" className="sc-btn ghost" onClick={() => void stop()} disabled={!anyRunning && Object.keys(starting).length === 0}>
            <Square size={14} aria-hidden="true" /> Stop all
          </button>
        </div>
      </header>
      <div className="sc-sumo-grid">
        {controllers.map((c) => {
          const on = running.includes(c.id);
          const wait = isStarting(c.id);
          return (
            <article key={c.id} className={`sc-sumo-card${on ? " on" : ""}`}>
              <div className="sc-sumo-top">
                <span className="sc-dot" style={{ background: colors[c.id] ?? "#64748b" }} aria-hidden="true" />
                <h3>{c.label}</h3>
              </div>
              <p>{BLURB[c.id] ?? ""}</p>
              {on ? (
                <button type="button" className="sc-btn stop" onClick={() => void stop(c.id)}><Square size={14} aria-hidden="true" /> Stop</button>
              ) : (
                <button type="button" className="sc-btn" disabled={!c.available || wait} onClick={() => void launch(c.id)}>
                  {wait ? <><Loader2 size={15} className="sc-spin" aria-hidden="true" /> Opening…</> : <><MonitorPlay size={15} aria-hidden="true" /> {c.available ? "Open in SUMO" : "Model file missing"}</>}
                </button>
              )}
              <span className={`sc-run${on ? " on" : failures[c.id] ? " bad" : ""}`} role="status">
                {on ? "running in SUMO" : wait ? "starting, please wait" : failures[c.id] ? "stopped" : "idle"}
              </span>
              {!on && !wait && failures[c.id] && <span className="sc-run-why">{failures[c.id]}</span>}
            </article>
          );
        })}
      </div>
      {error && <p className="sc-error" role="alert">{error}</p>}
      {!error && note && <p className="sc-note" role="status">{note}</p>}
    </section>
  );
};
