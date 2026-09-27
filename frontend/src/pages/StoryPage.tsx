import React from "react";
import {
  BookOpen, HelpCircle, Bug, Table2, AlertTriangle, GitBranch, Wrench,
  ScanSearch, Sliders, Clock,
} from "lucide-react";
import { CONTROLLER_LABELS as CTRL_LABELS, CONTROLLER_COLORS as CTRL_COLORS } from "../constants/controllers";

type StoryRow = {
  controller: string;
  trips: string;
  timeloss: string;
  backlog: string;
  teleports: string;
  wait: string;
};

// The corrected peak (07:00-08:00) headline numbers, mean over seeds
// 42/123/777, straight from the committed artifacts/metrics/*.json files.
// Do not re-derive or re-round these — they are quoted exactly as measured.
const STORY_ROWS: StoryRow[] = [
  { controller: "sumo_actuated", trips: "11,600", timeloss: "284.4", backlog: "3,257", teleports: "7.7", wait: "6.06" },
  { controller: "fixed_time", trips: "11,454", timeloss: "301.8", backlog: "3,472", teleports: "7.7", wait: "6.97" },
  { controller: "max_pressure", trips: "10,408", timeloss: "342.9", backlog: "3,755", teleports: "50.3", wait: "12.62" },
  { controller: "ppo", trips: "9,546", timeloss: "354.7", backlog: "4,112", teleports: "19.0", wait: "7.61" },
  { controller: "ippo", trips: "9,520", timeloss: "347.8", backlog: "4,086", teleports: "10.0", wait: "5.26" },
];

const SectionCard: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div className="glass-card" style={{ padding: "1.5rem", marginBottom: "1.25rem", ...style }}>
    {children}
  </div>
);

const SectionHeading: React.FC<{ icon: React.ReactNode; title: string; badge?: React.ReactNode }> = ({ icon, title, badge }) => (
  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.75rem", marginBottom: "0.85rem", flexWrap: "wrap" }}>
    <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
      {icon}
      <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)" }}>{title}</h3>
    </div>
    {badge}
  </div>
);

export const StoryPage: React.FC = () => {
  return (
    <div style={{ animation: "fade-in 0.4s ease both" }}>
      {/* Header */}
      <div style={{ marginBottom: "1.25rem" }}>
        <h2 className="section-title">Project Story</h2>
        <p className="section-subtitle">
          <BookOpen size={13} style={{ display: "inline", verticalAlign: "middle", marginRight: 4 }} aria-hidden="true" />
          How this project was built, what it found, and where it honestly stands today
        </p>
      </div>

      {/* 1. Hero / premise */}
      <SectionCard>
        <SectionHeading icon={<HelpCircle size={18} color="var(--accent-cyan)" aria-hidden="true" />} title="Can reinforcement learning beat calibrated traffic-signal control on a real city network?" />
        <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7 }}>
          This project trains and evaluates reinforcement-learning traffic-signal controllers on a SUMO
          microsimulation of the InTAS Ingolstadt network — 98 signalized intersections — and compares them,
          on identical demand and identical seeds, against a genuine static fixed-timer baseline and against
          SUMO's own calibrated adaptive (gap-based actuated) control. The rest of this page walks through
          what was found, including the parts that did not go as hoped.
        </p>
      </SectionCard>

      {/* 2. What we found was broken first */}
      <SectionCard>
        <SectionHeading icon={<Bug size={18} color="var(--signal-yellow)" aria-hidden="true" />} title="What we found was broken first" />
        <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: "0.85rem" }}>
          Before trusting any RL result, we audited the evaluation setup itself and found two problems that
          would have made every downstream comparison meaningless:
        </p>
        <ul style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, paddingLeft: "1.25rem", marginBottom: "0.85rem" }}>
          <li style={{ marginBottom: "0.5rem" }}>
            <strong style={{ color: "var(--text-primary)" }}>A mislabeled baseline.</strong> The controller we were
            calling <code style={{ fontFamily: "var(--font-mono)", fontSize: "0.82rem" }}>fixed_time</code> was
            actually running InTAS's own calibrated adaptive (gap-based actuated) signal programs, not a real
            static timer. Comparing RL against it was comparing against a moving target, not a fixed one — and
            the original training/evaluation window was midnight, when traffic is near zero and the difference
            between any two controllers barely shows up at all.
          </li>
          <li>
            <strong style={{ color: "var(--text-primary)" }}>A universal ~5x undercount.</strong> Every controller's
            departure and arrival counts were pulled from a TraCI "since last call" counter that was queried far
            too infrequently, silently dropping most of the vehicles it should have counted, in every run, for
            every controller.
          </li>
        </ul>
        <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.7 }}>
          Neither of these was fixed by changing an algorithm — they were fixed by fixing what we were measuring.
          Catching this <em>before</em> trusting any RL result is what makes everything after it on this page
          trustworthy.
        </p>
      </SectionCard>

      {/* 3. The corrected picture */}
      <SectionCard>
        <SectionHeading icon={<Table2 size={18} color="var(--accent-cyan)" aria-hidden="true" />} title="The corrected picture" badge={<span className="badge-blue">Peak 07:00–08:00 · mean of 3 seeds</span>} />
        <div style={{ overflowX: "auto", marginBottom: "1rem" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Controller</th>
                <th>Trips Completed</th>
                <th>Mean TimeLoss (s)</th>
                <th>Backlog</th>
                <th>Teleports</th>
                <th>Mean Wait (s)</th>
              </tr>
            </thead>
            <tbody>
              {STORY_ROWS.map((r) => (
                <tr key={r.controller}>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ width: 10, height: 10, borderRadius: "50%", background: CTRL_COLORS[r.controller] ?? "#64748b", display: "inline-block" }} />
                      <span style={{ fontWeight: 600 }}>{CTRL_LABELS[r.controller] ?? r.controller}</span>
                    </div>
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{r.trips}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{r.timeloss}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{r.backlog}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{r.teleports}</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--accent-cyan)" }}>{r.wait}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* This callout is the most important element on the page — impossible
            to miss, amber/warning tone (never green), and phrased so it cannot
            be screenshotted as "IPPO wins." */}
        <div
          role="alert"
          style={{
            display: "flex",
            gap: "0.75rem",
            alignItems: "flex-start",
            background: "rgba(245,158,11,0.1)",
            border: "1px solid rgba(245,158,11,0.4)",
            borderRadius: "var(--radius-md)",
            padding: "1rem 1.15rem",
          }}
        >
          <AlertTriangle size={20} color="var(--signal-yellow)" style={{ flexShrink: 0, marginTop: 2 }} aria-hidden="true" />
          <div>
            <p style={{ fontSize: "0.9rem", fontWeight: 700, color: "var(--signal-yellow)", marginBottom: "0.35rem" }}>
              Neither RL policy beats the baselines on trips completed.
            </p>
            <p style={{ fontSize: "0.87rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>
              IPPO's lower mean wait time (5.26s) is a <strong style={{ color: "var(--text-primary)" }}>starvation
              artifact</strong>, not a real win: that number is measured only over vehicles still in the network.
              Against the calibrated SUMO-Actuated baseline, IPPO completes about 18% fewer trips (9,520 vs
              11,600) and leaves about 26% more vehicles stuck in backlog (4,086 vs 3,257). A controller that
              finishes fewer trips while stranding more vehicles has not solved traffic control, no matter what
              its waiting-time number says on its own.
            </p>
          </div>
        </div>
      </SectionCard>

      {/* 4. Root cause: the action space */}
      <SectionCard>
        <SectionHeading icon={<GitBranch size={18} color="var(--accent-indigo)" aria-hidden="true" />} title="Root cause: the action space" />
        <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: "0.85rem" }}>
          Every controller above — including both RL policies — was choosing actions from the same limited menu:
          at each decision point, either keep the current green phase, or advance to the next phase in a fixed
          cycle. That action can never skip past an already-empty movement to go serve a congested one elsewhere
          at the same intersection; it can only step forward one phase at a time, in order.
        </p>
        <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7 }}>
          The clearest evidence that this is a mechanical limitation, not a training problem, is Max-Pressure:
          it is a provably throughput-optimal algorithm under a good action space, with no learning involved at
          all, and yet under this same binary action space it hit roughly the same performance ceiling as the
          learned policies (342.9s mean timeloss, worse trip completion than either fixed baseline). If a
          provably optimal method cannot do better here, the ceiling is coming from what the controller is
          allowed to do, not from how well anything learns.
        </p>
      </SectionCard>

      {/* 5. The fix: phase selection */}
      <SectionCard>
        <SectionHeading icon={<Sliders size={18} color="var(--accent-purple)" aria-hidden="true" />} title="The fix: phase selection" />
        <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7 }}>
          The action space was replaced with a masked <code style={{ fontFamily: "var(--font-mono)", fontSize: "0.82rem" }}>Discrete(6)</code>{" "}
          phase-selection action: at each decision point, the agent picks directly among any of that
          intersection's own green phases, with MaskablePPO restricting the choice to phases that are actually
          valid for that intersection. The observation was extended with per-phase pressure, so the agent can
          see which specific movement is under load rather than only an aggregate figure for the whole
          intersection. This runs as parameter-shared independent PPO (IPPO) across all 98 intersections.
        </p>
      </SectionCard>

      {/* 6. First attempt failed */}
      <SectionCard>
        <SectionHeading icon={<ScanSearch size={18} color="var(--signal-red)" aria-hidden="true" />} title="First attempt failed — and we know exactly why" badge={<span className="badge-red">Gate G1</span>} />
        <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: "0.85rem" }}>
          The first training run under the new phase-selection action space failed badly: 7,716 trips completed,
          5,853 backlog, roughly 1,817 teleports per episode — worse than every baseline, and worse than the old
          binary-action IPPO it was meant to replace.
        </p>
        <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7 }}>
          The trip-level metrics built while fixing the undercount bug in section 2 made the diagnosis possible:
          watching one real intersection directly, a phase's own pressure climbed from 6 to 9+ over 15 real
          minutes while the policy sat on an already-empty phase the whole time. The reward it was trained on
          used <em>aggregate</em> intersection pressure, which barely moves in an agent-attributable way when you
          switch phases, and a switching penalty was the only reward term firing reliably every step — so the
          policy learned the locally-optimal-looking behavior of simply never switching. The same rigor that
          caught the first bug caught this one.
        </p>
      </SectionCard>

      {/* 7. The reward fix */}
      <SectionCard>
        <SectionHeading icon={<Wrench size={18} color="var(--accent-cyan)" aria-hidden="true" />} title="The reward fix" />
        <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7 }}>
          Three changes were made to the reward: a new "neglected pressure" term that directly penalizes the
          worst pressure among phases <em>not</em> currently being served, a restored waiting-delta term, and a
          6x cut to the switching penalty so it no longer dominates every other signal. Forcing the exact
          "always hold" failure pattern from section 6 through the new reward now produces a real, sustained
          penalty, where before it produced none.
        </p>
      </SectionCard>

      {/* 8. Where things stand right now */}
      <SectionCard style={{ border: "1px solid rgba(245,158,11,0.3)" }}>
        <SectionHeading
          icon={<Clock size={18} color="var(--signal-yellow)" aria-hidden="true" />}
          title="Where things stand right now"
          badge={<span className="badge-yellow">IN PROGRESS</span>}
        />
        <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: "0.6rem" }}>
          A full retrain under the fixed reward (controller id{" "}
          <code style={{ fontFamily: "var(--font-mono)", fontSize: "0.82rem" }}>ippo_phase_select</code>) is
          currently running. It has already been interrupted twice by infrastructure issues and resumed from
          checkpoint — not by any bug in the policy itself.
        </p>
        <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7 }}>
          There is no final evaluated result for it yet, so it does not appear in the table in section 3 and it
          makes no performance claim here. A fix for the section 6 failure is training now; results will appear
          once evaluation completes. Check the Comparison page for the latest evaluated numbers.
        </p>
      </SectionCard>
    </div>
  );
};
