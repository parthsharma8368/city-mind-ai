import React from "react";
import { Link } from "react-router-dom";
import { BookOpen, HelpCircle, Table2, AlertTriangle, Wrench, Sliders, FlaskConical } from "lucide-react";
import { TenSeedTable } from "../components/TenSeedTable";

const SectionCard: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div className="glass-card" style={{ padding: "1.5rem", marginBottom: "1.25rem", ...style }}>
    {children}
  </div>
);

const SectionHeading: React.FC<{ icon: React.ReactNode; title: string }> = ({ icon, title }) => (
  <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.85rem" }}>
    {icon}
    <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)" }}>{title}</h3>
  </div>
);

const P: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: "0.7rem" }}>{children}</p>
);

const Mono: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <code style={{ fontFamily: "var(--font-mono)", fontSize: "0.82rem" }}>{children}</code>
);

const FIXES: Array<{ problem: string; fix: string }> = [
  {
    problem: "The network was barely learning.",
    fix: "PPO clips the gradient to keep updates stable, but the policy and the value network were clipped together. The value network's gradient (~3,830) dwarfed the policy's (~0.19), so the policy's steps shrank about 7,600×. Now each is clipped on its own (src/agents/split_clip.py).",
  },
  {
    problem: "The reward could be gamed.",
    fix: "It paid for the change in queue, so a controller could let a queue grow and release it at once. It now pays for the level of waiting, so the return is minus the total delay (src/environment/local_reward.py).",
  },
  {
    problem: "The inputs were dead, and blind to timing.",
    fix: "Two inputs sat at their maximum for most of the hour and the signal the baseline reacts to - time in a green relative to its own minimum - was missing. The new 22-number observation has it (src/environment/local_observation.py).",
  },
  {
    problem: "Blind behind short stop-line lanes.",
    fix: "A 20 m lane holds about 2 cars but 60 can wait on the road behind it. The agent now counts halted cars up to 300 m upstream.",
  },
  {
    problem: "Blind to cars waiting outside the network.",
    fix: "Stage 2: cars that cannot enter yet count towards the demand of the lane they will queue at, and the trained network is fine-tuned for 10 (A) or 20 (B) more updates. That closed the gap at the network entrances.",
  },
];

export const StoryPage: React.FC = () => (
  <div style={{ animation: "fade-in 0.4s ease both" }}>
    <div style={{ marginBottom: "1.25rem" }}>
      <h2 className="section-title">Project Story</h2>
      <p className="section-subtitle">
        <BookOpen size={13} style={{ display: "inline", verticalAlign: "middle", marginRight: 4 }} aria-hidden="true" />
        One neural network for 98 traffic lights: what it does, why it works and where it stops
      </p>
    </div>

    <SectionCard>
      <SectionHeading icon={<HelpCircle size={18} color="var(--accent-cyan)" aria-hidden="true" />} title="Can one learned policy beat SUMO's own signal control?" />
      <P>
        We simulate the 07:00-08:00 rush hour of Ingolstadt (InTAS, about 15,000 cars, 98 signalised junctions) in SUMO. The opponent is SUMO's vehicle-actuated
        control, which keeps a green on while cars keep arriving. We train <strong>one</strong> neural network, copied to every junction (PPO used as IPPO). Every
        5 simulated seconds each junction shows it 22 numbers about its traffic and gets one answer: stay green or move to the next green.
      </P>
      <P>
        <strong>On this scenario: yes.</strong> Two independently trained models were tested on 10 random traffic seeds against the baseline on identical traffic: model A is better on
        12 of 14 measures and model B on 11 of 14, and neither is worse on any. The measures that are not won are listed under honest limits.
      </P>
    </SectionCard>

    <SectionCard>
      <SectionHeading icon={<Table2 size={18} color="var(--signal-green)" aria-hidden="true" />} title="The evidence" />
      <TenSeedTable keys={["lag_per_scheduled_car_s", "mean_waiting_time", "mean_queue", "mean_speed", "trips_completed", "backlog", "never_inserted", "teleports"]} />
      <P>
        Lag per scheduled car is the main score: for every car that was supposed to drive, how many seconds later than on an empty road did it finish -
        cars that never finish or never enter count as late until 08:00. See <Link to="/comparison" style={{ color: "var(--accent-cyan)" }}>Comparison</Link> for all 14 measures and
        {" "}<Link to="/replay" style={{ color: "var(--accent-cyan)" }}>Replay</Link> to watch the traffic.
      </P>
    </SectionCard>

    <SectionCard>
      <SectionHeading icon={<Sliders size={18} color="var(--accent-purple)" aria-hidden="true" />} title="How it works" />
      <ul style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, paddingLeft: "1.25rem" }}>
        <li style={{ marginBottom: "0.5rem" }}><strong style={{ color: "var(--text-primary)" }}>State.</strong> 22 numbers per junction: which green is on, how long, the demand of each
          green (cars waiting on its lanes and the roads feeding them, minus cars on the road it flows into), cars waiting outside the network, and the queues of the road neighbours.</li>
        <li style={{ marginBottom: "0.5rem" }}><strong style={{ color: "var(--text-primary)" }}>Action.</strong> Keep the green or advance to the next green. A safety layer enforces each
          phase's minimum and maximum green and always runs the yellow / all-red clearance.</li>
        <li style={{ marginBottom: "0.5rem" }}><strong style={{ color: "var(--text-primary)" }}>Reward.</strong> <Mono>r = -0.1 · min(Q + 2D + S + 0.5·P_own + 0.5·P_all/98, 200)</Mono> - standing cars on the junction's lanes (Q),
          on the roads it feeds (D), a price per switch (S) and cars waiting outside (P).</li>
        <li><strong style={{ color: "var(--text-primary)" }}>Learning.</strong> PPO with GAE (γ 0.99, λ 0.95, clip 0.2); networks 22→256→256 (tanh); 105,840 decisions per update
          (180 steps × 98 junctions × 6 parallel SUMO runs); 100 updates, then a short fine-tune.</li>
      </ul>
    </SectionCard>

    <SectionCard>
      <SectionHeading icon={<Wrench size={18} color="var(--signal-yellow)" aria-hidden="true" />} title="What changed to beat the baseline" />
      <ol style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, paddingLeft: "1.25rem" }}>
        {FIXES.map((f) => (
          <li key={f.problem} style={{ marginBottom: "0.6rem" }}>
            <strong style={{ color: "var(--text-primary)" }}>{f.problem}</strong> {f.fix}
          </li>
        ))}
      </ol>
    </SectionCard>

    <SectionCard style={{ borderColor: "rgba(245, 158, 11, 0.35)" }}>
      <SectionHeading icon={<AlertTriangle size={18} color="var(--signal-yellow)" aria-hidden="true" />} title="Honest limits" />
      <ul style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, paddingLeft: "1.25rem" }}>
        <li style={{ marginBottom: "0.4rem" }}><strong style={{ color: "var(--text-primary)" }}>Cars never inserted is level</strong> (503 vs 501 / 481): the fix closed the gap at the network entrances but did not open a lead.</li>
        <li style={{ marginBottom: "0.4rem" }}><strong style={{ color: "var(--text-primary)" }}>Teleports</strong> (gridlock events) are lower on average but too noisy to claim.</li>
        <li style={{ marginBottom: "0.4rem" }}>One scenario: the same network and demand pattern for training and test; only the random details differ between seeds. Another city or real traffic is untested.</li>
        <li style={{ marginBottom: "0.4rem" }}>The agent reads exact halted-car counts, including cars waiting outside the network; noisy real detectors were not tested.</li>
        <li>The models change phase more often (about 17,600 / 18,600 per hour against 14,600 for actuated control).</li>
      </ul>
      <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.8rem", display: "flex", alignItems: "center", gap: 6 }}>
        <FlaskConical size={13} aria-hidden="true" /> Full write-up: <Mono>docs/TEST_REPORT_10_SEEDS.md</Mono> and <Mono>docs/PROJECT_REPORT_FOR_PROFESSOR.pdf</Mono>.
      </p>
    </SectionCard>
  </div>
);
