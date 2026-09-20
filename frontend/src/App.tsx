import React from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import {
  Activity, Layers, PlayCircle, BarChart3, Brain, Cpu, Monitor,
} from "lucide-react";
import { DashboardPage } from "./pages/DashboardPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { ComparisonPage } from "./pages/ComparisonPage";
import { XAIPage } from "./pages/XAIPage";
import { ReplayPage } from "./pages/ReplayPage";
import { SimulationSocketProvider } from "./context/SimulationSocketContext";
import { useSimulationSocket } from "./context/simulationSocketStore";

const NAV_ITEMS = [
  { to: "/",             label: "Dashboard",    icon: <Activity size={16} aria-hidden="true" /> },
  { to: "/replay",       label: "Replay",       icon: <PlayCircle size={16} aria-hidden="true" /> },
  { to: "/experiments",  label: "Experiments",  icon: <Layers size={16} aria-hidden="true" /> },
  { to: "/comparison",   label: "Comparison",   icon: <BarChart3 size={16} aria-hidden="true" /> },
  { to: "/xai",          label: "XAI Analysis", icon: <Brain size={16} aria-hidden="true" /> },
];

const ConnectionBadge: React.FC = () => {
  const { status } = useSimulationSocket();
  const label = status === "connected" ? "Live" : status === "connecting" ? "Connecting…" : "Offline";
  return (
    <div
      className="ws-indicator"
      role="status"
      aria-live="polite"
      title={`Live SUMO status stream: ${label}`}
    >
      <span className={`ws-dot${status !== "connected" ? " disconnected" : ""}`} aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
};

export function App() {
  return (
    <SimulationSocketProvider>
      <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
        <a href="#main-content" className="skip-link">Skip to main content</a>

        {/* ── Navigation Bar ──────────────────────────────────────────────── */}
        <header className="navbar">
          <div className="logo-container">
            <div className="logo-badge" aria-hidden="true">CM</div>
            <div>
              <h1 style={{ fontSize: "1.1rem", fontWeight: 700, letterSpacing: "-0.02em" }}>CityMind AI</h1>
              <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", letterSpacing: "0.02em" }}>
                Multi-Agent RL Traffic Intelligence
              </p>
            </div>
          </div>

          <nav className="nav-links" aria-label="Primary">
            {NAV_ITEMS.map(({ to, label, icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}
              >
                {icon} {label}
              </NavLink>
            ))}
          </nav>

          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <ConnectionBadge />
            <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>v1.0.0</span>
          </div>
        </header>

        {/* ── Main Content ─────────────────────────────────────────────────── */}
        <main id="main-content" tabIndex={-1} style={{ flex: 1, padding: "2rem", maxWidth: "1360px", width: "100%", margin: "0 auto" }}>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/replay" element={<ReplayPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
            <Route path="/comparison" element={<ComparisonPage />} />
            <Route path="/xai" element={<XAIPage />} />
          </Routes>
        </main>

        {/* ── Footer ───────────────────────────────────────────────────────── */}
        <footer style={{
          borderTop: "1px solid var(--border-subtle)",
          padding: "0.85rem 2rem",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "0.5rem",
          fontSize: "0.75rem",
          color: "var(--text-muted)",
        }}>
          <span>CityMind AI — Multi-Intersection RL Traffic Control (PPO + IPPO)</span>
          <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <Cpu size={12} aria-hidden="true" /> PPO · IPPO · SB3 · SUMO 1.27.1
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <Monitor size={12} aria-hidden="true" /> Local visual demo via XQuartz
            </span>
            <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer"
              style={{ color: "var(--accent-cyan)", textDecoration: "none", fontFamily: "var(--font-mono)" }}>
              API Docs →
            </a>
          </div>
        </footer>
      </div>
    </SimulationSocketProvider>
  );
}
