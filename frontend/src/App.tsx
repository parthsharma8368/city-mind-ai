import React from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import {
  Activity, PlayCircle, BarChart3, Cpu, BookOpen,
} from "lucide-react";
import { StoryPage } from "./pages/StoryPage";
import { DashboardPage } from "./pages/DashboardPage";
import { ComparisonPage } from "./pages/ComparisonPage";
import { ReplayPage } from "./pages/ReplayPage";
import { SimulationSocketProvider } from "./context/SimulationSocketContext";
import { useSimulationSocket } from "./context/simulationSocketStore";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { API_DOCS_URL } from "./api/client";

const NAV_ITEMS = [
  { to: "/story",         label: "Story",        icon: <BookOpen size={16} aria-hidden="true" /> },
  { to: "/",             label: "Dashboard",    icon: <Activity size={16} aria-hidden="true" /> },
  { to: "/replay",       label: "Replay",       icon: <PlayCircle size={16} aria-hidden="true" /> },
  { to: "/comparison",   label: "Comparison",   icon: <BarChart3 size={16} aria-hidden="true" /> },
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

function NotFound() {
  return (
    <div className="glass-card" style={{ padding: "3rem", textAlign: "center" }}>
      <h2 className="section-title">Page not found</h2>
      <p className="section-subtitle" style={{ marginTop: "0.5rem" }}>
        There is nothing at this address. Use the navigation above to pick a page.
      </p>
    </div>
  );
}

export function App() {
  const { pathname } = useLocation();
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
                One shared RL policy · 98 traffic signals
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
            
          </div>
        </header>

        {/* ── Main Content ─────────────────────────────────────────────────── */}
        <main id="main-content" tabIndex={-1} className="app-main">
          {/* Page-level boundary: a render error in one page keeps the header/nav
              usable, and navigating (pathname change) resets it. */}
          <ErrorBoundary compact resetKey={pathname}>
          <Routes>
            <Route path="/story" element={<StoryPage />} />
            <Route path="/" element={<DashboardPage />} />
            <Route path="/replay" element={<ReplayPage />} />
            <Route path="/comparison" element={<ComparisonPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
          </ErrorBoundary>
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
          <span>CityMind AI — shared-policy PPO (IPPO) traffic-signal control vs SUMO actuated</span>
          <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <Cpu size={12} aria-hidden="true" /> PPO · Stable-Baselines3 · SUMO
            </span>
            <a href={API_DOCS_URL} target="_blank" rel="noreferrer"
              style={{ color: "var(--accent-cyan)", textDecoration: "none", fontFamily: "var(--font-mono)" }}>
              API Docs →
            </a>
          </div>
        </footer>
      </div>
    </SimulationSocketProvider>
  );
}
