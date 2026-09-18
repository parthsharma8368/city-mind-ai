import React, { useState } from "react";
import {
  Activity, Layers, PlayCircle, BarChart3, Brain, Cpu, Monitor,
} from "lucide-react";
import { DashboardPage } from "./pages/DashboardPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { ComparisonPage } from "./pages/ComparisonPage";
import { XAIPage } from "./pages/XAIPage";
import { ReplayPage } from "./pages/ReplayPage";

type Tab = "dashboard" | "replay" | "experiments" | "comparison" | "xai";

const NAV_ITEMS: Array<{ key: Tab; label: string; icon: React.ReactNode }> = [
  { key: "dashboard",   label: "Dashboard",    icon: <Activity size={16} /> },
  { key: "replay",      label: "Replay",       icon: <PlayCircle size={16} /> },
  { key: "experiments", label: "Experiments",  icon: <Layers size={16} /> },
  { key: "comparison",  label: "Comparison",   icon: <BarChart3 size={16} /> },
  { key: "xai",         label: "XAI Analysis", icon: <Brain size={16} /> },
];


export function App() {
  const [activeTab, setActiveTab] = useState<Tab>("dashboard");

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* ── Navigation Bar ──────────────────────────────────────────────── */}
      <header className="navbar">
        <div className="logo-container">
          <div className="logo-badge">CM</div>
          <div>
            <h1 style={{ fontSize: "1.1rem", fontWeight: 700, letterSpacing: "-0.02em" }}>CityMind AI</h1>
            <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", letterSpacing: "0.02em" }}>
              Centralized RL Traffic Intelligence
            </p>
          </div>
        </div>

        <nav className="nav-links">
          {NAV_ITEMS.map(({ key, label, icon }) => (
            <button
              key={key}
              id={`nav-${key}`}
              className={`nav-item${activeTab === key ? " active" : ""}`}
              onClick={() => setActiveTab(key)}
            >
              {icon} {label}
            </button>
          ))}
        </nav>

        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div className="ws-indicator" title="Live SUMO status stream">
            <span className="ws-dot" />
            <span>Live</span>
          </div>
          <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>v1.0.0</span>
        </div>
      </header>

      {/* ── Main Content ─────────────────────────────────────────────────── */}
      <main style={{ flex: 1, padding: "2rem", maxWidth: "1360px", width: "100%", margin: "0 auto" }}>
        {activeTab === "dashboard"   && <DashboardPage />}
        {activeTab === "replay"      && <ReplayPage />}
        {activeTab === "experiments" && <ExperimentsPage />}
        {activeTab === "comparison"  && <ComparisonPage />}
        {activeTab === "xai"         && <XAIPage />}
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
        <span>CityMind AI — Centralized Multi-Intersection RL Traffic Control</span>
        <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <Cpu size={12} /> PPO · SB3 · SUMO 1.27.1
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <Monitor size={12} /> XQuartz
          </span>
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer"
            style={{ color: "var(--accent-cyan)", textDecoration: "none", fontFamily: "var(--font-mono)" }}>
            API Docs →
          </a>
        </div>
      </footer>
    </div>
  );
}
