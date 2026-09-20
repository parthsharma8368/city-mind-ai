import React, { useEffect, useState } from "react";
import { Brain, AlertCircle, ChevronDown, ChevronUp } from "lucide-react";
import { api } from "../api/client";
import type { XAIJunctionExplanation } from "../api/client";

const GROUP_COLORS: Record<string, string> = {
  phase:         "#22d3ee",
  elapsed:       "#3b82f6",
  queue:         "#ef4444",
  vehicle_count: "#f59e0b",
  speed:         "#10b981",
  density:       "#818cf8",
  global:        "#a855f7",
};

function FeatureBar({ label, value, max }: { label: string; value: number; max: number }) {
  const pct = max > 0 ? Math.abs(value) / max * 100 : 0;
  const isPos = value >= 0;
  return (
    <div className="feature-bar-row">
      <div className="feature-bar-label" title={label}>{label}</div>
      <div className="feature-bar-track">
        <div
          className="feature-bar-fill"
          style={{
            width: `${pct}%`,
            background: isPos ? "var(--signal-green)" : "var(--signal-red)",
          }}
        />
      </div>
      <div className="feature-bar-value">{value.toFixed(4)}</div>
    </div>
  );
}

function JunctionCard({ expl }: { expl: XAIJunctionExplanation }) {
  const [expanded, setExpanded] = useState(false);
  const allFeatures = [...expl.top_positive_features, ...expl.top_negative_features];
  const maxAttr = Math.max(...allFeatures.map((f) => Math.abs(f.attribution)), 0.0001);

  const action = expl.action === 1 ? "SWITCH" : "HOLD";
  const actionColor = expl.action === 1 ? "var(--signal-yellow)" : "var(--signal-green)";

  return (
    <div className="glass-card" style={{ padding: "1rem 1.25rem", marginBottom: "0.75rem" }}>
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        style={{
          display: "flex", alignItems: "center", justifyContent: "space-between", width: "100%",
          marginBottom: expanded ? "0.75rem" : 0, cursor: "pointer",
          background: "none", border: "none", padding: 0, font: "inherit", color: "inherit", textAlign: "left",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "var(--accent-cyan)", fontWeight: 600 }}>
            {expl.tls_id}
          </span>
          <span style={{ fontSize: "0.75rem", fontWeight: 700, padding: "0.15rem 0.5rem", borderRadius: 4, background: expl.action === 1 ? "rgba(245,158,11,0.15)" : "rgba(16,185,129,0.15)", color: actionColor, border: `1px solid ${actionColor}55` }}>
            {action}
          </span>
          <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
            Local: {expl.local_influence_pct.toFixed(1)}% · Neighbor: {expl.neighbor_influence_pct.toFixed(1)}%
          </span>
        </div>
        {expanded ? <ChevronUp size={16} color="var(--text-muted)" /> : <ChevronDown size={16} color="var(--text-muted)" />}
      </button>

      {expanded && (
        <div style={{ animation: "fade-in 0.25s ease both" }}>
          <p style={{ fontSize: "0.83rem", color: "var(--text-secondary)", marginBottom: "0.75rem", lineHeight: 1.5, borderLeft: "2px solid var(--accent-cyan)", paddingLeft: "0.75rem" }}>
            {expl.narrative}
          </p>

          {/* Group Breakdown */}
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "0.75rem" }}>
            {Object.entries(expl.group_breakdown).map(([grp, val]) => (
              <span key={grp} style={{ fontSize: "0.72rem", padding: "0.2rem 0.5rem", borderRadius: 4, background: `${GROUP_COLORS[grp] ?? "#64748b"}20`, color: GROUP_COLORS[grp] ?? "#64748b", border: `1px solid ${GROUP_COLORS[grp] ?? "#64748b"}40`, fontFamily: "var(--font-mono)" }}>
                {grp}: {(val * 100).toFixed(1)}%
              </span>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
            <div>
              <p style={{ fontSize: "0.72rem", color: "var(--signal-green)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.4rem" }}>
                ▲ Switch Drivers
              </p>
              {expl.top_positive_features.slice(0, 5).map((f) => (
                <FeatureBar key={f.feature_index} label={f.name.slice(0, 28)} value={f.attribution} max={maxAttr} />
              ))}
            </div>
            <div>
              <p style={{ fontSize: "0.72rem", color: "var(--signal-red)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.4rem" }}>
                ▼ Hold Drivers
              </p>
              {expl.top_negative_features.slice(0, 5).map((f) => (
                <FeatureBar key={f.feature_index} label={f.name.slice(0, 28)} value={f.attribution} max={maxAttr} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export const XAIPanel: React.FC<{ maxItems?: number }> = ({ maxItems = 10 }) => {
  const [data, setData] = useState<XAIJunctionExplanation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getXAISummary()
      .then((d) => setData(d.slice(0, maxItems)))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [maxItems]);

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--text-muted)", padding: "2rem" }}>
      <span className="spinner" /> Loading XAI explanations…
    </div>
  );

  if (error) return (
    <div className="glass-card" style={{ padding: "2rem", textAlign: "center" }}>
      <AlertCircle size={32} style={{ color: "var(--signal-yellow)", margin: "0 auto 0.75rem" }} />
      <p style={{ color: "var(--text-secondary)", fontWeight: 600 }}>XAI Report Not Available</p>
      <p style={{ fontSize: "0.875rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
        Run the XAI analyzer to generate explanations:
      </p>
      <code style={{ display: "block", marginTop: "0.75rem", padding: "0.6rem 1rem", background: "rgba(255,255,255,0.05)", borderRadius: 6, fontSize: "0.8rem", color: "var(--accent-cyan)", fontFamily: "var(--font-mono)", textAlign: "left" }}>
        python scripts/run_xai.py --model artifacts/experiments/ppo/[run]/model_final.zip
      </code>
    </div>
  );

  if (data.length === 0) return (
    <div className="empty-state"><Brain size={32} /><p>No XAI data found.</p></div>
  );

  return (
    <div>
      {data.map((expl) => (
        <JunctionCard key={expl.tls_id} expl={expl} />
      ))}
    </div>
  );
};
