import React, { useEffect, useState } from "react";
import { AlertTriangle, Brain, Info, Grid, Maximize2, Sparkles, Layers, Image as ImageIcon } from "lucide-react";
import { XAIPanel } from "../components/XAIPanel";
import { api, API_BASE, isAbortError } from "../api/client";
import type { XAIMeta } from "../api/client";
import { CONTROLLER_LABELS } from "../constants/controllers";

const SAMPLE_DRIVER_JUNCTIONS = ["1863241632", "2033094557", "214620430"];
const XAI_CONTROLLERS = ["ppo", "ippo"] as const;

function ImageLoadError({ label }: { label: string }) {
  return (
    <div style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)" }}>
      <AlertTriangle size={24} color="var(--signal-yellow)" style={{ margin: "0 auto 0.5rem" }} />
      <p style={{ fontSize: "0.85rem" }}>Couldn't load the {label}.</p>
      <p style={{ fontSize: "0.75rem", marginTop: "0.25rem" }}>Run the XAI analyzer to generate it, then refresh.</p>
    </div>
  );
}

export const XAIPage: React.FC = () => {
  const [controller, setController] = useState<typeof XAI_CONTROLLERS[number]>("ppo");
  const [rawMeta, setMeta] = useState<XAIMeta | null>(null);
  // Only trust meta that describes the controller currently selected.
  const meta = rawMeta?.controller === controller ? rawMeta : null;
  const [activeView, setActiveView] = useState<"attributions" | "heatmap" | "drivers">("attributions");
  const [selectedDriverJunction, setSelectedDriverJunction] = useState("1863241632");
  const [heatmapExpanded, setHeatmapExpanded] = useState(false);
  const [heatmapFailed, setHeatmapFailed] = useState(false);
  const [driverPlotFailed, setDriverPlotFailed] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    api.getXAIMeta(controller, ctrl.signal)
      .then((m) => { if (!ctrl.signal.aborted) setMeta(m); })
      .catch((e) => { if (!ctrl.signal.aborted && !isAbortError(e)) setMeta(null); });
    return () => ctrl.abort();
  }, [controller]);

  const heatmapUrl = `${API_BASE}/xai/coordination-heatmap?controller=${encodeURIComponent(controller)}`;
  const driverPlotUrl = `${API_BASE}/xai/driver-plot/${encodeURIComponent(selectedDriverJunction)}?controller=${encodeURIComponent(controller)}`;
  const isEmpirical = meta?.coordination_evidence_type === "empirical_correlation";

  return (
    <div style={{ animation: "fade-in 0.4s ease both" }}>
      {/* Header */}
      <div style={{
        marginBottom: "1.5rem",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-end",
        flexWrap: "wrap",
        gap: "0.75rem"
      }}>
        <div>
          <h2 className="section-title">Explainable AI (XAI) Analysis</h2>
          <p className="section-subtitle">
            <Brain size={13} style={{ display: "inline", verticalAlign: "middle", marginRight: 4 }} />
            Integrated-Gradients feature attribution & inter-intersection coordination analysis
          </p>
        </div>

        {/* Controller Toggle */}
        <div role="group" aria-label="Controller to explain" style={{
          display: "flex", background: "rgba(255,255,255,0.05)", borderRadius: 20, padding: 3,
          border: "1px solid var(--border-subtle)",
        }}>
          {XAI_CONTROLLERS.map((id) => (
            <button
              key={id}
              className={`btn btn-sm ${controller === id ? "btn-primary" : ""}`}
              style={{
                borderRadius: 18, padding: "0.35rem 0.9rem", fontSize: "0.8rem",
                background: controller === id ? "var(--accent-cyan)" : "transparent",
                color: controller === id ? "#060910" : "var(--text-secondary)",
                fontWeight: controller === id ? 700 : 500,
              }}
              onClick={() => { setController(id); setHeatmapFailed(false); setDriverPlotFailed(false); }}
              aria-pressed={controller === id}
            >
              {CONTROLLER_LABELS[id] ?? id}
            </button>
          ))}
        </div>

        {/* View Switcher Tabs */}
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <button
            className={`btn btn-sm ${activeView === "attributions" ? "btn-primary" : ""}`}
            style={{
              borderRadius: 20,
              padding: "0.4rem 0.9rem",
              fontSize: "0.8rem",
              border: activeView === "attributions" ? "1px solid var(--accent-cyan)" : "1px solid var(--border-subtle)",
              background: activeView === "attributions" ? "rgba(34,211,238,0.15)" : "rgba(255,255,255,0.03)",
              color: activeView === "attributions" ? "var(--accent-cyan)" : "var(--text-secondary)",
            }}
            onClick={() => setActiveView("attributions")}
            aria-pressed={activeView === "attributions"}
          >
            <Layers size={13} /> Junction Attributions
          </button>
          <button
            className={`btn btn-sm ${activeView === "heatmap" ? "btn-primary" : ""}`}
            style={{
              borderRadius: 20,
              padding: "0.4rem 0.9rem",
              fontSize: "0.8rem",
              border: activeView === "heatmap" ? "1px solid var(--accent-cyan)" : "1px solid var(--border-subtle)",
              background: activeView === "heatmap" ? "rgba(34,211,238,0.15)" : "rgba(255,255,255,0.03)",
              color: activeView === "heatmap" ? "var(--accent-cyan)" : "var(--text-secondary)",
            }}
            onClick={() => setActiveView("heatmap")}
            aria-pressed={activeView === "heatmap"}
          >
            <Grid size={13} /> Coordination Heatmap
          </button>
          <button
            className={`btn btn-sm ${activeView === "drivers" ? "btn-primary" : ""}`}
            style={{
              borderRadius: 20,
              padding: "0.4rem 0.9rem",
              fontSize: "0.8rem",
              border: activeView === "drivers" ? "1px solid var(--accent-cyan)" : "1px solid var(--border-subtle)",
              background: activeView === "drivers" ? "rgba(34,211,238,0.15)" : "rgba(255,255,255,0.03)",
              color: activeView === "drivers" ? "var(--accent-cyan)" : "var(--text-secondary)",
            }}
            onClick={() => setActiveView("drivers")}
            aria-pressed={activeView === "drivers"}
          >
            <ImageIcon size={13} /> Driver Plots
          </button>
        </div>
      </div>

      {/* Info banner */}
      <div style={{
        padding: "0.85rem 1.25rem",
        background: "rgba(59,130,246,0.08)",
        border: "1px solid rgba(59,130,246,0.2)",
        borderRadius: 8,
        display: "flex",
        gap: "0.75rem",
        marginBottom: "1.5rem",
        alignItems: "flex-start",
      }}>
        <Info size={16} color="var(--accent-blue)" style={{ flexShrink: 0, marginTop: 2 }} />
        <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
          <strong style={{ color: "var(--text-primary)" }}>How XAI Works in CityMind:</strong>{" "}
          {meta?.explains ?? (controller === "ippo"
            ? "IPPO — one shared policy invoked independently per intersection, each seeing only its own local observation."
            : "The centralized PPO agent processes full network observations (queue lengths, elapsed green times, downstream neighbor states).")}
          {" "}Integrated Gradients backpropagates from each junction's switch-decision logit to every input feature
          along a straight-line path from a zeroed baseline, revealing whether signal timing actions are driven by
          immediate local queues or downstream pressure.
          {controller === "ippo" && (
            <>
              {" "}<strong style={{ color: "var(--text-primary)" }}>Note:</strong> IPPO's per-junction attribution
              (Junction Attributions / Driver Plots tabs) reflects only that junction's own 10-dimensional observation —
              there's no explicit neighboring-junction identity in it, only one aggregate downstream-queue signal. The
              Coordination Heatmap tab below is computed differently for this reason — see its own description.
            </>
          )}
        </div>
      </div>

      {/* ── View 1: Feature Attributions per Junction ──────────────────────── */}
      {activeView === "attributions" && (
        <div>
          <XAIPanel key={controller} maxItems={20} controller={controller} />
        </div>
      )}

      {/* ── View 2: Coordination Heatmap ───────────────────────────────────── */}
      {activeView === "heatmap" && (
        <div className="glass-card" style={{ padding: "1.5rem", animation: "fade-in 0.3s ease both" }}>
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "1.25rem",
            flexWrap: "wrap",
            gap: "0.5rem"
          }}>
            <div>
              <h3 style={{ fontSize: "1rem", fontWeight: 600, display: "flex", alignItems: "center", gap: 6 }}>
                <Sparkles size={16} color="var(--accent-cyan)" /> {meta?.coordination_evidence_label ?? "Inter-Intersection Coordination Heatmap"}
                {meta && (
                  <span className={isEmpirical ? "badge-yellow" : "badge-blue"} style={{ fontSize: "0.65rem", marginLeft: 4 }}>
                    {isEmpirical ? "Empirical" : "Model Attribution"}
                  </span>
                )}
              </h3>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
                {meta?.coordination_evidence_description ?? "Shows cross-correlation of Integrated-Gradients attributions between controlled TLS junctions"}
              </p>
            </div>
            <button
              className="btn btn-sm"
              onClick={() => setHeatmapExpanded(!heatmapExpanded)}
              aria-pressed={heatmapExpanded}
              style={{ display: "flex", alignItems: "center", gap: 5 }}
            >
              <Maximize2 size={13} /> {heatmapExpanded ? "Standard Size" : "Full Resolution"}
            </button>
          </div>

          <div style={{
            background: "rgba(10,15,28,0.6)",
            borderRadius: 8,
            padding: "1rem",
            border: "1px solid var(--border-subtle)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            overflow: "hidden"
          }}>
            {heatmapFailed ? (
              <ImageLoadError label="coordination heatmap" />
            ) : (
              <img
                src={heatmapUrl}
                alt="Inter-Intersection Coordination Heatmap"
                style={{
                  maxWidth: "100%",
                  maxHeight: heatmapExpanded ? "85vh" : "500px",
                  objectFit: "contain",
                  borderRadius: 4,
                  boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
                }}
                onError={() => setHeatmapFailed(true)}
              />
            )}
          </div>

          <div style={{ marginTop: "1rem", fontSize: "0.82rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
            <span style={{ color: "var(--accent-cyan)", fontWeight: 600 }}>Key Insight: </span>
            {isEmpirical
              ? "This measures correlated switch timing between geographically-nearby junctions in the recorded episode — real behavioral evidence, but not the same as PPO's model-attribution matrix above, since IPPO's architecture has no gradient path between one junction's decision and a specific neighbor's state. Sparse, weak correlation here is expected and consistent with IPPO agents acting largely independently — a case for the coordination-focused RL alternatives (MAPPO, CoLight) as a next step."
              : "Strong off-diagonal intensity indicates cooperative emergent behavior where upstream junctions hold or cycle phases to meter vehicles arriving at bottleneck intersections downstream."}
          </div>
        </div>
      )}

      {/* ── View 3: Driver Attribution Plots ────────────────────────────────── */}
      {activeView === "drivers" && (
        <div className="glass-card" style={{ padding: "1.5rem", animation: "fade-in 0.3s ease both" }}>
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "1.25rem",
            flexWrap: "wrap",
            gap: "0.75rem"
          }}>
            <div>
              <h3 style={{ fontSize: "1rem", fontWeight: 600, display: "flex", alignItems: "center", gap: 6 }}>
                <Brain size={16} color="var(--accent-cyan)" /> Junction Driver Plots
              </h3>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
                Detailed visual feature waterfall charts for key arterial intersections
              </p>
            </div>

            {/* Junction Selector */}
            <div style={{ display: "flex", gap: "0.4rem" }}>
              {SAMPLE_DRIVER_JUNCTIONS.map((jId) => (
                <button
                  key={jId}
                  className={`btn btn-sm ${selectedDriverJunction === jId ? "btn-primary" : ""}`}
                  style={{
                    fontSize: "0.75rem",
                    fontFamily: "var(--font-mono)",
                    background: selectedDriverJunction === jId ? "var(--accent-cyan)" : "rgba(255,255,255,0.04)",
                    color: selectedDriverJunction === jId ? "#060910" : "var(--text-secondary)",
                  }}
                  aria-pressed={selectedDriverJunction === jId}
                  onClick={() => { setSelectedDriverJunction(jId); setDriverPlotFailed(false); }}
                >
                  TLS {jId}
                </button>
              ))}
            </div>
          </div>

          <div style={{
            background: "rgba(10,15,28,0.6)",
            borderRadius: 8,
            padding: "1rem",
            border: "1px solid var(--border-subtle)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center"
          }}>
            {driverPlotFailed ? (
              <ImageLoadError label={`driver plot for TLS ${selectedDriverJunction}`} />
            ) : (
              <img
                key={selectedDriverJunction}
                src={driverPlotUrl}
                alt={`Driver plot for TLS ${selectedDriverJunction}`}
                style={{
                  maxWidth: "100%",
                  maxHeight: "520px",
                  objectFit: "contain",
                  borderRadius: 4,
                  boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
                }}
                onError={() => setDriverPlotFailed(true)}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};
