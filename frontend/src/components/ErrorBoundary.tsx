import React from "react";
import { AlertTriangle } from "lucide-react";

interface Props {
  children: React.ReactNode;
  /** Page-level mode: no full-viewport takeover, so the nav stays usable. */
  compact?: boolean;
  /** When this value changes (e.g. the route path) a caught error is cleared. */
  resetKey?: string;
}

interface State {
  error: Error | null;
  resetKey?: string;
}

/**
 * Top-level render-error guard. Without this, an uncaught error in any page
 * (a bad API payload, a null-deref in a chart) unmounts the entire app to a
 * blank white screen with no indication of what happened.
 */
export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null, resetKey: this.props.resetKey };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  // Clear a caught error when resetKey (e.g. the route path) changes.
  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    if (props.resetKey !== state.resetKey) return { error: null, resetKey: props.resetKey };
    return null;
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("Unhandled UI error:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      const compact = this.props.compact;
      return (
        <div
          role="alert"
          style={{
            minHeight: compact ? "40vh" : "100vh", display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center", gap: "0.75rem",
            padding: "2rem", textAlign: "center", background: compact ? "transparent" : "#050811", color: "#e5e9f0",
          }}
        >
          <AlertTriangle size={32} color="#ef4444" aria-hidden="true" />
          <h1 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Something went wrong</h1>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8", maxWidth: 480 }}>
            {this.state.error.message || "The dashboard hit an unexpected error while rendering."}
          </p>
          <button
            className="btn btn-sm"
            onClick={() => this.setState({ error: null })}
            style={{ marginTop: "0.5rem" }}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
