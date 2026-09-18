import React from "react";
import { AlertTriangle } from "lucide-react";

interface Props {
  children: React.ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Top-level render-error guard. Without this, an uncaught error in any page
 * (a bad API payload, a null-deref in a chart) unmounts the entire app to a
 * blank white screen with no indication of what happened.
 */
export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("Unhandled UI error:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div
          role="alert"
          style={{
            minHeight: "100vh", display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center", gap: "0.75rem",
            padding: "2rem", textAlign: "center", background: "#050811", color: "#e5e9f0",
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
