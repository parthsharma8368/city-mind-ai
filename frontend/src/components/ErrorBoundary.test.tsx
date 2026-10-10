import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { ErrorBoundary } from "./ErrorBoundary";

function Boom({ explode }: { explode: boolean }): React.ReactElement {
  if (explode) throw new Error("kaboom");
  return <p>fine</p>;
}

afterEach(() => vi.restoreAllMocks());

describe("ErrorBoundary", () => {
  it("shows the error and keeps siblings outside the boundary (e.g. the nav) mounted", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    render(
      <div>
        <nav>nav stays</nav>
        <ErrorBoundary compact resetKey="/a"><Boom explode /></ErrorBoundary>
      </div>,
    );
    expect(screen.getByRole("alert").textContent).toContain("kaboom");
    expect(screen.getByText("nav stays")).toBeInTheDocument();
  });

  it("clears the error when resetKey (route) changes", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const { rerender } = render(<ErrorBoundary resetKey="/a"><Boom explode /></ErrorBoundary>);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    rerender(<ErrorBoundary resetKey="/b"><Boom explode={false} /></ErrorBoundary>);
    expect(screen.getByText("fine")).toBeInTheDocument();
  });
});
