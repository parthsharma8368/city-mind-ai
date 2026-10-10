import { describe, it, expect } from "vitest";
import css from "./index.css?raw";
import app from "../App.tsx?raw";

// jsdom has no layout engine, so this is a CSS-level sanity check that the
// phone breakpoint which lets the navbar wrap (frontend-8) stays in place.
const phoneBlock = /@media \(max-width: 600px\) \{([\s\S]*?)\n\}\n/.exec(css)?.[1] ?? "";

describe("phone breakpoint", () => {
  it("lets the navbar and nav links wrap and shrinks main padding", () => {
    expect(phoneBlock).toMatch(/\.navbar\s*\{[^}]*flex-wrap:\s*wrap/);
    expect(phoneBlock).toMatch(/\.nav-links\s*\{[^}]*flex-wrap:\s*wrap/);
    expect(phoneBlock).toMatch(/\.app-main\s*\{[^}]*padding:\s*1rem/);
  });
  it("main padding is a class, not an inline fixed 2rem", () => {
    expect(app).toContain('className="app-main"');
    expect(app).not.toMatch(/padding:\s*"2rem"/);
  });
});
