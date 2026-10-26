import { describe, expect, it } from "vitest";
import type { PolicyDTO } from "../api/client";
import { buildMlp } from "./forward";
import { OBS_SIZE } from "./junction";
import { buildScene, fitScale, pickActiveUnits, project, SHOWN, type Camera } from "./scene";

/** A random-looking but deterministic 22 -> 24 -> 24 -> out network with the real shapes' proportions. */
function fakeDto(hidden = 24): PolicyDTO {
  let s = 1;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5) * 0.8;
  const dense = (rows: number, cols: number) => ({ shape: [rows, cols] as [number, number], w: Array.from({ length: rows * cols }, rnd), b: Array.from({ length: rows }, rnd) });
  const net = (out: number) => ({ l1: dense(hidden, OBS_SIZE), l2: dense(hidden, hidden), out: dense(out, hidden) });
  return { controller: "t", label: "T", obs_names: Array.from({ length: OBS_SIZE }, (_, i) => `o${i}`), actions: ["keep", "advance"], activation: "tanh", actor: net(2), critic: net(1) };
}

describe("scene", () => {
  const mlp = buildMlp(fakeDto());
  const { picks, valueRange } = pickActiveUnits(mlp, 120, SHOWN);
  const scene = buildScene(mlp, picks, valueRange);

  it("picks SHOWN distinct, sorted units per hidden layer", () => {
    for (const list of [picks.actor.h1, picks.actor.h2, picks.critic.h1, picks.critic.h2]) {
      expect(list).toHaveLength(SHOWN);
      expect(new Set(list).size).toBe(SHOWN);
      expect([...list].sort((a, b) => a - b)).toEqual(list);
    }
    expect(valueRange[1]).toBeGreaterThan(valueRange[0]);
  });

  it("draws 22 inputs, 16+16 per branch and 2+1 outputs", () => {
    const count = (layer: number, branch?: string) => scene.nodes.filter((n) => n.layer === layer && (!branch || n.branch === branch)).length;
    expect(count(0)).toBe(OBS_SIZE);
    expect(count(1, "actor")).toBe(SHOWN);
    expect(count(2, "critic")).toBe(SHOWN);
    expect(count(3, "actor")).toBe(2);
    expect(count(3, "critic")).toBe(1);
  });

  it("places the actor above the critic and the layers left to right", () => {
    const y = (branch: string) => scene.nodes.filter((n) => n.layer === 1 && n.branch === branch).map((n) => n.pos.y);
    expect(Math.min(...y("actor"))).toBeGreaterThan(Math.max(...y("critic")));
    const x = (layer: number) => scene.nodes.find((n) => n.layer === layer)!.pos.x;
    expect(x(0)).toBeLessThan(x(1));
    expect(x(1)).toBeLessThan(x(2));
    expect(x(2)).toBeLessThan(x(3));
  });

  it("connects each layer to the next with the model's real weights", () => {
    expect(scene.edges.length).toBeGreaterThan(300);
    const e = scene.edges.find((edge) => edge.block === 1 && edge.branch === "actor")!;
    const from = scene.nodes[e.from];
    const to = scene.nodes[e.to];
    expect(to.layer).toBe(from.layer + 1);
    expect(e.w).toBe(mlp.actor.l2.w[to.unit * mlp.actor.l2.cols + from.unit]);
    for (const edge of scene.edges) expect(edge.branch === scene.nodes[edge.to].branch).toBe(true);
  });
});

describe("project", () => {
  const cam: Camera = { yaw: 0, pitch: 0, dist: 16 };
  const W = 800;
  const H = 420;

  it("puts the origin at the middle of the canvas", () => {
    const p = project({ x: 0, y: 0, z: 0 }, cam, W, H);
    expect(p.sx).toBeCloseTo(W / 2);
    expect(p.sy).toBeCloseTo(H / 2);
    expect(p.scale).toBeCloseTo(fitScale(W, H));
  });

  it("makes nearer points larger and farther points smaller", () => {
    const near = project({ x: 0, y: 0, z: -4 }, cam, W, H);
    const far = project({ x: 0, y: 0, z: 4 }, cam, W, H);
    expect(near.scale).toBeGreaterThan(far.scale);
    expect(near.depth).toBeLessThan(far.depth);
  });

  it("moves up the screen for positive y and right for positive x", () => {
    const p = project({ x: 2, y: 1, z: 0 }, cam, W, H);
    expect(p.sx).toBeGreaterThan(W / 2);
    expect(p.sy).toBeLessThan(H / 2);
  });

  it("a quarter turn brings the left end of the figure to the front or back", () => {
    const turned = project({ x: 5, y: 0, z: 0 }, { ...cam, yaw: Math.PI / 2 }, W, H);
    expect(turned.sx).toBeCloseTo(W / 2, 0);
    expect(turned.depth).not.toBeCloseTo(1, 1);
  });
});
