import { describe, expect, it } from "vitest";
import type { PolicyDTO } from "../api/client";
import { buildMlp, denseForward, evaluate, parameterCount, runNet, softmax } from "./forward";

/** A 2 -> 2 -> 2 -> out network small enough to check by hand. */
const dense = (rows: number, cols: number, w: number[], b: number[]) => ({ shape: [rows, cols] as [number, number], w, b });
const dto = (outRows: number): PolicyDTO => ({
  controller: "t", label: "T", obs_names: ["a", "b"], actions: ["keep", "advance"], activation: "tanh",
  actor: { l1: dense(2, 2, [1, 0, 0, 1], [0, 0]), l2: dense(2, 2, [1, 1, 1, -1], [0, 0]), out: dense(outRows, 2, outRows === 2 ? [0, 1, 0, -1] : [2, 0], new Array(outRows).fill(0)) },
  critic: { l1: dense(2, 2, [1, 0, 0, 1], [0, 0]), l2: dense(2, 2, [1, 1, 1, -1], [0, 0]), out: dense(1, 2, [2, 0], [0.5]) },
});

describe("forward pass", () => {
  it("applies W x + b with tanh on the hidden layers and none on the output", () => {
    const net = buildMlp(dto(2)).actor;
    const trace = runNet(net, [0.5, -0.25]);
    expect(trace.h1[0]).toBeCloseTo(Math.tanh(0.5));
    expect(trace.h1[1]).toBeCloseTo(Math.tanh(-0.25));
    expect(trace.h2[0]).toBeCloseTo(Math.tanh(Math.tanh(0.5) + Math.tanh(-0.25)));
    expect(trace.out[0]).toBeCloseTo(trace.h2[1]);
    expect(trace.out[1]).toBeCloseTo(-trace.h2[1]);
  });

  it("softmax sums to one and is stable for large scores", () => {
    const p = softmax([1000, 1001]);
    expect(p[0] + p[1]).toBeCloseTo(1);
    expect(p[1]).toBeGreaterThan(p[0]);
    expect(softmax([0, 0])).toEqual([0.5, 0.5]);
  });

  it("evaluate returns the actor's choice and the critic's value", () => {
    const mlp = buildMlp(dto(2));
    const e = evaluate(mlp, [0.5, -0.25]);
    expect(e.probs[0] + e.probs[1]).toBeCloseTo(1);
    expect(e.value).toBeCloseTo(runNet(mlp.critic, [0.5, -0.25]).out[0]);
    expect(e.action).toBe(e.probs[1] > e.probs[0] ? 1 : 0);
  });

  it("counts weights and biases", () => {
    expect(parameterCount(buildMlp(dto(2)).actor)).toBe(4 + 2 + 4 + 2 + 4 + 2);
  });

  it("rejects weights that do not match their shape or chain", () => {
    const bad = dto(2);
    bad.actor.l2.w = [1, 2, 3];
    expect(() => buildMlp(bad)).toThrow(/do not match/);
    const chain = dto(2);
    chain.actor.out = dense(2, 3, [0, 0, 0, 0, 0, 0], [0, 0]);
    expect(() => buildMlp(chain)).toThrow(/chain/);
  });

  it("denseForward writes into the given buffer", () => {
    const out = new Float32Array(1);
    const d = { rows: 1, cols: 2, w: Float32Array.of(2, 3), b: Float32Array.of(1) };
    expect(denseForward(d, [1, 1], out, false)[0]).toBe(6);
  });
});
