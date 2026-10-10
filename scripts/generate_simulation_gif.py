"""generate_simulation_gif.py — Generates an animated GIF of a recorded CityMind AI traffic simulation.

Purely illustrative: it replays ONE recorded trajectory (default: the IPPO
seed-42 run) and says nothing about controller quality. For controller
comparisons use artifacts/metrics/drain_scores.* (scripts/rescore_drain_metrics.py).

Usage:
  python scripts/generate_simulation_gif.py [--trajectory PATH] [--output PATH] [--overwrite]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def parse_args():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectory", default=str(root / "artifacts" / "trajectories" / "ippo_seed42_1.0x.parquet"))
    p.add_argument("--output", default=str(root / "artifacts" / "traffic_simulation_showcase.gif"))
    p.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    geojson_path = root / "data" / "processed" / "network.geojson"
    tls_path = root / "artifacts" / "scenario" / "tls_metadata.json"
    # NOTE: this depicts a recorded run for illustration only. It is NOT a claim that
    # the policy is the best controller: on the drain-aware protocol no RL controller
    # beats the SUMO-actuated baseline (see artifacts/metrics/drain_scores.md).
    traj_path = Path(args.trajectory)
    out_gif = Path(args.output)
    if out_gif.exists() and not args.overwrite:
        sys.exit(f"{out_gif} exists; pass --overwrite to replace it")

    print("Loading network and trajectory data...")
    with open(geojson_path) as f:
        geo = json.load(f)

    with open(tls_path) as f:
        tls_meta_list = json.load(f)
        tls_meta = {t["tls_id"]: t for t in tls_meta_list}

    df = pd.read_parquet(traj_path)

    # Ingolstadt centered bounding box
    min_x, max_x = 209500.0, 218500.0
    min_y, max_y = 446500.0, 455500.0

    W, H = 840, 640
    pad = 30

    scale_x = (W - pad * 2) / (max_x - min_x)
    scale_y = (H - pad * 2) / (max_y - min_y)
    scale = min(scale_x, scale_y)
    offset_x = pad + (W - pad * 2 - (max_x - min_x) * scale) / 2
    offset_y = pad + (H - pad * 2 - (max_y - min_y) * scale) / 2

    def to_screen(x, y):
        sx = offset_x + (x - min_x) * scale
        sy = H - offset_y - (y - min_y) * scale
        return sx, sy

    # Pre-filter road LineStrings
    roads = []
    for feat in geo["features"]:
        if feat["geometry"]["type"] == "LineString":
            pts = [to_screen(c[0], c[1]) for c in feat["geometry"]["coordinates"]]
            is_tls = feat["properties"].get("is_tls", False)
            roads.append((pts, is_tls))

    # Pre-filter TLS Junction points
    tls_points = []
    for feat in geo["features"]:
        if feat["geometry"]["type"] == "Point" and feat["properties"].get("is_tls"):
            coords = feat["geometry"]["coordinates"]
            sx, sy = to_screen(coords[0], coords[1])
            tls_id = str(feat["properties"].get("tls_id") or "0")
            tls_points.append((sx, sy, tls_id))

    # Pre-render base road network onto a static background Image
    base_img = Image.new("RGB", (W, H), (8, 12, 23))
    base_draw = ImageDraw.Draw(base_img)

    # Subtle grid lines
    for gx in range(0, W, 40):
        base_draw.line([(gx, 0), (gx, H)], fill=(15, 22, 38))
    for gy in range(0, H, 40):
        base_draw.line([(0, gy), (W, gy)], fill=(15, 22, 38))

    # Roads
    for pts, is_tls in roads:
        color = (40, 90, 140) if is_tls else (25, 40, 65)
        width = 2 if is_tls else 1
        base_draw.line(pts, fill=color, width=width)

    # Select up to 60 frames. IPPO's trajectory is recorded every 5s
    # (matching its decision interval) rather than PPO's 1s, so the window
    # needs to be wider to get a comparable number of frames.
    times = sorted(df["time"].unique())
    selected_times = [t for t in times if 10.0 <= t <= 310.0][:60]
    print(f"Rendering {len(selected_times)} frames...")

    frames = []
    for idx, sim_t in enumerate(selected_times):
        frame = base_img.copy()
        draw = ImageDraw.Draw(frame)

        # 1. Vehicles
        sub = df[df["time"] == sim_t]
        for _, row in sub.iterrows():
            vx, vy = to_screen(row["x"], row["y"])
            if 0 <= vx < W and 0 <= vy < H:
                spd = row["speed"]
                if spd < 2.0:
                    car_color = (236, 72, 153)  # Hot Pink (Queued)
                    r = 3.0
                elif spd > 8.0:
                    car_color = (56, 189, 248)  # Electric Cyan (Fast)
                    r = 2.4
                else:
                    car_color = (168, 85, 247)  # Lavender (Moving)
                    r = 2.4
                draw.ellipse([vx - r, vy - r, vx + r, vy + r], fill=car_color)

        # 2. Traffic Lights (Rendered ON TOP of vehicles)
        for sx, sy, tls_id in tls_points:
            if not (0 <= sx < W and 0 <= sy < H):
                continue

            # Calculate phase color from InTAS metadata
            signal_color = (16, 185, 129)  # Green
            if tls_id in tls_meta:
                meta = tls_meta[tls_id]
                phases = meta.get("phases", [])
                total_cycle = sum(p.get("duration", 5.0) for p in phases) or 60.0
                t_cycle = sim_t % total_cycle
                acc = 0.0
                active_phase = 0
                active_state = ""
                for p in phases:
                    dur = p.get("duration", 5.0)
                    if t_cycle < acc + dur:
                        active_phase = p.get("index", 0)
                        active_state = p.get("state", "").lower()
                        break
                    acc += dur

                if "y" in active_state:
                    signal_color = (245, 158, 11)  # Yellow
                elif active_phase == 0:
                    signal_color = (16, 185, 129)  # Green
                elif active_phase % 2 == 1:
                    signal_color = (245, 158, 11)  # Yellow
                else:
                    signal_color = (239, 68, 68)   # Red
            else:
                # Deterministic fallback
                h = sum(ord(c) for c in tls_id)
                mod = (sim_t + h * 3.7) % 60.0
                if mod < 28:
                    signal_color = (16, 185, 129)
                elif mod < 32:
                    signal_color = (245, 158, 11)
                else:
                    signal_color = (239, 68, 68)

            # Draw outer collar (dark bezel + white ring)
            draw.ellipse([sx - 5.0, sy - 5.0, sx + 5.0, sy + 5.0], fill=(5, 8, 17), outline=(255, 255, 255), width=1)
            # Draw LED lens
            draw.ellipse([sx - 3.2, sy - 3.2, sx + 3.2, sy + 3.2], fill=signal_color)
            # Specular highlight
            draw.point((int(sx - 1), int(sy - 1)), fill=(255, 255, 255))

        # 3. Header & Telemetry Overlay
        draw.rectangle([15, 15, 340, 52], fill=(12, 18, 32), outline=(30, 45, 70))
        draw.text((25, 22), f"CityMind AI · IPPO Policy · InTAS Ingolstadt Replay", fill=(148, 163, 184))
        draw.text((25, 36), f"Sim Time: {sim_t:.1f}s  |  Active Vehicles: {len(sub)}", fill=(56, 189, 248))

        # Legend Overlay bottom-right
        draw.rectangle([W - 360, H - 38, W - 15, H - 12], fill=(12, 18, 32), outline=(30, 45, 70))
        draw.ellipse([W - 350, H - 28, W - 342, H - 20], fill=(16, 185, 129))
        draw.text((W - 338, H - 29), "Green", fill=(16, 185, 129))
        draw.ellipse([W - 295, H - 28, W - 287, H - 20], fill=(245, 158, 11))
        draw.text((W - 283, H - 29), "Yellow", fill=(245, 158, 11))
        draw.ellipse([W - 240, H - 28, W - 232, H - 20], fill=(239, 68, 68))
        draw.text((W - 228, H - 29), "Red", fill=(239, 68, 68))
        draw.text((W - 195, H - 29), "• Fast  • Moving  • Queued", fill=(148, 163, 184))

        frames.append(frame)

    print(f"Saving animated GIF to {out_gif}...")
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=120,  # 120ms per frame ~ 8.3 fps
        loop=0,
        optimize=True,
    )
    print("GIF generated successfully!")

if __name__ == "__main__":
    main()
