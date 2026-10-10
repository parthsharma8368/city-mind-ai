"""
!!! UNREVIEWED / DO NOT PUBLISH !!!
===================================
This script renders SYNTHETIC and STALE content that must not be presented as
empirical results:
  * the "Empirical Results" table is hard-coded from the pre-2026-09-21
    midnight window (PPO 13.47 s, Max-Pressure 37.38 s, "Fixed-Time" 3.02 s,
    "IPPO (Optimal 40M)" 4.15 s, "Fastest travel speed!") and contradicts the
    repository's current finding that no RL controller beats the SUMO-actuated
    baseline (artifacts/metrics/drain_scores.md);
  * the "live" observation bars are sine waves, the action probabilities and
    signal colours are derived from sim_time % 60, and the "40M steps"
    progress bar is a fake counter;
  * it says "12-dimensional" observation (the environment is 18-dimensional).
It was moved here from scripts/ (audit finding eval-scripts-9) and refuses to
run unless --allow-synthetic is passed; any output must be labelled
"illustrative animation, synthetic values". Do not commit its outputs
(artifacts/rl_training_explanation.webp, artifacts/rl_training_visualizer.*).
See scripts/_unreviewed/README.md.
"""
# generate_training_video.py
# ==========================
# Generates a comprehensive animated video demonstrating how CityMind AI trains
# its reinforcement learning policy on the live microscopic SUMO traffic environment.
#
# Visualizes:
# 1. Live microscopic traffic network with vehicle physics & queues
# 2. Normalized 12-dimensional local observation vector extraction
# 3. Deep neural network (Actor-Critic MLP) forward pass & action logits
# 4. Deterministic safety shield verification (min-green & clearance)
# 5. Localized reward computation solving the credit assignment problem
# 6. PPO rollout buffer collection & clipped surrogate loss optimization
# 7. Policy progression from early training (7M steps) to optimal policy (40M steps)
#
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_VIDEO = ROOT / "artifacts" / "rl_training_explanation.webp"

def get_font(size: int, bold: bool = False):
    try:
        font_path = "/System/Library/Fonts/Helvetica.ttc"
        index = 1 if bold else 0
        return ImageFont.truetype(font_path, size=size, index=index)
    except Exception:
        return ImageFont.load_default()

def main():
    global OUTPUT_VIDEO
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="UNREVIEWED: renders synthetic values (see module docstring)")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="acknowledge that every number/animation is synthetic and not empirical")
    ap.add_argument("--output", default=str(OUTPUT_VIDEO))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not args.allow_synthetic:
        sys.exit("Refusing to run: this script renders synthetic/stale numbers as 'Empirical Results'. "
                 "Read the header of this file; pass --allow-synthetic to render an ILLUSTRATIVE animation.")
    out = Path(args.output)
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} exists; pass --overwrite")
    OUTPUT_VIDEO = out
    print("[Video Generator] Initializing training explanation video render...")

    W, H = 1000, 680
    font_title = get_font(18, bold=True)
    font_section = get_font(14, bold=True)
    font_body = get_font(12, bold=False)
    font_mono = get_font(11, bold=False)
    font_badge = get_font(10, bold=True)
    font_huge = get_font(26, bold=True)

    # Load trajectory data
    traj_path = ROOT / "artifacts" / "trajectories" / "ippo_seed42_1.0x.parquet"
    df = pd.read_parquet(traj_path)

    # Load network geojson
    with open(ROOT / "data" / "processed" / "network.geojson") as f:
        geo = json.load(f)

    # Focus on an active intersection in Ingolstadt
    # Let's find an active cluster of traffic lights and vehicles
    times = sorted(df["time"].unique())
    selected_times = [t for t in times if 30.0 <= t <= 450.0]
    # Sample 80 key timesteps
    sample_indices = np.linspace(0, len(selected_times) - 1, 80, dtype=int)
    chosen_times = [selected_times[i] for i in sample_indices]

    # Pre-render road network for left panel (traffic view)
    sim_w, sim_h = 510, 560
    sim_origin_x, sim_origin_y = 20, 85

    # Center bounding box for Ingolstadt core
    cx, cy = 214000.0, 451000.0
    span = 2600.0
    min_x, max_x = cx - span, cx + span
    min_y, max_y = cy - span, cy + span

    def to_sim_screen(x, y):
        sx = sim_origin_x + (x - min_x) / (max_x - min_x) * sim_w
        sy = sim_origin_y + sim_h - (y - min_y) / (max_y - min_y) * sim_h
        return sx, sy

    # Extract roads
    roads = []
    for feat in geo["features"]:
        if feat["geometry"]["type"] == "LineString":
            coords = feat["geometry"]["coordinates"]
            pts = [to_sim_screen(c[0], c[1]) for c in coords]
            is_tls = feat["properties"].get("is_tls", False)
            roads.append((pts, is_tls))

    tls_points = []
    for feat in geo["features"]:
        if feat["geometry"]["type"] == "Point" and feat["properties"].get("is_tls"):
            c = feat["geometry"]["coordinates"]
            sx, sy = to_sim_screen(c[0], c[1])
            tls_points.append((sx, sy))

    frames = []
    total_frames = len(chosen_times)
    print(f"[Video Generator] Rendering {total_frames} frames at {W}x{H}...")

    for f_idx, sim_t in enumerate(chosen_times):
        # Progress phase (0 to 1)
        phase_ratio = f_idx / total_frames

        # Create blank canvas
        img = Image.new("RGB", (W, H), (10, 15, 29))
        draw = ImageDraw.Draw(img)

        # -------------------------------------------------------------
        # 1. TOP HEADER & HUD
        # -------------------------------------------------------------
        # Header background
        draw.rectangle([0, 0, W, 65], fill=(14, 21, 38))
        draw.line([(0, 65), (W, 65)], fill=(30, 48, 80), width=1)

        # Brand Badge & Title
        draw.rounded_rectangle([20, 14, 56, 50], radius=6, fill=(14, 165, 233))
        draw.text((27, 20), "CM", fill=(255, 255, 255), font=font_title)
        draw.text((70, 14), "CityMind AI — Reinforcement Learning Training Pipeline", fill=(241, 245, 249), font=font_title)
        draw.text((70, 38), "Live Microscopic SUMO Simulation • Decentralized Parameter-Shared IPPO (98 Intersections)", fill=(148, 163, 184), font=font_mono)

        # Stage indicator pill
        if f_idx < 20:
            stage_name = "STAGE 1: Live Observation Extraction"
            stage_color = (56, 189, 248)
        elif f_idx < 40:
            stage_name = "STAGE 2: Policy Forward Pass & Safety Shield"
            stage_color = (168, 85, 247)
        elif f_idx < 60:
            stage_name = "STAGE 3: Live Environment Actuation & Reward"
            stage_color = (16, 185, 129)
        else:
            stage_name = "STAGE 4: PPO Policy Optimization (40M Steps)"
            stage_color = (245, 158, 11)

        draw.rounded_rectangle([W - 330, 18, W - 20, 46], radius=14, fill=(stage_color[0]//5, stage_color[1]//5, stage_color[2]//5), outline=stage_color, width=1)
        draw.ellipse([W - 318, 28, W - 310, 36], fill=stage_color)
        draw.text((W - 302, 24), stage_name, fill=stage_color, font=font_badge)

        # -------------------------------------------------------------
        # 2. LEFT PANEL: LIVE SIMULATION ENVIRONMENT (SUMO)
        # -------------------------------------------------------------
        # Panel container
        draw.rounded_rectangle([sim_origin_x - 5, sim_origin_y - 5, sim_origin_x + sim_w + 5, sim_origin_y + sim_h + 5], radius=10, fill=(6, 9, 18), outline=(30, 48, 80), width=1)

        # Draw road network
        for pts, is_tls in roads:
            valid_pts = [p for p in pts if sim_origin_x <= p[0] <= sim_origin_x + sim_w and sim_origin_y <= p[1] <= sim_origin_y + sim_h]
            if len(valid_pts) >= 2:
                col = (40, 75, 115) if is_tls else (20, 32, 52)
                draw.line(valid_pts, fill=col, width=2 if is_tls else 1)

        # Draw vehicles from actual parquet trajectory
        sub_df = df[df["time"] == sim_t]
        stopped_count = 0
        speed_sum = 0.0
        active_veh_count = 0

        for _, row in sub_df.iterrows():
            vx, vy = to_sim_screen(row["x"], row["y"])
            if sim_origin_x <= vx <= sim_origin_x + sim_w and sim_origin_y <= vy <= sim_origin_y + sim_h:
                spd = row["speed"]
                speed_sum += spd
                active_veh_count += 1
                if spd < 1.5:
                    stopped_count += 1
                    car_col = (236, 72, 153)  # Hot Pink (Queued)
                    r = 3.2
                elif spd > 8.0:
                    car_col = (56, 189, 248)  # Electric Cyan (High Speed)
                    r = 2.4
                else:
                    car_col = (168, 85, 247)  # Lavender (Moving)
                    r = 2.4
                draw.ellipse([vx - r, vy - r, vx + r, vy + r], fill=car_col)

        # Draw traffic light systems
        cycle_t = sim_t % 60.0
        for sx, sy in tls_points:
            if sim_origin_x <= sx <= sim_origin_x + sim_w and sim_origin_y <= sy <= sim_origin_y + sim_h:
                # Signal color cycling
                if cycle_t < 30.0:
                    tl_col = (16, 185, 129) # Green
                elif cycle_t < 34.0:
                    tl_col = (245, 158, 11) # Yellow
                else:
                    tl_col = (239, 68, 68)  # Red

                draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=(15, 23, 42), outline=(100, 116, 139), width=1)
                draw.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=tl_col)

        # Highlight a specific primary intersection for the RL Agent
        target_tls_x = sim_origin_x + sim_w // 2 + 35
        target_tls_y = sim_origin_y + sim_h // 2 - 20
        # Radar focus ring
        radar_r = 45 + 10 * math.sin(f_idx * 0.3)
        draw.ellipse([target_tls_x - radar_r, target_tls_y - radar_r, target_tls_x + radar_r, target_tls_y + radar_r], outline=(56, 189, 248, 100), width=1)
        draw.rounded_rectangle([target_tls_x - 55, target_tls_y - 65, target_tls_x + 55, target_tls_y - 45], radius=4, fill=(15, 23, 42), outline=(56, 189, 248), width=1)
        draw.text((target_tls_x - 48, target_tls_y - 62), "AGENT TLS_42", fill=(56, 189, 248), font=font_badge)

        # Simulation HUD overlay
        hud_x, hud_y = sim_origin_x + 15, sim_origin_y + 15
        draw.rounded_rectangle([hud_x, hud_y, hud_x + 220, hud_y + 105], radius=8, fill=(10, 15, 29, 230), outline=(56, 189, 248, 150), width=1)
        draw.text((hud_x + 10, hud_y + 8), f"SUMO Sim Time: {sim_t:.1f}s", fill=(241, 245, 249), font=font_section)
        draw.text((hud_x + 10, hud_y + 30), f"Active Vehicles: {active_veh_count}", fill=(148, 163, 184), font=font_mono)
        avg_spd = (speed_sum / max(1, active_veh_count)) * 3.6
        draw.text((hud_x + 10, hud_y + 48), f"Network Mean Speed: {avg_spd:.1f} km/h", fill=(56, 189, 248), font=font_mono)
        draw.text((hud_x + 10, hud_y + 66), f"Queued Vehicles: {stopped_count}", fill=(236, 72, 153), font=font_mono)
        draw.text((hud_x + 10, hud_y + 84), f"Decision Step: {int(sim_t // 5)} (Δt=5s)", fill=(16, 185, 129), font=font_mono)

        # -------------------------------------------------------------
        # 3. RIGHT PANEL (UPPER): STATE OBSERVATION & NEURAL POLICY
        # -------------------------------------------------------------
        rp_x, rp_y = sim_origin_x + sim_w + 20, sim_origin_y
        rp_w = W - rp_x - 20
        rp_h_top = 270

        draw.rounded_rectangle([rp_x, rp_y - 5, rp_x + rp_w, rp_y + rp_h_top], radius=10, fill=(14, 21, 38), outline=(30, 48, 80), width=1)
        draw.text((rp_x + 15, rp_y + 10), "Step 1 & 2: Local State & Neural Policy", fill=(241, 245, 249), font=font_section)

        # Draw 12-D Observation Bars
        obs_start_y = rp_y + 38
        obs_features = [
            ("Phase Elapsed", 0.45 + 0.3 * math.sin(f_idx * 0.15)),
            ("Mean Queue", 0.18 + 0.12 * math.cos(f_idx * 0.2)),
            ("Max Queue", 0.30 + 0.15 * math.sin(f_idx * 0.25)),
            ("Mean Speed", 0.75 + 0.15 * math.sin(f_idx * 0.1)),
            ("Downstream Q", 0.12 + 0.08 * math.cos(f_idx * 0.18)),
            ("Neighbor Q1", 0.22 + 0.1 * math.sin(f_idx * 0.12)),
        ]

        for i, (name, val) in enumerate(obs_features):
            val = max(0.05, min(0.95, val))
            row_y = obs_start_y + i * 22
            draw.text((rp_x + 15, row_y), name, fill=(148, 163, 184), font=font_mono)
            # Bar background
            draw.rounded_rectangle([rp_x + 115, row_y + 2, rp_x + 230, row_y + 12], radius=3, fill=(20, 30, 50))
            # Bar fill
            bar_color = (56, 189, 248) if "Speed" not in name else (16, 185, 129)
            if "Max" in name and val > 0.35:
                bar_color = (236, 72, 153)
            draw.rounded_rectangle([rp_x + 115, row_y + 2, rp_x + 115 + int(115 * val), row_y + 12], radius=3, fill=bar_color)
            draw.text((rp_x + 238, row_y), f"{val:.2f}", fill=(241, 245, 249), font=font_mono)

        # Neural Net Diagram (Miniature)
        nn_cx = rp_x + 330
        draw.text((nn_cx - 20, obs_start_y), "Shared Policy π_θ", fill=(168, 85, 247), font=font_badge)

        # 3 layers of nodes
        layer_xs = [nn_cx - 25, nn_cx + 25, nn_cx + 75]
        for l_idx, lx in enumerate(layer_xs):
            num_nodes = 5 if l_idx < 2 else 2
            for n_idx in range(num_nodes):
                ny = obs_start_y + 25 + n_idx * 24
                is_active = (math.sin(f_idx * 0.4 + l_idx + n_idx) > -0.2)
                node_col = (168, 85, 247) if is_active else (40, 50, 75)
                draw.ellipse([lx - 5, ny - 5, lx + 5, ny + 5], fill=node_col, outline=(148, 163, 184), width=1)

        # Action probabilities
        prob_extend = 0.82 if cycle_t < 25.0 else 0.18
        prob_switch = 1.0 - prob_extend
        act_y = rp_y + 185

        draw.rounded_rectangle([rp_x + 15, act_y, rp_x + rp_w // 2 - 10, act_y + 45], radius=6,
                               fill=(16, 185, 129, 40) if prob_extend > 0.5 else (20, 30, 50),
                               outline=(16, 185, 129) if prob_extend > 0.5 else (50, 70, 100), width=1)
        draw.text((rp_x + 25, act_y + 6), "Action 0: EXTEND", fill=(241, 245, 249), font=font_badge)
        draw.text((rp_x + 25, act_y + 22), f"Prob: {prob_extend*100:.1f}%", fill=(16, 185, 129), font=font_section)

        draw.rounded_rectangle([rp_x + rp_w // 2, act_y, rp_x + rp_w - 15, act_y + 45], radius=6,
                               fill=(239, 68, 68, 40) if prob_switch > 0.5 else (20, 30, 50),
                               outline=(239, 68, 68) if prob_switch > 0.5 else (50, 70, 100), width=1)
        draw.text((rp_x + rp_w // 2 + 10, act_y + 6), "Action 1: SWITCH", fill=(241, 245, 249), font=font_badge)
        draw.text((rp_x + rp_w // 2 + 10, act_y + 22), f"Prob: {prob_switch*100:.1f}%", fill=(239, 68, 68), font=font_section)

        # Safety Gate
        draw.rounded_rectangle([rp_x + 15, act_y + 52, rp_x + rp_w - 15, act_y + 78], radius=6, fill=(245, 158, 11, 25), outline=(245, 158, 11), width=1)
        draw.text((rp_x + 25, act_y + 58), "🛡️ Safety Shield: Min green (5s) validated • 3s Yellow clearance enforced", fill=(245, 158, 11), font=font_mono)

        # -------------------------------------------------------------
        # 4. RIGHT PANEL (LOWER): REWARD, PPO UPDATE & CONVERGENCE
        # -------------------------------------------------------------
        rp_y_bot = rp_y + rp_h_top + 15
        rp_h_bot = sim_h - rp_h_top - 15

        draw.rounded_rectangle([rp_x, rp_y_bot, rp_x + rp_w, rp_y_bot + rp_h_bot + 5], radius=10, fill=(14, 21, 38), outline=(30, 48, 80), width=1)
        draw.text((rp_x + 15, rp_y_bot + 10), "Step 3 & 4: Reward Engine & PPO Optimization", fill=(241, 245, 249), font=font_section)

        # Reward equation callout
        eq_y = rp_y_bot + 36
        draw.rounded_rectangle([rp_x + 15, eq_y, rp_x + rp_w - 15, eq_y + 44], radius=6, fill=(10, 15, 29), outline=(16, 185, 129, 120), width=1)
        draw.text((rp_x + 22, eq_y + 6), "Local Reward:", fill=(16, 185, 129), font=font_badge)
        draw.text((rp_x + 22, eq_y + 22), "r_t = -ΔWait_local - 0.3*Switch - 0.2*Spillback", fill=(241, 245, 249), font=font_mono)

        # Training Progression vs Checkpoint Metrics
        metrics_y = eq_y + 52
        draw.text((rp_x + 15, metrics_y), "SYNTHETIC / STALE values - NOT empirical:", fill=(148, 163, 184), font=font_badge)

        # Table rows of controllers
        rows = [
            ("PPO (Centralized)", "13.47s", "12.76 m/s", (236, 72, 153), "Credit assign failure"),
            ("Max-Pressure", "37.38s", "6.74 m/s", (245, 158, 11), "Heuristic gridlock"),
            ("Fixed-Time (Pre-timed)", "3.02s", "10.53 m/s", (56, 189, 248), "Tuned city baseline"),
            ("IPPO (Optimal 40M)", "4.15s", "13.02 m/s", (16, 185, 129), "Fastest travel speed!"),
        ]

        row_start_y = metrics_y + 18
        for r_i, (c_name, wait, speed, col, note) in enumerate(rows):
            ry = row_start_y + r_i * 26
            is_winner = ("IPPO" in c_name)
            if is_winner:
                draw.rounded_rectangle([rp_x + 12, ry - 2, rp_x + rp_w - 12, ry + 22], radius=4, fill=(col[0]//6, col[1]//6, col[2]//6), outline=col, width=1)
            draw.text((rp_x + 18, ry + 2), c_name, fill=(241, 245, 249) if not is_winner else col, font=font_mono)
            draw.text((rp_x + 175, ry + 2), wait, fill=col, font=font_mono)
            draw.text((rp_x + 230, ry + 2), speed, fill=col, font=font_mono)
            draw.text((rp_x + 315, ry + 2), note, fill=(148, 163, 184), font=font_badge)

        # Progress bar of 40M step training
        bar_y = rp_y_bot + rp_h_bot - 20
        draw.text((rp_x + 15, bar_y - 14), f"Training Progress: {int(phase_ratio * 40)}M / 40M Steps", fill=(148, 163, 184), font=font_badge)
        draw.rounded_rectangle([rp_x + 15, bar_y, rp_x + rp_w - 15, bar_y + 8], radius=4, fill=(20, 30, 50))
        draw.rounded_rectangle([rp_x + 15, bar_y, rp_x + 15 + int((rp_w - 30) * max(0.05, phase_ratio)), bar_y + 8], radius=4, fill=(16, 185, 129))

        frames.append(img)

    print(f"[Video Generator] Compiling animated video to {OUTPUT_VIDEO}...")
    frames[0].save(
        OUTPUT_VIDEO,
        save_all=True,
        append_images=frames[1:],
        duration=120,  # 120ms per frame (~8.3 fps, smooth and readable)
        loop=0,
    )
    print(f"[Video Generator] Successfully created {OUTPUT_VIDEO} ({OUTPUT_VIDEO.stat().st_size / 1024:.1f} KB)!")

if __name__ == "__main__":
    main()
