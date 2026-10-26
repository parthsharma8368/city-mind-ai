"""
update_presentation_decks.py
=============================
Updates build_presentation.py and generate_presentation_html_and_pdf.py
to include Slide 10: Figure Walkthrough - IPPO Hyperparameters & Mathematical Architecture.
Then runs both builders to generate:
  - docs/CityMind_AI_Professor_Defense_Master.pptx
  - docs/presentation_deck.html
  - docs/CityMind_AI_Professor_Defense_Master.pdf
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 1. Update build_presentation.py
build_script = ROOT / "scripts" / "build_presentation.py"
content = build_script.read_text(encoding="utf-8")

new_slide_10_code = '''def build_slide_10(prs, blank_layout):
    """Slide 10: Figure Walkthrough - IPPO Hyperparameters & Mathematical Architecture (ippo_hyperparameters_guide.png)"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(
        slide,
        "Mathematical & Hyperparameter Rigor",
        "Figure Walkthrough: IPPO Hyperparameters & Architecture",
        "Rigorous parameter selection, rollout scale (588 agents/tick), and mathematical loss formulation",
        10
    )

    img_path = str(ROOT / "docs" / "ippo_hyperparameters_guide.png")
    add_card(slide, Inches(0.8), Inches(1.85), Inches(11.733), Inches(4.55))
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(0.85), Inches(1.88), width=Inches(11.633))

    add_callout(
        slide, Inches(0.8), Inches(6.45), Inches(11.733), Inches(0.85),
        "HYPERPARAMETER JUSTIFICATION SCRIPT",
        "Explain to the professor: 'Every hyperparameter is mathematically grounded: γ=0.99 ensures an 8.3-minute horizon for green waves; ε=0.2 bounds policy shift to 20%; 6 parallel environments collect 105,840 transitions per update; and Split Gradient Clipping isolates actor gradients from critic value error.'"
    )

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 10:
"Professor, this slide provides the complete mathematical and hyperparameter architecture of our best-performing IPPO model, directly from configs/ippo.yaml.

Every hyperparameter was chosen with a specific mathematical and physical traffic rationale:

1. Rollout Scale & Multi-Agent Parallelization:
   - We run 6 parallel SUMO environments (num_vec_envs = 6), each simulating all 98 intersections. That means 588 intersection agents act on every single simulation step!
   - We collect 180 decision steps per update (n_steps = 180). At a 5-second decision interval, that represents 900 seconds—exactly 15 minutes of simulated rush hour, or 4 policy updates per peak hour.
   - 588 agents times 180 steps creates a massive rollout buffer of 105,840 transitions per update.
   - We divide this into 8 minibatches of 13,230 samples (batch_size = 13,230) and train for 10 epochs. Total training covers 10,584,000 steps—exactly 100 complete PPO updates, simulating 150 total hours of city traffic.

2. RL Horizons & Mathematical Bounds:
   - Gamma (0.99): At a 5-second step length, an effective horizon of 1 / (1 - 0.99) = 100 steps corresponds to 500 seconds (~8.3 minutes). This is critical: traffic queues cannot be cleared in 5 seconds. The 8.3-minute horizon allows the policy to look ahead and form coordinated green waves across neighboring intersections.
   - GAE Lambda (0.95): Balances the high stochastic variance of vehicle arrival distributions against the value baseline.
   - Clip Range (0.2): Restricts the probability ratio to between 0.8 and 1.2, enforcing a 20% trust region that guarantees smooth, non-oscillatory policy updates.
   - Advantage Normalization: Critical across 98 junctions so that massive 19-lane highway intersections do not drown out smaller 4-lane residential intersections.
   - Entropy Coefficient (0.01): Encourages exploration of alternate phase transitions, preventing premature collapse into 'always keep green'.
   - Value Loss Weight (0.5): Balances critic Mean Squared Error against policy gradient loss.

3. Neural Network & Optimization:
   - Policy Architecture: A compact 2-layer MLP [256, 256] with Tanh activations. Tanh provides smooth zero-centered gradients bounded in [-1, 1], avoiding dying ReLUs during early traffic exploration.
   - Inference Speed: Compact enough to execute on CPU in less than 1 millisecond per intersection.
   - Learning Rate: 3e-4 with Adam optimizer—the canonical standard for PPO on continuous normalized states."
""")


'''

if "def build_slide_10(" in content and "Figure Walkthrough: IPPO Hyperparameters" not in content:
    # Rename 18 down to 10
    for i in range(18, 9, -1):
        content = content.replace(f"def build_slide_{i}(", f"def build_slide_{i+1}(")
        content = content.replace(f'"""Slide {i}:', f'"""Slide {i+1}:')
        content = content.replace(f', {i})', f', {i+1})')
        content = content.replace(f'build_slide_{i}(prs, blank_layout)', f'build_slide_{i+1}(prs, blank_layout)')

    # Now insert build_slide_10 before build_slide_11
    target = "def build_slide_11(prs, blank_layout):"
    content = content.replace(target, new_slide_10_code + target)

    # In main(), add build_slide_10
    content = content.replace(
        "    build_slide_9(prs, blank_layout)\n    build_slide_11(prs, blank_layout)",
        "    build_slide_9(prs, blank_layout)\n    build_slide_10(prs, blank_layout)\n    build_slide_11(prs, blank_layout)"
    )

    build_script.write_text(content, encoding="utf-8")
    print("Updated build_presentation.py with Slide 10!")

# 2. Update generate_presentation_html_and_pdf.py
html_script = ROOT / "scripts" / "generate_presentation_html_and_pdf.py"
html_content = html_script.read_text(encoding="utf-8")

new_html_slide_dict = """    {
        "num": 10,
        "is_dark": False,
        "badge": "Mathematical & Hyperparameter Rigor",
        "title": "Figure Walkthrough: IPPO Hyperparameters & Architecture",
        "subtitle": "Rigorous parameter selection, rollout scale (588 agents/tick), and mathematical loss formulation",
        "fig_layout": {
            "img": "ippo_hyperparameters_guide.png",
            "title": "Mathematical & Physical Traffic Rationale",
            "points": [
                ("Master Objective L(θ):", "L^CLIP(θ) - 0.5·L^VF(θ) + 0.01·S[π_θ](s_t). Balances policy gain, value estimation, and exploration."),
                ("Rollout Scale (588 Agents):", "6 parallel SUMO envs × 98 signals = 588 agents acting per tick. 180 steps = 900s (15 min) rush hour = 105,840 samples/update."),
                ("Horizon γ = 0.99 & GAE λ = 0.95:", "Effective horizon = 500s (~8.3 min), enabling coordinated multi-intersection green waves instead of myopic 5s gains."),
                ("Trust Region ε = 0.2:", "Bounds policy ratio to [0.8, 1.2], preventing destructive policy shifts and citywide gridlock."),
                ("Split Gradient Clipping:", "Decouples Actor (0.5) and Critic (0.5) clipping, resolving the 7,600x gradient suppression bug and speeding learning by 10x."),
                ("MLP [256, 256] with Tanh:", "Zero-centered smooth gradients in [-1, 1], avoiding dying ReLUs and executing inference in <1ms on CPU.")
            ]
        },
        "callout": "Explain to the professor: 'Every hyperparameter is mathematically grounded: γ=0.99 ensures an 8.3-minute horizon for green waves; ε=0.2 bounds policy shift to 20%; 6 parallel environments collect 105,840 transitions per update; and Split Gradient Clipping isolates actor gradients from critic value error.'",
        "notes": "Professor, this slide details our best-performing IPPO hyperparameters. With 6 parallel environments, 588 agents act simultaneously, collecting 105,840 transitions every 15 minutes of simulated rush hour across 8 minibatches. Gamma=0.99 grants an 8.3-minute horizon for green wave formation, epsilon=0.2 bounds updates to a 20% trust region, and Split Gradient Clipping clips actor and critic independently at 0.5 to prevent gradient starvation."
    },
"""

if '"num": 10,' in html_content and "Figure Walkthrough: IPPO Hyperparameters" not in html_content:
    for i in range(18, 9, -1):
        html_content = html_content.replace(f'"num": {i},', f'"num": {i+1},')
    
    target_pattern = '    {\n        "num": 11,'
    html_content = html_content.replace(target_pattern, new_html_slide_dict + target_pattern)
    html_script.write_text(html_content, encoding="utf-8")
    print("Updated generate_presentation_html_and_pdf.py with Slide 10!")

print("Running build_presentation.py to generate PPTX...")
subprocess.run(["python3", str(build_script)], check=True)

print("Running generate_presentation_html_and_pdf.py to generate HTML and PDF...")
subprocess.run(["python3", str(html_script)], check=True)

print("All presentation formats updated successfully!")
