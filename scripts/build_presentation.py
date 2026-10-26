"""
Script to build the Master Professor Defense PowerPoint Presentation for CityMind AI.
Outputs: docs/CityMind_AI_Professor_Defense_Master.pptx
"""
import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Root Directory
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "docs" / "CityMind_AI_Professor_Defense_Master.pptx"

# Dimensions: 16:9 widescreen (13.333 x 7.5 inches)
SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)

# Color Palette: Clean Executive Dark / Light Academic Palette
NAVY_DARK = RGBColor(11, 25, 44)       # #0B192C (Title / Dark Slide Background)
SLATE_DARK = RGBColor(15, 23, 42)      # #0F172A
BG_LIGHT = RGBColor(248, 250, 252)     # #F8FAFC (Standard Canvas)
CARD_BG = RGBColor(255, 255, 255)      # Pure White Card
CARD_BORDER = RGBColor(226, 232, 240)  # #E2E8F0 Subtle Border
BLUE_ACCENT = RGBColor(42, 120, 214)   # #2A78D6 Brand Electric Blue
CORAL_ACCENT = RGBColor(235, 104, 52)  # #EB6834 Warm Amber/Coral (Model B)
GREEN_ACCENT = RGBColor(16, 185, 129)  # #10B981 Success Green
TEXT_MAIN = RGBColor(15, 23, 42)       # Charcoal text
TEXT_MUTED = RGBColor(100, 116, 139)   # Slate 500
TEXT_LIGHT = RGBColor(241, 245, 249)   # Slate 100 for dark slides
BADGE_BG_BLUE = RGBColor(239, 246, 255)
BADGE_TXT_BLUE = RGBColor(29, 78, 216)
CALLOUT_BG = RGBColor(240, 253, 244)   # Soft green tint
CALLOUT_BORDER = RGBColor(187, 247, 208)
CALLOUT_TXT = RGBColor(22, 101, 52)
AMBER_BG = RGBColor(254, 243, 199)
AMBER_TXT = RGBColor(146, 64, 15)

FONT_HEADING = "Helvetica"
FONT_BODY = "Arial"


def create_presentation():
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT
    blank_layout = prs.slide_layouts[6]
    return prs, blank_layout


def add_canvas(slide, is_dark=False):
    """Draws background rectangle covering entire slide."""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
    shape.fill.solid()
    shape.fill.fore_color.rgb = NAVY_DARK if is_dark else BG_LIGHT
    shape.line.color.rgb = NAVY_DARK if is_dark else BG_LIGHT
    return shape


def add_header(slide, badge_text, title_text, subtitle_text, slide_num=None):
    """Adds a standard executive header to a light slide."""
    # Badge
    badge_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.45), Inches(3.2), Inches(0.32))
    badge_box.fill.solid()
    badge_box.fill.fore_color.rgb = BADGE_BG_BLUE
    badge_box.line.color.rgb = BADGE_BG_BLUE
    tf_b = badge_box.text_frame
    tf_b.margin_top = tf_b.margin_bottom = tf_b.margin_left = tf_b.margin_right = 0
    tf_b.word_wrap = True
    p_b = tf_b.paragraphs[0]
    p_b.alignment = PP_ALIGN.CENTER
    p_b.text = badge_text.upper()
    p_b.font.name = FONT_HEADING
    p_b.font.size = Pt(9.5)
    p_b.font.bold = True
    p_b.font.color.rgb = BADGE_TXT_BLUE

    # Slide number
    if slide_num:
        num_box = slide.shapes.add_textbox(Inches(11.5), Inches(0.4), Inches(1.0), Inches(0.35))
        tf_n = num_box.text_frame
        p_n = tf_n.paragraphs[0]
        p_n.alignment = PP_ALIGN.RIGHT
        p_n.text = f"{slide_num:02d} / 18"
        p_n.font.name = FONT_BODY
        p_n.font.size = Pt(10)
        p_n.font.color.rgb = TEXT_MUTED

    # Title & Subtitle in one text frame for perfect flow
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11.733), Inches(0.95))
    tf = title_box.text_frame
    tf.word_wrap = True
    tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0

    p_t = tf.paragraphs[0]
    p_t.text = title_text
    p_t.font.name = FONT_HEADING
    p_t.font.size = Pt(21)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_MAIN

    p_s = tf.add_paragraph()
    p_s.text = subtitle_text
    p_s.font.name = FONT_BODY
    p_s.font.size = Pt(11.5)
    p_s.font.color.rgb = TEXT_MUTED


def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
    """Draws a clean white or tinted card with subtle border."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.2)
    return card


def add_callout(slide, left, top, width, height, label, text):
    """Draws an intuitive 'How to explain to the Professor in 20 seconds' callout banner."""
    box = add_card(slide, left, top, width, height, CALLOUT_BG, CALLOUT_BORDER)
    tb = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.08), width - Inches(0.3), height - Inches(0.16))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0

    p1 = tf.paragraphs[0]
    p1.text = f"💡 HOW TO EXPLAIN TO THE PROFESSOR IN 20 SECONDS ({label}):"
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(9.5)
    p1.font.bold = True
    p1.font.color.rgb = CALLOUT_TXT

    p2 = tf.add_paragraph()
    p2.text = text
    p2.font.name = FONT_BODY
    p2.font.size = Pt(10)
    p2.font.color.rgb = TEXT_MAIN


def set_speaker_notes(slide, notes_text):
    """Sets comprehensive speaker notes for the slide."""
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = notes_text.strip()


# ==============================================================================
# SLIDE BUILDERS
# ==============================================================================

def build_slide_1(prs, blank_layout):
    """Slide 1: Title Slide (Dark Executive Theme)"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide, is_dark=True)

    # Accent decorative bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.2), Inches(1.2), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE_ACCENT
    bar.line.color.rgb = BLUE_ACCENT

    # Category Pill
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.45), Inches(3.8), Inches(0.38))
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(30, 41, 59)
    pill.line.color.rgb = BLUE_ACCENT
    tf_p = pill.text_frame
    p_p = tf_p.paragraphs[0]
    p_p.alignment = PP_ALIGN.CENTER
    p_p.text = "FINAL RL PROJECT REVIEW & DEFENSE"
    p_p.font.name = FONT_HEADING
    p_p.font.size = Pt(10.5)
    p_p.font.bold = True
    p_p.font.color.rgb = RGBColor(147, 197, 253)

    # Main Title
    t_box = slide.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(11.3), Inches(2.2))
    tf = t_box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "CityMind AI"
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(255, 255, 255)

    p2 = tf.add_paragraph()
    p2.text = "Decentralized Deep Reinforcement Learning for City-Wide Traffic Signal Control"
    p2.font.name = FONT_HEADING
    p2.font.size = Pt(22)
    p2.font.color.rgb = RGBColor(203, 213, 225)

    p3 = tf.add_paragraph()
    p3.text = "Beating Calibrated Civil-Engineering Actuation on 98 Real-World Intersections in Ingolstadt, Germany"
    p3.font.name = FONT_BODY
    p3.font.size = Pt(14)
    p3.font.color.rgb = BLUE_ACCENT

    # 4 Key Value Stat Cards
    metrics = [
        ("-41.4 s", "Lag per scheduled car", "10/10 seeds won (p=0.002)"),
        ("-41.1%", "Mean waiting time", "Dropped from 6.39s to 3.77s"),
        ("-256 Cars", "City backlog at 08:00", "Zero gridlock spillback"),
        ("10 / 10", "Statistical victory", "Paired seed-by-seed tests")
    ]
    card_w = Inches(2.65)
    card_h = Inches(1.35)
    for i, (val, label, sub) in enumerate(metrics):
        cx = Inches(1.0 + i * 2.88)
        cy = Inches(4.7)
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, cy, card_w, card_h)
        card.fill.solid()
        card.fill.fore_color.rgb = RGBColor(24, 34, 53)
        card.line.color.rgb = RGBColor(51, 65, 85)
        tf_c = card.text_frame
        tf_c.word_wrap = True
        tf_c.margin_top = Inches(0.12)
        
        p_val = tf_c.paragraphs[0]
        p_val.text = val
        p_val.font.name = FONT_HEADING
        p_val.font.size = Pt(20)
        p_val.font.bold = True
        p_val.font.color.rgb = GREEN_ACCENT if "%" in val or "won" in sub else BLUE_ACCENT

        p_lbl = tf_c.add_paragraph()
        p_lbl.text = label
        p_lbl.font.name = FONT_BODY
        p_lbl.font.size = Pt(10.5)
        p_lbl.font.bold = True
        p_lbl.font.color.rgb = RGBColor(241, 245, 249)

        p_sub = tf_c.add_paragraph()
        p_sub.text = sub
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(8.5)
        p_sub.font.color.rgb = RGBColor(148, 163, 184)

    # Footer Metadata
    f_box = slide.shapes.add_textbox(Inches(1.0), Inches(6.5), Inches(11.3), Inches(0.5))
    tf_f = f_box.text_frame
    p_f = tf_f.paragraphs[0]
    p_f.text = "Environment: SUMO Microscopic Physics (TraCI / libsumo) | Algorithm: IPPO + Split Gradient Clipping | Baseline: SUMO Actuated Induction Loops"
    p_f.font.name = FONT_BODY
    p_f.font.size = Pt(10)
    p_f.font.color.rgb = RGBColor(148, 163, 184)

    set_speaker_notes(slide, """
5-MINUTE OPENING DEFENSE HOOK (Deliver with confidence):
"Good morning, Professor. Today I am presenting CityMind AI, our final project on Decentralized Deep Reinforcement Learning for city-wide traffic signal control.

Most academic RL traffic papers test on synthetic, 4-way toy grids with uncalibrated fixed-time timers and claim massive improvements. We asked a much tougher question: Can Deep RL outperform real-world civil engineering on a genuine city scale?

We tested on the Ingolstadt InTAS network in Germany—98 heterogeneous intersections during the morning rush hour with over 15,000 scheduled trips. Our benchmark is not an artificial baseline, but SUMO's calibrated vehicle-actuated controller running dynamic induction loops.

As shown on this title slide, our final model beats the civil engineering baseline on 10 out of 10 independent random seeds, slashing waiting time by 41.1% and saving 41.4 seconds of lag per vehicle across the entire city. Today I will walk you through the theory from Sudharsan Ravichandiran's textbook, our architectural journey, the exact engineering breakthroughs that unlocked this performance, and the full statistical proof."
""")


def build_slide_2(prs, blank_layout):
    """Slide 2: Executive Summary & Motivation"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Problem Statement & Motivation", "The Reality Check: Escaping Academic Toy Grids", "Why typical RL traffic papers fail in reality, and how CityMind evaluates real urban scale", 2)

    # 3 Comparison Columns
    col_w = Inches(3.64)
    col_h = Inches(3.9)
    top_pos = Inches(1.9)

    # Card 1: Toy Grids
    add_card(slide, Inches(0.8), top_pos, col_w, col_h)
    tb1 = slide.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.2), col_w - Inches(0.4), col_h - Inches(0.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "1. Academic Toy Grids ❌"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CORAL_ACCENT

    bullets1 = [
        "Synthetic 4-way intersections with identical 2-lane approaches.",
        "Uncalibrated fixed-time timers as the comparison baseline (trivial to beat).",
        "Assumes infinite queue capacity; ignores turning-bay spillback and lane blocking.",
        "Reports 'Mean Wait Time of Completed Cars', disguising massive gridlock on side streets (Survivor Bias).",
        "Fails completely when deployed on real urban topology."
    ]
    for b in bullets1:
        pb = tf1.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(10)
        pb.font.color.rgb = TEXT_MAIN

    # Card 2: Ingolstadt InTAS
    add_card(slide, Inches(4.84), top_pos, col_w, col_h)
    tb2 = slide.shapes.add_textbox(Inches(5.04), top_pos + Inches(0.2), col_w - Inches(0.4), col_h - Inches(0.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "2. Real Ingolstadt (InTAS) 🇩🇪"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    bullets2 = [
        "Genuine European city network: 98 signalized intersections in Ingolstadt, Germany.",
        "Severe structural heterogeneity: Junctions have between 4 and 19 lanes, varying green phases (2 to 6).",
        "Morning rush hour scenario (07:00–08:00) with over 15,000 scheduled trips.",
        "Realistic driver behavior: Krauss car-following, LC2013 lane change dynamics, clearance delays.",
        "Real-world bottlenecks: Stop lines holding only 2 cars with 60-car feeder jams."
    ]
    for b in bullets2:
        pb = tf2.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(10)
        pb.font.color.rgb = TEXT_MAIN

    # Card 3: The Benchmark
    add_card(slide, Inches(8.88), top_pos, col_w, col_h)
    tb3 = slide.shapes.add_textbox(Inches(9.08), top_pos + Inches(0.2), col_w - Inches(0.4), col_h - Inches(0.4))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "3. The True Civil Baseline 🎯"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    bullets3 = [
        "SUMO Calibrated Vehicle-Actuated Controller (the real-world industry standard).",
        "Uses embedded induction loop detectors that sense gap-out headway in real time.",
        "Extends green phases dynamically when platoons arrive and terminates when gaps appear.",
        "Governed by authentic civil-engineering minimum, maximum, and yellow clearance intervals.",
        "Our Goal: Prove Deep RL can outperform calibrated civil engineering across the full network."
    ]
    for b in bullets3:
        pb = tf3.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(10)
        pb.font.color.rgb = TEXT_MAIN

    # Bottom Callout Box
    add_callout(slide, Inches(0.8), Inches(6.0), Inches(11.733), Inches(1.05), "MOTIVATION",
                "Explain to the professor: 'We refused to test on toy 4-way synthetic grids. We took the full Ingolstadt city network with 98 real intersections and matched our RL agent against SUMO's calibrated induction-loop actuated controller. If our model wins here, it proves RL is ready for real cities.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 2:
"Professor, the first thing I want to highlight is our motivation. If you look at standard RL traffic literature, 90% of papers evaluate on synthetic 4-way grids with pre-timed fixed timers. They claim 20% to 30% improvements, but when deployed on a real city, those policies collapse immediately.

Why? Because real cities don't look like grids. In Ingolstadt, Germany—the InTAS scenario we used—intersections have anywhere from 4 to 19 lanes, varying green phase structures, and short 20-meter stop lines where traffic backs up for hundreds of meters.

Furthermore, we didn't compare against an uncalibrated, dumb fixed-time baseline. We benchmarked against SUMO's calibrated vehicle-actuated controller. That controller uses real-time induction loops with gap-out logic—the exact system civil engineers use in modern cities today. Our objective was to prove whether multi-agent Deep RL can fundamentally beat calibrated civil engineering across an entire city network."
""")


def build_slide_3(prs, blank_layout):
    """Slide 3: Real-World InTAS & Microscopic Simulation Physics"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Simulation & Physical Modeling", "Microscopic Vehicle Physics in Eclipse SUMO", "Simulating continuous-space driver dynamics, car-following, and urban heterogeneity", 3)

    # Left Column: Microscopic Physics
    add_card(slide, Inches(0.8), Inches(1.9), Inches(5.6), Inches(4.0))
    tb_l = slide.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(3.6))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "Microscopic Simulation Engine: Eclipse SUMO"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    bullets_l = [
        "Continuous Space & Discrete Time: Tracks every single vehicle individually at 1.0-second physics increments.",
        "Krauss Car-Following Model: Models realistic acceleration, safe following distance, deceleration limits, and driver reaction latencies.",
        "LC2013 Lane-Changing Model: Simulates tactical overtaking, lane blocking, merge conflicts, and turning-lane queues.",
        "Intersection Clearance Physics: Yellow change intervals (3–4s) and all-red red-clearance intervals are explicitly simulated to prevent phantom teleportations.",
        "Spillback Dynamics: Traffic backing up past intersection stop-lines actively blocks upstream cross-traffic, capturing cascading urban gridlock."
    ]
    for b in bullets_l:
        pb = tf_l.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    # Right Column: Network Heterogeneity
    add_card(slide, Inches(6.8), Inches(1.9), Inches(5.733), Inches(4.0))
    tb_r = slide.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.333), Inches(3.6))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "Ingolstadt InTAS Network Specifications"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    bullets_r = [
        "98 Controlled Intersections: Spans high-speed arterial corridors, highway entry ramps, and dense downtown streets.",
        "Heterogeneous Junction Geometry: Signals feature between 4 and 19 controlled incoming lanes and 2 to 6 unique green phase configurations.",
        "Morning Rush Hour Window: 07:00 to 08:00 (3,600 simulated seconds), representing peak commuting demand.",
        ">15,000 Scheduled Vehicle Trips: Realistic origin-destination trip demand calibrated from German municipal mobility surveys.",
        "Decision Step: The RL agent acts every 5 seconds (720 discrete decision steps per hour), allowing vehicle platoons to clear without jitter.",
        "Parallel Training Throughput: 6 parallel SUMO environments (588 simultaneous signal transitions per step) for high sample efficiency."
    ]
    for b in bullets_r:
        pb = tf_r.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "SIMULATION PHYSICS",
                "Explain to the professor: 'SUMO doesn't treat traffic as abstract fluid flow. Every car has an engine, brakes, driver latency, and lane-changing logic. If an intersection turns green, cars don't instantly move—they accelerate one by one according to Krauss physics. That makes signal coordination genuinely challenging.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 3:
"Professor, to model traffic accurately, we selected Eclipse SUMO (Simulation of Urban MObility). 

Why SUMO? Because SUMO is a microscopic simulator. It does not treat traffic as an aggregate fluid. It tracks every vehicle individually. It uses the Krauss car-following model for realistic acceleration, headway spacing, and brake delays, along with the LC2013 lane-changing model to simulate merge conflicts. When a queue backs up, it creates spillback that physically blocks upstream intersections, exactly like real life.

The network is the Ingolstadt InTAS scenario in Germany: 98 signalized intersections, 15,000 scheduled trips during the 07:00 to 08:00 morning rush hour. Our decision step is 5 seconds—meaning the agent makes 720 decisions per hour per signal. This 5-second interval gives vehicles enough time to cross stop-lines without inducing light-flickering instability."
""")


def build_slide_4(prs, blank_layout):
    """Slide 4: TraCI Middleware & Codebase Connection"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Middleware & Integration", "TraCI: The Bi-directional TCP Socket Protocol", "How our Python RL environment queries sensors and actuates signal phases in SUMO", 4)

    col_w = Inches(3.64)
    col_h = Inches(3.95)
    top_pos = Inches(1.9)

    # Card 1: What is TraCI?
    add_card(slide, Inches(0.8), top_pos, col_w, col_h)
    tb1 = slide.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "1. What is TraCI? 🔌"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    bullets1 = [
        "TraCI = Traffic Control Interface.",
        "Client-Server Architecture: SUMO runs as a socket server (listening on a TCP port); our Python environment connects as a client.",
        "Bidirectional Interaction: Retrieves simulation state (sensors) and commands simulation objects (actuators).",
        "Deterministic Stepping: The simulation clock is frozen until Python sends traci.simulationStep().",
        "Speed Optimization: In our codebase (src/environment/sumo_connection.py), we use libsumo—a direct C++ SWIG binding eliminating TCP socket serialization overhead during training."
    ]
    for b in bullets1:
        pb = tf1.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    # Card 2: Sensor Ingestion
    add_card(slide, Inches(4.84), top_pos, col_w, col_h)
    tb2 = slide.shapes.add_textbox(Inches(5.04), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "2. Sensor Ingestion (Every 5s) 📡"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    bullets2 = [
        "Queue Measurements: Calls traci.lane.getLastStepHaltingNumber(lane) to detect stopped vehicles (<0.1 m/s).",
        "Upstream Feeder Walking: Recursively crawls lane topology via traci.lane.getLinks() to inspect traffic up to 300 meters behind stop-lines.",
        "Downstream Speeds: Queries traci.lane.getLastStepMeanSpeed(lane) to monitor exit road congestion and prevent spillback.",
        "Outside Network Demand: Queries traci.simulation.getPendingVehicles() to count vehicles queued at city boundary entry gates waiting to spawn."
    ]
    for b in bullets2:
        pb = tf2.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    # Card 3: Phase Actuation
    add_card(slide, Inches(8.88), top_pos, col_w, col_h)
    tb3 = slide.shapes.add_textbox(Inches(9.08), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "3. Phase Actuation & Safety 🚦"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CORAL_ACCENT

    bullets3 = [
        "RL Decision: Agent chooses action a in {0: Keep, 1: Advance}.",
        "Deterministic Safety Filter: Actions pass through SafetyLayer (safety.py) before reaching SUMO.",
        "Guarantees Min/Max Green: Prevents rapid oscillation and starvation.",
        "Executes Clearance: Automatically schedules yellow and all-red clearance intervals.",
        "Direct Actuation Command: Sends traci.trafficlight.setPhase(tls_id, phase_index) to command the hardware.",
        "Advances Clock: Calls traci.simulationStep() to progress physics by 1 second."
    ]
    for b in bullets3:
        pb = tf3.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "TRACI CODEBASE HOOKS",
                "Explain to the professor: 'TraCI is our TCP socket middleware. In src/environment/sumo_connection.py, we connect Python to SUMO. Every 5 seconds, Python queries vehicle queues and speeds, runs our RL policy, checks the safety layer, and commands traci.trafficlight.setPhase(). It is 100% automated and deterministic.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 4:
"Professor, one of the most critical engineering components is TraCI—the Traffic Control Interface.

What is TraCI? TraCI is a TCP socket protocol that enables bidirectional interaction between SUMO and Python. SUMO acts as the socket server listening on a TCP port, and our Python environment acts as the client. In production and training, we also support libsumo, which uses direct C++ SWIG bindings to bypass network sockets for maximum speed.

In our codebase, TraCI performs two core jobs every 5 seconds:
First, Sensor Ingestion: We query halted cars using traci.lane.getLastStepHaltingNumber, check downstream lane speeds, and inspect pending vehicles queued at city entry gates.
Second, Actuation: When the RL policy picks an action (Keep or Advance), the action is filtered through our deterministic SafetyLayer, which then calls traci.trafficlight.setPhase() to command the junction signal, followed by traci.simulationStep() to advance time. TraCI is the bridge between our AI policy and the physical simulator."
""")


def build_slide_5(prs, blank_layout):
    """Slide 5: Graph Walkthrough - The Agent-Environment Loop (fig_loop.png)"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "System Architecture Figure", "Figure 1 Walkthrough: The Closed-Loop Control System", "Explaining the 720-cycle agent-environment interaction loop and the safety layer", 5)

    # Left: Embedded Image
    img_path = str(ROOT / "docs" / "professor_report" / "fig" / "fig_loop.png")
    add_card(slide, Inches(0.8), Inches(1.9), Inches(6.8), Inches(4.0))
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(0.95), Inches(2.1), width=Inches(6.5))

    # Right: Step-by-Step Breakdown Card
    add_card(slide, Inches(7.8), Inches(1.9), Inches(4.733), Inches(4.0))
    tb_r = slide.shapes.add_textbox(Inches(8.0), Inches(2.05), Inches(4.333), Inches(3.7))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "Deconstructing Figure 1"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    points = [
        ("Left Box (The Agent):", "ONE shared neural network for all 98 signals. It takes a 22-number state vector and outputs an action."),
        ("Right Box (The Environment):", "SUMO simulating 98 signals, ~15,000 cars, and microscopic road physics across Ingolstadt."),
        ("Top Arrow (Action):", "Sent every 5 seconds: 0 = Keep Green, 1 = Advance to Next Phase."),
        ("Middle Arrow (State):", "22 normalized features: current phase, elapsed green time, queue lengths, 300m feeder jams, and neighbor traffic."),
        ("Bottom Arrow (Reward):", "A direct cost penalty proportional to stopped queues, driver delays, and city boundary backlogs."),
        ("The Bottom Banner (Safety Layer):", "A deterministic filter ensuring minimum green, maximum green, and yellow clearance are never violated.")
    ]
    for title, desc in points:
        pb = tf_r.add_paragraph()
        pb.text = f"{title} {desc}"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "FIGURE 1 WALKTHROUGH",
                "Explain to the professor: 'Point directly to Figure 1: On the left is our single shared neural network. On the right is SUMO. Every 5 seconds, the agent receives 22 numbers describing traffic, picks an action, and the safety layer ensures yellow and red clearance are never skipped. This loop runs 720 times per hour.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 5:
"Professor, Figure 1 on the screen illustrates our closed-loop control architecture, which repeats 720 times every simulated hour.

Let me break down the four components:
1. The Left Box is our RL Agent: Notice that it is ONE single neural network shared across all 98 traffic signals.
2. The Right Box is the Environment: Eclipse SUMO executing the microscopic physics of 15,000 vehicles across Ingolstadt.
3. The Three Arrows represent the communication cycle:
   - State (Middle Arrow): At every 5-second tick, SUMO feeds 22 normalized numbers per intersection into the policy.
   - Action (Top Arrow): The policy outputs a discrete decision: 0 (keep green) or 1 (advance to next green phase).
   - Reward (Bottom Arrow): SUMO returns the delay cost based on stopped vehicles and queue lengths.
4. Finally, look at the note at the bottom: The Deterministic Safety Layer. The RL agent is never allowed to directly switch lights recklessly. The safety layer enforces minimum green times (so lights don't flicker), maximum green caps (so side streets don't starve), and yellow/all-red clearance intervals so vehicles never collide."
""")


def build_slide_6(prs, blank_layout):
    """Slide 6: Architectural Evolution - 3 Approaches Tried (Handwritten Notes)"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "System Architecture Evolution", "Architectural Exploration: 3 Approaches Tested", "Directly reflecting our handwritten engineering journey from Centralized to IPPO", 6)

    col_w = Inches(3.64)
    col_h = Inches(3.95)
    top_pos = Inches(1.9)

    # Card 1: Centralized
    add_card(slide, Inches(0.8), top_pos, col_w, col_h)
    tb1 = slide.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "Approach 1: Centralized Agent ❌"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CORAL_ACCENT

    bullets1 = [
        "Concept: One single central controller observes the entire city state and selects joint actions for all 98 intersections.",
        "Observation Space: Combined state vector of thousands of raw lane queues.",
        "Action Space Explosion: With 98 binary signals, joint discrete action space is 2^98 ≈ 3.16 × 10^29 actions per step!",
        "Why It Failed: Mathematically intractable. Neural networks cannot sample or converge across 10^29 joint actions. Severe sample inefficiency."
    ]
    for b in bullets1:
        pb = tf1.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    # Card 2: Decentralized Independent
    add_card(slide, Inches(4.84), top_pos, col_w, col_h)
    tb2 = slide.shapes.add_textbox(Inches(5.04), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Approach 2: 98 Separate Agents ❌"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = AMBER_TXT

    bullets2 = [
        "Concept: Every single intersection gets its own dedicated, specialized neural network policy.",
        "Theoretical Appeal: Each policy network can specialize in the unique geometry of its intersection.",
        "Massive Overhead: Training, checkpointing, and optimizing 98 distinct neural networks simultaneously.",
        "Why It Failed: Environment non-stationarity. Each agent changed the traffic environment for its neighbors, destabilizing learning. Zero knowledge transfer between junctions."
    ]
    for b in bullets2:
        pb = tf2.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    # Card 3: IPPO with Parameter Sharing
    add_card(slide, Inches(8.88), top_pos, col_w, col_h)
    tb3 = slide.shapes.add_textbox(Inches(9.08), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "Approach 3: IPPO + Shared Network ✅"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    bullets3 = [
        "Concept: Independent PPO (IPPO) with Parameter Sharing + Neighbor Observation Sharing.",
        "One Shared Brain: All 98 intersections share one identical Actor-Critic policy (22 -> 256 -> 256 -> 2).",
        "Local Observations: Each signal inputs its own 22-dim local view (including queues from immediate neighbor junctions).",
        "Why It Won: Massive sample efficiency! Every 5s step produces 98 experience transitions. Knowledge learned on a major avenue immediately generalizes across the city."
    ]
    for b in bullets3:
        pb = tf3.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "ARCHITECTURAL LESSON",
                "Explain to the professor: 'As written in our review notes, we tested all three architectures: A centralized agent suffers from 2^98 action space explosion; 98 separate networks suffer from non-stationarity and compute bloat; but IPPO with parameter sharing gives us 98x sample efficiency and spatial generalization.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 6:
"Professor, in our project review notes, we documented our systematic investigation into three distinct multi-agent architectures:

First, we explored a Centralized Agent. One massive network observing all 98 intersections. Why did that fail? Because of the curse of dimensionality: with 98 signals, the joint action space is 2^98—over 10^29 possible combinations every 5 seconds. Finding the optimal action is mathematically intractable.

Second, we evaluated Decentralized Independent Agents—training 98 separate neural networks, one for each intersection. While theoretically capable of specializing, this created two fatal problems: extreme compute overhead, and severe non-stationarity. When Agent 12 changed its policy, it altered vehicle arrival rates for Agent 13, causing policy oscillations and divergence.

Third, our winning architecture: IPPO with Parameter Sharing. All 98 intersections share ONE single neural network policy, but each feeds its own local 22-dimensional observation vector (which includes queue information from adjacent neighbors). This gave us 98 transitions per simulation step, allowing what the policy learns at a congested intersection to instantly improve control at other intersections. This achieved the highest stability and best citywide results."
""")


def build_slide_7(prs, blank_layout):
    """Slide 7: Why PPO? Algorithm Selection & Sudharsan Ravichandiran Foundations"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Theoretical Foundations", "Why PPO? Grounded in Sudharsan Ravichandiran (2018)", "Comparing classical RL algorithms against PPO across stability, variance, and sample efficiency", 7)

    # Table comparing algorithms
    rows = 6
    cols = 4
    left = Inches(0.8)
    top = Inches(1.9)
    width = Inches(11.733)
    height = Inches(3.95)

    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    table.columns[0].width = Inches(2.2)
    table.columns[1].width = Inches(1.8)
    table.columns[2].width = Inches(4.2)
    table.columns[3].width = Inches(3.533)

    headers = ["Algorithm", "Book Reference", "Why It Fails on 98-Signal City Control", "Why PPO / IPPO Wins"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = SLATE_DARK
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_HEADING
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)

    data = [
        ("Q-Learning / Tabular TD", "Ch 3 & 5 (pp. 53–120)", "State space is continuous & infinite (22 numbers per signal). Tabular Q cannot generalize.", "PPO uses deep neural network function approximation mapping continuous states to actions."),
        ("Deep Q-Networks (DQN)", "Ch 8 (pp. 165–188)", "Joint action space is 2^98. In multi-agent, off-policy non-stationarity invalidates replay buffer.", "PPO is on-policy; handles multi-agent decentralized transitions naturally without replay corruption."),
        ("REINFORCE (Vanilla PG)", "Ch 11 (pp. 226–230)", "Relies on full episode Monte Carlo returns. Massive variance, slow learning, no baseline.", "PPO uses Actor-Critic with GAE (gamma=0.99, lambda=0.95), drastically reducing return variance."),
        ("A2C / A3C (Actor-Critic)", "Ch 10 (pp. 208–224)", "Unconstrained policy updates. One bad gradient can push policy off a cliff into irrecoverable collapse.", "PPO clips the surrogate objective to [1-eps, 1+eps], guaranteeing safe, monotonic policy updates."),
        ("TRPO (Trust Region)", "Ch 11 (pp. 240–244)", "Requires calculating second-order Fisher Information matrix inversions with conjugate gradients.", "PPO achieves trust-region safety using first-order clipping optimized with standard Adam.")
    ]

    for r, row_data in enumerate(data, start=1):
        bg = CARD_BG if r % 2 == 1 else RGBColor(241, 245, 249)
        for c, val in enumerate(row_data):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_BODY
            p.font.size = Pt(9.5)
            p.font.color.rgb = TEXT_MAIN
            if c == 0:
                p.font.bold = True

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "ALGORITHM JUSTIFICATION",
                "Explain to the professor: 'Grounded in Sudharsan Ravichandiran's book: DQN fails due to the 2^98 action space and replay buffer corruption; REINFORCE has high Monte Carlo variance; TRPO requires expensive second-order Hessian inversions. PPO gives us first-order trust-region clipping with GAE variance reduction.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 7:
"Professor, to justify our choice of algorithm, we grounded our work directly in the theoretical foundations from Sudharsan Ravichandiran’s textbook 'Hands-on Reinforcement Learning with Python':

1. Why not Q-Learning or SARSA (Chapters 3 & 5)? Because tabular methods cannot handle continuous state spaces. Our state has 22 continuous dimensions per signal.
2. Why not DQN (Chapter 8)? DQN is off-policy and relies on an Experience Replay Buffer. In a multi-agent city where 98 signals change simultaneously, the environment is non-stationary, which invalidates old transitions in the replay buffer. Furthermore, centralized DQN cannot optimize over 2^98 joint actions.
3. Why not REINFORCE (Chapter 11)? REINFORCE calculates gradients using full Monte Carlo returns across the entire episode. This produces high variance and poor sample efficiency.
4. Why not TRPO (Chapter 11, p. 240)? While TRPO enforces a trust region, it requires computing the second-order Fisher Information Matrix and conjugate gradient inversions, which is computationally prohibitive for high-throughput multi-agent training.
5. Why PPO (Chapter 11, p. 245)? PPO delivers the exact same trust-region mathematical stability as TRPO, but uses a first-order clipped surrogate objective that trains seamlessly via Adam optimization, combined with Generalized Advantage Estimation (GAE) to minimize variance."
""")


def build_slide_8(prs, blank_layout):
    """Slide 8: Graph Walkthrough - PPO Clipped Surrogate Objective (fig_ppo_clip.png)"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Mathematical Mechanics Figure", "Figure 2 Walkthrough: The Clipped Surrogate Objective", "Understanding how PPO prevents policy collapse through probability ratio bounding", 8)

    # Left: Embedded Image
    img_path = str(ROOT / "docs" / "professor_report" / "fig" / "fig_ppo_clip.png")
    add_card(slide, Inches(0.8), Inches(1.9), Inches(6.8), Inches(4.0))
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(0.95), Inches(2.2), width=Inches(6.5))

    # Right: Conceptual Breakdown
    add_card(slide, Inches(7.8), Inches(1.9), Inches(4.733), Inches(4.0))
    tb_r = slide.shapes.add_textbox(Inches(8.0), Inches(2.05), Inches(4.333), Inches(3.7))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "Mathematical Intuition of Figure 2"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    points = [
        ("The Probability Ratio r(θ):", "r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t). Measures how much the updated policy differs from the old policy."),
        ("Left Plot (Advantage > 0, Good Action):", "The dashed line shows unclipped gain. But PPO's blue curve FLATTENS at r = 1 + ε = 1.2. The model gets no extra credit for making an update larger than 20%."),
        ("Right Plot (Advantage < 0, Bad Action):", "If an action was bad, we want to decrease its probability. But the blue curve flattens at r = 1 - ε = 0.8. It prevents destructive over-penalization."),
        ("The Shaded Corridor [0.8, 1.2]:", "This is the Trust Region (ε = 0.2). Updates within this corridor are smooth; updates outside are clipped."),
        ("GAE (γ=0.99, λ=0.95):", "Generalized Advantage Estimation balances immediate reward (low variance) against future traffic delay (low bias).")
    ]
    for title, desc in points:
        pb = tf_r.add_paragraph()
        pb.text = f"{title} {desc}"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "FIGURE 2 WALKTHROUGH",
                "Explain to the professor: 'Point to Figure 2: The x-axis is r (how much the policy changed). On the left, if the action was good, the blue line clips at 1.2 so the agent doesn't get greedy. On the right, if the action was bad, it clips at 0.8 so the policy doesn't collapse. That 20% corridor guarantees training stability.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 8:
"Professor, Figure 2 visualizes the mathematical core of PPO: the Clipped Surrogate Objective from Ravichandiran Chapter 11.

Let's look closely at the two subplots:
1. The x-axis is r_t(theta)—the ratio of the new policy's probability over the old policy's probability. When r=1, the policy hasn't changed.
2. In the Left Subplot, the action was GOOD (Advantage > 0): Notice how the unclipped dashed line keeps rising indefinitely. If left unconstrained, the optimizer would take a massive, greedy step and overfit to one lucky batch. But the blue line—PPO's objective—flattens at 1 + epsilon = 1.2. Any probability increase beyond 20% yields zero additional gradient benefit!
3. In the Right Subplot, the action was BAD (Advantage < 0): We want to reduce the action's probability, but if we reduce it too aggressively, the policy can collapse into irreversible paralysis. PPO clips the loss at 1 - epsilon = 0.8.
4. The shaded blue corridor between 0.8 and 1.2 is our Trust Region. By restricting policy ratio movement to 20%, PPO guarantees smooth, monotonic policy improvements without requiring second-order Hessian inversions."
""")


def build_slide_9(prs, blank_layout):
    """Slide 9: Graph Walkthrough - Split Gradient Clipping (fig_gradnorm.png)"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Key Breakthrough Figure", "Figure 3 Walkthrough: Split Gradient Clipping Fix", "How decoupling Actor and Critic gradient norms solved the hidden 7,600x suppression bug", 9)

    # Left: Embedded Image
    img_path = str(ROOT / "docs" / "professor_report" / "fig" / "fig_gradnorm.png")
    add_card(slide, Inches(0.8), Inches(1.9), Inches(6.8), Inches(4.0))
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(0.95), Inches(2.35), width=Inches(6.5))

    # Right: The Discovery and Fix
    add_card(slide, Inches(7.8), Inches(1.9), Inches(4.733), Inches(4.0))
    tb_r = slide.shapes.add_textbox(Inches(8.0), Inches(2.05), Inches(4.333), Inches(3.7))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "The 'Aha!' Engineering Breakthrough"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = CORAL_ACCENT

    points = [
        ("The Initial Mystery:", "Early in the project, standard PPO failed against SUMO actuated. Policy entropy remained frozen at ~90%, and lights switched almost randomly."),
        ("The Discovery (Figure 3):", "We probed the raw gradients: Critic gradient norm was ~3,830 (due to large early value errors), while Actor gradient norm was only ~0.19!"),
        ("The Joint Clipping Bug:", "Standard Stable-Baselines3 clips all parameters jointly: Scale = 0.5 / sqrt(3830^2 + 0.19^2) ≈ 0.5 / 3830 ≈ 1.3 × 10^-4."),
        ("The Catastrophe:", "The actor's gradient (0.19) was multiplied by 1.3 × 10^-4, shrinking to 0.000025—below Adam optimizer's epsilon threshold (10^-8)! The actor was effectively turned off!"),
        ("Our Fix (split_clip.py):", "We decoupled gradient clipping: Actor clipped at 0.5; Critic clipped at 0.5 independently."),
        ("The Result:", "Policy learning accelerated by 10x immediately, and CityMind began decisively beating the civil engineering baseline.")
    ]
    for title, desc in points:
        pb = tf_r.add_paragraph()
        pb.text = f"{title} {desc}"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "FIGURE 3 WALKTHROUGH",
                "Explain to the professor: 'Point to Figure 3: Look at the top bar—the critic gradient was 3,830. The actor gradient at the bottom was only 0.19. Standard libraries clip them together, which scaled down the actor by 7,600 times and froze the policy. By clipping them independently, we solved the bug and unlocked 10x faster learning.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 9:
"Professor, this slide represents our single most important algorithmic breakthrough: Split Gradient Clipping, implemented in src/agents/split_clip.py.

During early training, we noticed a critical flaw: our PPO agent was stagnating. Policy entropy remained frozen at 90%, meaning the lights were switching almost randomly, and it couldn't beat SUMO actuated.

When we probed the gradient norms, we found the shocking disparity shown in Figure 3:
- The Critic network's gradient norm was surging to approximately 3,830 because traffic delay values created massive initial Temporal Difference errors.
- Meanwhile, the Actor network's gradient norm was only 0.19!

In standard RL libraries like Stable-Baselines3, actor and critic gradients are pooled together under a single global norm clip of 0.5. When SB3 normalized the total gradient, the scale factor was 0.5 divided by 3830—which is 1.3 times 10^-4. 

This scaled down the actor's gradient from 0.19 to 0.000025. That was smaller than Adam optimizer's epsilon threshold! The actor was literally put to sleep while the critic dominated the entire optimization budget.

Our solution was simple yet profound: SplitClipPPO. We separated the parameter groups and clipped the actor at 0.5 and the critic at 0.5 independently. The actor immediately woke up, learning speed surged by 10x, and our policy beat the baseline on every single seed."
""")


def build_slide_10(prs, blank_layout):
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


def build_slide_11(prs, blank_layout):
    """Slide 11: State, Action & Safety Architecture"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Environment Specification", "22-Dim Observation Space, Action Space & Safety Layer", "How 22 numbers describe any intersection geometry and how safety rules prevent crashes", 11)

    col_w = Inches(3.64)
    col_h = Inches(3.95)
    top_pos = Inches(1.9)

    # Card 1: 22-Dim Observation
    add_card(slide, Inches(0.8), top_pos, col_w, col_h)
    tb1 = slide.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "1. Observation Vector (22-Dim) 📊"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    bullets1 = [
        "Normalized to [-1, 1] for stable NN input.",
        "0–5: Active green phase (one-hot vector across up to 6 green phases).",
        "6: In-transition flag (1.0 during yellow or red clearance).",
        "7–8: Elapsed green duration relative to authored Min Green and Max Green.",
        "9–14: Varaiya pressure per green movement (Upstream + Feeders - Downstream).",
        "15: Max queue on worst approach (300m feeders).",
        "16: Pending outside-network vehicles waiting to enter.",
        "17: Number of valid green phases / 6.",
        "18–21: Neighbor queues (2 upstream, 2 downstream junctions)."
    ]
    for b in bullets1:
        pb = tf1.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9)
        pb.font.color.rgb = TEXT_MAIN

    # Card 2: Action Space
    add_card(slide, Inches(4.84), top_pos, col_w, col_h)
    tb2 = slide.shapes.add_textbox(Inches(5.04), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "2. Action Space: Discrete(2) 🚦"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    bullets2 = [
        "Action 0: KEEP current green phase.",
        "Action 1: ADVANCE to next sequential green phase.",
        "Why Discrete(2)? Avoids combinatorial phase selection bugs.",
        "Spatial Invariance: A 2-phase pedestrian crossing and a 6-phase major boulevard use the exact same action contract.",
        "Safety Decoupling: The neural network never directly commands raw light colors. It expresses intent ('keep' or 'advance'), which is evaluated by the SafetyLayer."
    ]
    for b in bullets2:
        pb = tf2.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    # Card 3: Safety Layer
    add_card(slide, Inches(8.88), top_pos, col_w, col_h)
    tb3 = slide.shapes.add_textbox(Inches(9.08), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "3. Deterministic Safety Layer 🛡️"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CORAL_ACCENT

    bullets3 = [
        "Implemented in src/environment/safety.py as a deterministic finite-state machine.",
        "Minimum Green Enforcement: Even if agent chooses 'advance', phase cannot switch until minimum green (10s) expires. Prevents light flickering.",
        "Maximum Green Cap: Forces phase advance if maximum green (60s) is exceeded. Prevents lane starvation.",
        "Mandatory Clearance Phases: Yellow (3-4s) and all-red clearance intervals are strictly executed in hardware before green switches.",
        "Zero Collision Risk: The agent physically cannot cause an accident in the simulator."
    ]
    for b in bullets3:
        pb = tf3.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9)
        pb.font.color.rgb = TEXT_MAIN

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "STATE & SAFETY CONTRACT",
                "Explain to the professor: 'Because junctions have different numbers of lanes (4 to 19), feeding raw lanes into a neural network is impossible. We engineered a compact 22-dimensional feature vector that works across all 98 signals, and backed it with a deterministic safety layer that enforces clearance intervals.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 10:
"Professor, on this slide we detail our observation space, action space, and safety guarantees:

First, how do you feed 98 completely different intersections—some with 4 lanes, some with 19 lanes—into one shared neural network? 
If you feed raw lane vectors, the input dimension changes at every intersection. Instead, we engineered a fixed 22-dimensional normalized feature vector:
- Features 0 to 5 encode the active green phase.
- Features 7 and 8 encode green time elapsed over minimum and maximum limits.
- Features 9 to 14 encode Varaiya pressure for each movement.
- Feature 15 tracks maximum queue on the worst feeder approach.
- Feature 16 tracks vehicles waiting outside the city entry gates.
- Features 18 to 21 encode neighbor queue lengths from the two most strongly connected upstream and downstream junctions.

Second, Action Space: We use Discrete(2): 0 means keep the current phase; 1 means advance to the next sequential phase.

Third, the Deterministic Safety Layer (src/environment/safety.py): In real-world civil engineering, an AI model cannot be allowed to command lights directly. If the agent demands an advance before minimum green is met, the safety layer ignores the action. If a light stays green too long, the maximum green timer overrides it. Furthermore, yellow change and all-red clearance intervals are executed deterministically. There is zero possibility of a collision."
""")


def build_slide_12(prs, blank_layout):
    """Slide 12: Other Breakthroughs (300m Feeders & Reward Function)"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Civil Engineering Realism", "Engineering Breakthroughs: 300m Feeders & Delay Cost", "Overcoming stop-line blindness and designing a non-gameable reward function", 12)

    col_w = Inches(5.733)
    col_h = Inches(3.95)
    top_pos = Inches(1.9)

    # Left: 300m Feeder Walker
    add_card(slide, Inches(0.8), top_pos, col_w, col_h)
    tb_l = slide.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "Breakthrough 2: 300-Meter Feeder Walker 🚗"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    bullets_l = [
        "The Stop-Line Blindness Trap: In European city centers, stop-lines can be as short as 15 to 20 meters, holding only 2 or 3 vehicles per lane.",
        "The Blindspot: If an RL agent only queries the stop-line lane, a massive 60-car traffic jam sitting 30 meters behind it is completely invisible!",
        "Our Solution (local_observation.py): We implemented a recursive upstream road crawler.",
        "300-Meter Depth: The algorithm walks backwards along incoming edges up to 300 meters, aggregating halted vehicles on all connected feeder lanes.",
        "Result: The agent can see platoons and traffic jams forming hundreds of meters away, giving it the foresight to extend green phases before spillback blocks cross-traffic."
    ]
    for b in bullets_l:
        pb = tf_l.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    # Right: Delay-Level Reward
    add_card(slide, Inches(6.8), top_pos, col_w, col_h)
    tb_r = slide.shapes.add_textbox(Inches(7.0), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "Breakthrough 3: Delay-Level Cost Reward 🎯"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    bullets_r = [
        "Why Change in Waiting Time (ΔW) Failed: Academic papers often reward 'decrease in waiting time'. Agents quickly game this reward by rapidly flipping lights—briefly reducing wait time while causing massive physical gridlock.",
        "Our Delay-Cost Function (local_reward.py): Formulated as a direct economic cost penalty:",
        "r_i(t) = -0.1 × min(Q_i + 2D_i + S_i + 0.5P_i + 0.5(P_total / 98), 200)",
        "Q_i = Halted vehicles on stop-line and 300m feeder approaches.",
        "D_i = Stopped vehicles experiencing severe delay (>30s).",
        "S_i = Phase switching penalty (discourages unnecessary phase changes).",
        "P_i & P_total = Vehicles queued outside the network waiting to enter.",
        "Result: Impossible to game. The agent maximizes reward only by flushing vehicles cleanly through the network."
    ]
    for b in bullets_r:
        pb = tf_r.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "ENGINEERING REALISM",
                "Explain to the professor: 'Two practical civil engineering fixes: First, stop-lines only hold 2 cars, so we recursively crawl upstream roads 300 meters to see real queues. Second, we abandoned delta-wait-time (which causes light-flipping oscillation) and used a direct delay-cost penalty that includes outside entry queues.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 11:
"Professor, on this slide we present two essential domain-engineering breakthroughs:

First, the 300-Meter Feeder Walker (src/environment/local_observation.py):
In real cities like Ingolstadt, stop-line lanes at intersections are often only 20 meters long, holding just 2 or 3 cars. If your RL agent only checks the stop-line lane, it suffers from severe blindness: a 60-car traffic jam sitting 30 meters behind the stop line is completely invisible to the agent!
To solve this, we implemented a recursive upstream search. Every 5 seconds, the observation builder walks upstream road segments up to 300 meters away, summing up all halted vehicles across feeder approaches. Now, the agent can see oncoming platoons and hold the green phase to let them pass.

Second, our Delay-Level Reward Design (src/environment/local_reward.py):
Many RL papers use 'change in waiting time' (delta W) as the reward. We found that agents easily game this reward: by rapidly flipping lights, the instantaneous wait-time calculation momentarily drops, giving the agent artificial reward even though traffic is gridlocked.
We replaced this with a robust delay-cost penalty:
r_i(t) = -0.1 * min(Q_i + 2*D_i + S_i + 0.5*P_i + 0.5*(P_total/98), 200).
It directly penalizes stopped queues (Q_i), severe delay (D_i), light-switching frequency (S_i), and vehicles stuck outside the city entry gates (P_i). This reward cannot be gamed: the only way to maximize it is to keep traffic moving smoothly."
""")


def build_slide_13(prs, blank_layout):
    """Slide 13: Evaluation Protocol - Survivor Bias & Drain Metrics"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Scientific Evaluation Protocol", "The Survivor Bias Trap & Drain Metrics Protocol", "Why standard academic metrics lie, and how our evaluation scores all scheduled demand", 13)

    col_w = Inches(5.733)
    col_h = Inches(3.95)
    top_pos = Inches(1.9)

    # Left: Survivor Bias
    add_card(slide, Inches(0.8), top_pos, col_w, col_h)
    tb_l = slide.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "The Survivor Bias Trap in Standard RL ❌"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = CORAL_ACCENT

    bullets_l = [
        "What standard literature reports: 'Mean Waiting Time of Completed Trips'.",
        "The Fatal Flaw: In a failing, gridlocked simulation, vehicles attempting long journeys across the city get trapped in queues and NEVER reach their destinations.",
        "Who actually completes? Only lucky vehicles taking short, direct trips on uncongested perimeter roads.",
        "The Paradoxical Lie: The simulator computes average wait time ONLY across the completed cars. Because only the fast, uncongested cars finished, the metric reports a falsely low waiting time!",
        "Severe Disguise: An agent causing catastrophic citywide gridlock can appear to have 'low wait times' simply by trapping all difficult trips forever."
    ]
    for b in bullets_l:
        pb = tf_l.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    # Right: Drain Metrics
    add_card(slide, Inches(6.8), top_pos, col_w, col_h)
    tb_r = slide.shapes.add_textbox(Inches(7.0), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "Our Solution: The Drain Metrics Protocol ✅"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    bullets_r = [
        "Implemented in src/evaluation/drain_metrics.py to eliminate survivor bias.",
        "Scores ALL ~15,000 Scheduled Trips: Evaluates every vehicle that was supposed to travel during the rush hour.",
        "Free-Flow Ideal Baseline: Calculates the theoretical zero-delay completion time for each car: ideal_k = depart_k + free_flow_time_k.",
        "Lag per Scheduled Car: Lag_k = min(actual_arrival_k, T_end) - min(ideal_k, T_end).",
        "Trapped Car Penalty: If a vehicle gets stuck or never enters the city, its completion time is penalized at the simulation boundary (T_end = 08:00).",
        "Total Network Accountability: A controller cannot score well without draining the entire city cleanly."
    ]
    for b in bullets_r:
        pb = tf_r.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "SURVIVOR BIAS ANALOGY",
                "Explain to the professor: 'Evaluating traffic by only completed cars is like evaluating a hospital by only counting patients who walked out alive! If an RL policy causes gridlock, cars get trapped and never finish. Our Drain Metrics score all 15,000 scheduled cars against ideal free-flow arrival, eliminating survivor bias.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 12:
"Professor, in this slide we address a subtle but pervasive methodological flaw in academic RL traffic literature: Survivor Bias.

Almost every published paper evaluates performance by measuring 'Mean Waiting Time of Completed Trips'. 
Here is why that metric is fatally flawed: If a bad RL policy causes massive gridlock across downtown, all long-distance commuters are trapped in jams and never reach their destination before the simulation ends. The only vehicles that complete are lucky drivers making short trips on suburban bypasses. If you compute the average wait time only of finished trips, the result looks falsely positive! Severe gridlock is completely disguised.

To establish rigorous scientific accountability, we developed the Drain Metrics Protocol (src/evaluation/drain_metrics.py):
Instead of counting only the survivors, we evaluate the entire scheduled demand of ~15,000 vehicles. For every scheduled trip, we calculate its theoretical zero-delay free-flow travel time. If a vehicle gets trapped in traffic or delayed at an entry gate, it is penalized at the simulation end time (08:00).
Our primary KPI is 'Lag per Scheduled Car'. It measures the true delay experienced by the entire city population. This protocol is completely immune to survivor bias."
""")


def build_slide_14(prs, blank_layout):
    """Slide 14: Graph Walkthrough - 10-Seed Seed-by-Seed Victory (fig_seed_lag.png)"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Statistical Evidence Figure", "Figure 4 Walkthrough: 10-Seed Head-to-Head Victory", "Every single gray line slopes downward: CityMind wins 10 out of 10 independent random seeds", 14)

    # Left: Embedded Image
    img_path = str(ROOT / "docs" / "professor_report" / "fig" / "fig_seed_lag.png")
    add_card(slide, Inches(0.8), Inches(1.9), Inches(6.8), Inches(4.0))
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(0.95), Inches(2.2), width=Inches(6.5))

    # Right: Statistical Deep Dive
    add_card(slide, Inches(7.8), Inches(1.9), Inches(4.733), Inches(4.0))
    tb_r = slide.shapes.add_textbox(Inches(8.0), Inches(2.05), Inches(4.333), Inches(3.7))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "Statistical Rigor of Figure 4"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    points = [
        ("The Evaluation Protocol:", "40 full 1-hour simulation runs across 10 random seeds (7, 11, 19, 23, 31, 42, 57, 91, 123, 777)."),
        ("What the Gray Lines Represent:", "Each connecting gray line is ONE exact traffic realization tested head-to-head across SUMO actuated, Model A, and Model B."),
        ("The Unbroken Downward Slope:", "Notice that every single line slopes downward from baseline to our models. There is not a single seed where the baseline won!"),
        ("Baseline (SUMO Actuated):", "Mean lag of 276.8 seconds per scheduled vehicle."),
        ("CityMind Model A:", "Mean lag of 235.4 seconds (-41.4 seconds, -15.0%). Won 10/10 seeds (p = 0.002, exact sign test)."),
        ("CityMind Model B:", "Mean lag of 239.9 seconds (-36.9 seconds, -13.3%). Won 10/10 seeds (p = 0.002, exact sign test)."),
        ("Confidence Interval:", "95% t-interval for Model A difference is [-43.9 s, -38.9 s], safely excluding zero.")
    ]
    for title, desc in points:
        pb = tf_r.add_paragraph()
        pb.text = f"{title} {desc}"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "FIGURE 4 WALKTHROUGH",
                "Explain to the professor: 'Point directly to Figure 4: The y-axis is lag per car. The left column is the civil engineering baseline; the middle is Model A. Each gray line is the exact same traffic seed. Every single line slopes downward! That proves our victory is not a random lucky seed—it is a 10-out-of-10 statistically verified result (p=0.002).'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 13:
"Professor, Figure 4 provides our definitive statistical proof.

In RL research, evaluating on a single seed is meaningless because traffic seeds can be arbitrarily easy or difficult. We tested across 10 independent random simulation seeds—a total of 40 full one-hour simulation runs comparing SUMO actuated, Model A, and Model B.

Please observe how Figure 4 is plotted:
- The y-axis is 'Lag per scheduled car' (lower is better).
- The x-axis shows the three controllers: SUMO Actuated on the left, Model A in the center, Model B on the right.
- Each gray line connects the exact same random traffic seed across the three controllers.

Look at the gray lines: Every single gray line slopes downward from the baseline to our RL models! There is zero overlap. 
On SUMO actuated, the mean lag is 276.8 seconds. On Model A, the mean lag drops to 235.4 seconds—saving 41.4 seconds per scheduled car across the entire city.
Under the exact non-parametric sign test, winning 10 out of 10 seeds yields p = 0.002. Furthermore, our 95% paired t-interval spans from -43.9 seconds to -38.9 seconds, bounded far below zero. This proves our victory is statistically undeniable."
""")


def build_slide_15(prs, blank_layout):
    """Slide 15: Graph Walkthrough - Multi-Metric Citywide Improvements (fig_improvement.png)"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Benchmark Comparison Figure", "Figure 5 Walkthrough: Multi-Metric Citywide Gains", "Longer bars indicate greater percentage improvement over SUMO Actuated across 14 metrics", 15)

    # Left: Embedded Image
    img_path = str(ROOT / "docs" / "professor_report" / "fig" / "fig_improvement.png")
    add_card(slide, Inches(0.8), Inches(1.9), Inches(6.8), Inches(4.0))
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(1.4), Inches(1.95), width=Inches(5.6))

    # Right: Metric Highlights Card
    add_card(slide, Inches(7.8), Inches(1.9), Inches(4.733), Inches(4.0))
    tb_r = slide.shapes.add_textbox(Inches(8.0), Inches(2.05), Inches(4.333), Inches(3.7))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "Key Metric Breakdown in Figure 5"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    points = [
        ("How to Read the Chart:", "Horizontal bars show percentage improvement over SUMO Actuated (longer = better). Blue = Model A; Orange = Model B. Numbers show seeds won (out of 10)."),
        ("Mean Waiting Time (-41.1%):", "The longest bar at the top! Waiting time at intersections dropped from 6.39s to 3.77s (10/10 seeds won)."),
        ("Mean Queue Length (-27.9%):", "Average junction queue dropped from 0.44 to 0.32 halted cars (10/10 seeds won)."),
        ("City Backlog (-8.2%, -256 Cars):", "At 08:00, 256 fewer cars were left trapped inside the city network (10/10 seeds won)."),
        ("Trips Completed (+258 Cars):", "Finished journeys increased by +258 vehicles (from 11,585 to 11,842, 10/10 seeds won)."),
        ("Zero Degradation:", "No metric is worse. Fixed-time, in contrast, failed on 11 out of 14 metrics.")
    ]
    for title, desc in points:
        pb = tf_r.add_paragraph()
        pb.text = f"{title} {desc}"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "FIGURE 5 WALKTHROUGH",
                "Explain to the professor: 'Point to Figure 5: This bar chart displays all 14 traffic metrics. Notice that every bar extends to the right—meaning positive improvement over SUMO Actuated. Notice the small numbers on the right of each bar: 10/10, 10/10, 10/10. We didn't sacrifice one metric to boost another; the entire network improved.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 14:
"Professor, Figure 5 provides a holistic cross-metric analysis.

In traffic engineering, a naive system might optimize one specific metric—for instance, increasing speed on a main avenue—while creating massive gridlock and queue pileups on side streets. Figure 5 proves that CityMind AI improved the entire system holistically:

Look at the bars:
1. Every single bar extends to the right, indicating percentage improvement over the SUMO Actuated baseline. Blue represents Model A; Orange represents Model B.
2. Look at the top bar: Mean Waiting Time reduced by 41.1% (Model A) and 39.6% (Model B), winning 10 out of 10 seeds.
3. Mean Queue Length reduced by 27.9% (10/10 seeds won).
4. Backlog at 08:00: We cleared 256 trapped vehicles that SUMO actuated left stranded in the city (10/10 seeds won).
5. Trips Completed: We successfully delivered 258 additional finished journeys within the rush-hour window.
6. Notice the small grey numbers on the bars: 10/10 on lag, 10/10 on time loss, 10/10 on backlog, 10/10 on waiting time, 10/10 on queue length, 10/10 on speed, 10/10 on throughput, and 10/10 on completion. Not a single measure was degraded."
""")


def build_slide_16(prs, blank_layout):
    """Slide 16: Comprehensive Head-to-Head Performance Table"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Numerical Performance Matrix", "Comprehensive Head-to-Head Benchmark Table", "Statistical comparison across SUMO Actuated, Fixed-Time, Model A, and Model B", 16)

    # 8-column data table
    rows = 9
    cols = 7
    left = Inches(0.8)
    top = Inches(1.9)
    width = Inches(11.733)
    height = Inches(3.95)

    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    table.columns[0].width = Inches(2.7)
    table.columns[1].width = Inches(1.5)
    table.columns[2].width = Inches(1.4)
    table.columns[3].width = Inches(1.5)
    table.columns[4].width = Inches(1.7)
    table.columns[5].width = Inches(1.4)
    table.columns[6].width = Inches(1.533)

    headers = ["Metric Evaluated", "SUMO Actuated (Base)", "Fixed-Time", "CityMind Model A", "Difference (vs Base)", "Seeds Won", "Statistical Verdict"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = SLATE_DARK
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_HEADING
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)

    table_data = [
        ("Lag per scheduled car (s)", "276.8 s", "298.7 s (+7.9%)", "235.4 s", "-41.4 s (-15.0%)", "10 / 10", "BEATEN (p=0.002)"),
        ("Time loss per finished car (s)", "336.6 s", "363.5 s (+8.0%)", "286.4 s", "-50.2 s (-14.9%)", "10 / 10", "BEATEN (p=0.002)"),
        ("Mean waiting time (s)", "6.39 s", "7.14 s (+11.7%)", "3.77 s", "-2.63 s (-41.1%)", "10 / 10", "BEATEN (p=0.002)"),
        ("Backlog at 08:00 (cars)", "3,108 cars", "3,297 cars", "2,852 cars", "-256 cars (-8.2%)", "10 / 10", "BEATEN (p=0.002)"),
        ("Mean queue length (cars)", "0.440", "0.492 (+11.8%)", "0.318", "-0.123 (-27.9%)", "10 / 10", "BEATEN (p=0.002)"),
        ("Mean vehicle speed (m/s)", "8.70 m/s", "8.40 m/s (-3.4%)", "9.37 m/s", "+0.68 m/s (+7.8%)", "10 / 10", "BEATEN (p=0.002)"),
        ("Trips completed (vehicles)", "11,585", "11,432 (-152)", "11,842", "+258 cars (+2.2%)", "10 / 10", "BEATEN (p=0.002)"),
        ("Depart delay (s)", "15.7 s", "15.3 s", "14.1 s", "-1.59 s (-10.2%)", "10 / 10", "BEATEN (p=0.002)")
    ]

    for r, row in enumerate(table_data, start=1):
        bg = CARD_BG if r % 2 == 1 else RGBColor(241, 245, 249)
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_BODY
            p.font.size = Pt(9)
            p.font.color.rgb = TEXT_MAIN
            if c == 0:
                p.font.bold = True
            elif c == 4:
                p.font.bold = True
                p.font.color.rgb = GREEN_ACCENT if "-" in val and "cars" not in val else GREEN_ACCENT
            elif c == 6:
                p.font.bold = True
                p.font.color.rgb = GREEN_ACCENT

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "BENCHMARK TABLE TAKEAWAY",
                "Explain to the professor: 'This table summarizes the 40 experimental runs. Look at Fixed-Time in column 3: it is worse than the baseline on almost every metric (+7.9% lag, +11.7% wait time). In contrast, CityMind Model A beats SUMO Actuated across the board with p=0.002 statistical significance.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 15:
"Professor, this table aggregates the exact numerical findings from our 10-seed paired experimental protocol (detailed in docs/TEST_REPORT_10_SEEDS.md):

Let's examine the key columns:
1. Column 2 is our baseline: SUMO Actuated.
2. Column 3 is the Fixed-Time controller: As you can see, fixed-time timers perform significantly worse than SUMO actuated—increasing lag by +7.9% and waiting time by +11.7%, losing on 11 out of 14 metrics. This confirms that dynamic vehicle responsiveness is essential.
3. Column 4 and 5 are CityMind Model A and its difference:
   - Lag per scheduled car dropped by 41.4 seconds (-15.0%).
   - Time loss per finished car dropped by 50.2 seconds (-14.9%).
   - Mean waiting time dropped by 2.63 seconds (-41.1%).
   - The city backlog at 08:00 was reduced by 256 vehicles.
   - Mean vehicle speed increased by +7.8%.
   - Over 258 additional journeys were completed.
4. Finally, look at Columns 6 and 7: Every single core metric achieved a clean 10 out of 10 seed sweep, with p = 0.002 on the exact sign test. The performance gains are consistent, statistically significant, and reproducible."
""")


def build_slide_17(prs, blank_layout):
    """Slide 17: Codebase Architecture & Reproducibility"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Engineering & Reproducibility", "Modular Codebase Architecture & Live Execution", "Clean modular software design with verified reproducibility commands", 17)

    col_w = Inches(5.733)
    col_h = Inches(3.95)
    top_pos = Inches(1.9)

    # Left: Codebase Map
    add_card(slide, Inches(0.8), top_pos, col_w, col_h)
    tb_l = slide.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "Modular Codebase Architecture 📂"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    modules = [
        ("src/agents/ippo_agent.py:", "PettingZoo-to-VecEnv wrapper, GAE calculator, and training harness."),
        ("src/agents/split_clip.py:", "Our custom optimizer performing decoupled Actor/Critic gradient norm clipping."),
        ("src/environment/sumo_connection.py:", "Lazy TraCI / libsumo socket proxy with automated fallback."),
        ("src/environment/decentralized_sumo_env.py:", "98-agent PettingZoo parallel traffic environment."),
        ("src/environment/local_observation.py:", "22-dim feature extractor and 300-meter feeder queue crawler."),
        ("src/environment/local_reward.py:", "Delay-cost reward function with boundary backlog tracking."),
        ("src/environment/safety.py:", "Deterministic clearance and min/max green state machine."),
        ("src/evaluation/drain_metrics.py:", "Survivor-bias-free scheduled demand evaluation.")
    ]
    for mod, desc in modules:
        pb = tf_l.add_paragraph()
        pb.text = f"• {mod} {desc}"
        pb.font.size = Pt(9)
        pb.font.color.rgb = TEXT_MAIN

    # Right: CLI Verification
    add_card(slide, Inches(6.8), top_pos, col_w, col_h)
    tb_r = slide.shapes.add_textbox(Inches(7.0), top_pos + Inches(0.18), col_w - Inches(0.4), col_h - Inches(0.36))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "Live Verification & Reproduction Commands 💻"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    bullets_r = [
        "100% Runnable & Self-Contained: Managed with uv and virtual environment (.venv).",
        "Command 1: Run 10-Seed Head-to-Head Evaluation:",
        "  uv run python scripts/evaluate.py --help",
        "Command 2: Launch Live SUMO Visual GUI Demo:",
        "  uv run python scripts/run_visual_demo.py --render",
        "Command 3: Train IPPO from Scratch:",
        "  uv run python scripts/run_ippo_training.py",
        "Command 4: Re-generate Professor Figures & Report:",
        "  uv run python docs/professor_report/make_figures.py",
        "Automated CI & Testing: Passes pytest, ruff formatting, and model contract validation."
    ]
    for b in bullets_r:
        pb = tf_r.add_paragraph()
        pb.text = b
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_MAIN
        if "uv run" in b:
            pb.font.name = "Courier New"
            pb.font.color.rgb = BADGE_TXT_BLUE

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "SOFTWARE ENGINEERING RIGOR",
                "Explain to the professor: 'Everything in this project is modular, fully typed, unit-tested, and reproducible with simple uv run CLI commands. If you would like to see a live demonstration right now, we can run the visual GUI demo or inspection script immediately.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 16:
"Professor, beyond theoretical rigor, we treated this project with high software engineering standards.

On the left is our architectural file structure:
- src/agents contains our PettingZoo IPPO implementation and our custom SplitClip optimizer.
- src/environment isolates the TraCI connection, observation builder, reward function, and safety state machine.
- src/controllers maintains the SUMO actuated and fixed-time baselines.
- src/evaluation contains our drain metrics engine.

On the right are our execution and reproducibility commands. The entire environment is managed using uv and pyproject.toml:
- Anyone can clone the repository and re-run our 10-seed evaluation, launch a live SUMO graphical simulation using scripts/run_visual_demo.py, or train from scratch.
- Everything is working, tested, and ready for live demonstration."
""")


def build_slide_18(prs, blank_layout):
    """Slide 18: Summary & Conclusion (Dark Executive Theme)"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide, is_dark=True)

    # Accent decorative bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(0.8), Inches(1.2), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = GREEN_ACCENT
    bar.line.color.rgb = GREEN_ACCENT

    # Title
    t_box = slide.shapes.add_textbox(Inches(1.0), Inches(1.0), Inches(11.3), Inches(1.2))
    tf = t_box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "Conclusion: Why CityMind AI Succeeds"
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(255, 255, 255)

    p2 = tf.add_paragraph()
    p2.text = "Summary of key contributions and theoretical breakthroughs in city-wide traffic control"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(13)
    p2.font.color.rgb = RGBColor(203, 213, 225)

    # 4 Pillar Summary Cards
    pillars = [
        ("1. Real-World Scale 🇩🇪", "Evaluated on Ingolstadt InTAS: 98 heterogeneous intersections, >15k vehicles, Krauss car-following physics, beating calibrated induction loops."),
        ("2. Solid RL Theory 📚", "Formulated as Dec-POMDP with parameter-shared IPPO, GAE variance reduction, and clipped surrogate loss grounded in Ravichandiran (2018)."),
        ("3. Engineering Breakthroughs 🛠️", "Invented Split Gradient Clipping (fixing 7,600x actor suppression), 300m feeder walker, delay-cost reward, and deterministic safety layer."),
        ("4. Decisive 10-Seed Victory 🏆", "Slashing lag by -41.4s (-15%) and wait times by -41.1%, clearing 256 trapped vehicles, winning 10/10 seeds with p=0.002.")
    ]

    card_w = Inches(5.5)
    card_h = Inches(1.8)
    positions = [(Inches(1.0), Inches(2.3)), (Inches(6.8), Inches(2.3)),
                 (Inches(1.0), Inches(4.3)), (Inches(6.8), Inches(4.3))]

    for (title, desc), (pos_x, pos_y) in zip(pillars, positions):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, pos_x, pos_y, card_w, card_h)
        card.fill.solid()
        card.fill.fore_color.rgb = RGBColor(24, 34, 53)
        card.line.color.rgb = RGBColor(51, 65, 85)
        tf_c = card.text_frame
        tf_c.word_wrap = True
        tf_c.margin_top = Inches(0.18)
        tf_c.margin_left = Inches(0.2)
        tf_c.margin_right = Inches(0.2)

        p_t = tf_c.paragraphs[0]
        p_t.text = title
        p_t.font.name = FONT_HEADING
        p_t.font.size = Pt(13.5)
        p_t.font.bold = True
        p_t.font.color.rgb = GREEN_ACCENT if "Victory" in title else BLUE_ACCENT

        p_d = tf_c.add_paragraph()
        p_d.text = desc
        p_d.font.name = FONT_BODY
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = RGBColor(226, 232, 240)

    # Ready for Q&A Banner
    qa_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.7))
    qa_card.fill.solid()
    qa_card.fill.fore_color.rgb = RGBColor(30, 41, 59)
    qa_card.line.color.rgb = BLUE_ACCENT
    tf_q = qa_card.text_frame
    p_q = tf_q.paragraphs[0]
    p_q.alignment = PP_ALIGN.CENTER
    p_q.text = "🎯 THANK YOU, PROFESSOR. READY FOR YOUR QUESTIONS & DISCUSSION."
    p_q.font.name = FONT_HEADING
    p_q.font.size = Pt(12)
    p_q.font.bold = True
    p_q.font.color.rgb = RGBColor(255, 255, 255)

    set_speaker_notes(slide, """
CLOSING DEFENSE SCRIPT (Deliver with confidence):
"To conclude our presentation, Professor:
CityMind AI demonstrates that Decentralized Deep Reinforcement Learning can successfully outperform industry-standard calibrated civil engineering across an authentic, heterogeneous city network of 98 intersections.

Our success rests on four pillars:
1. First, realistic scale: We evaluated on genuine European city geometry in Ingolstadt, with microscopic Krauss physics and 15,000 scheduled trips.
2. Second, textbook RL foundations: Grounded in Sudharsan Ravichandiran’s textbook, we formulated traffic control as a Dec-POMDP with parameter-shared IPPO and GAE advantage estimation.
3. Third, critical engineering breakthroughs: We discovered and resolved the 7,600x actor gradient suppression bug through Split Gradient Clipping, expanded perception to 300-meter feeder queues, and enforced zero-collision physical safety.
4. And fourth, bulletproof empirical proof: Slashing lag by 41.4 seconds per vehicle, reducing intersection wait times by 41.1%, and beating the SUMO actuated baseline across 10 out of 10 independent random seeds with p = 0.002.

Thank you very much for your time and guidance throughout this semester. I am now delighted to take any questions or demonstrate the live code."
""")


def build_slide_19(prs, blank_layout):
    """Slide 19: Appendix - Rapid Defense Q&A Pocket Reference"""
    slide = prs.slides.add_slide(blank_layout)
    add_canvas(slide)
    add_header(slide, "Appendix & Oral Defense Reference", "Professor Q&A Cheat Sheet: Rapid-Fire Answers", "Targeted technical answers for anticipated professor questions", 19)

    # 6 Q&A Cards (2 rows of 3 columns)
    qa_items = [
        ("Q1: Why not Centralized DQN?",
         "Centralized joint action space is 2^98 ≈ 3.16 × 10^29 actions. Finding argmax_a Q(s,a) is NP-hard. In multi-agent, off-policy experience replay buffers become invalid due to environment non-stationarity."),

        ("Q2: What if TraCI socket drops or lags?",
         "We wrapped TraCI in a resilient connection hub (sumo_connection.py). For training, we use libsumo—a direct C++ SWIG binding that runs in-process, eliminating TCP sockets and network latency entirely."),

        ("Q3: Why Tanh activations in MLP?",
         "Tanh produces zero-centered, smooth gradients bounded in [-1, 1]. In continuous-control PPO, ReLU often leads to dead neurons and aggressive policy shifts, whereas Tanh ensures stable advantage updates."),

        ("Q4: How does Safety Layer prevent crashes?",
         "The RL agent never sets light colors directly. SafetyLayer implements a deterministic state machine: when advancing, it strictly inserts authentic yellow (3-4s) and all-red clearance phases before activating the next green."),

        ("Q5: Why GAE gamma=0.99 and lambda=0.95?",
         "Grounded in Ravichandiran Ch. 10: gamma=0.99 ensures the policy values clearing future downstream queues. lambda=0.95 optimally trades off Temporal Difference bias against Monte Carlo rollout variance."),

        ("Q6: Why are teleports noisy in results?",
         "Teleports occur only when extreme gridlock persists for 300+ seconds. In a 1-hour run, teleports are rare (7.5 vs 13.3). They dropped by 43%, but due to high variance across seeds, the sign test is p=0.508 (directional).")
    ]

    card_w = Inches(3.64)
    card_h = Inches(1.9)
    col_x = [Inches(0.8), Inches(4.84), Inches(8.88)]
    row_y = [Inches(1.9), Inches(4.0)]

    idx = 0
    for r in range(2):
        for c in range(3):
            q, a = qa_items[idx]
            card = add_card(slide, col_x[c], row_y[r], card_w, card_h)
            tb = slide.shapes.add_textbox(col_x[c] + Inches(0.15), row_y[r] + Inches(0.12), card_w - Inches(0.3), card_h - Inches(0.24))
            tf = tb.text_frame
            tf.word_wrap = True

            pq = tf.paragraphs[0]
            pq.text = q
            pq.font.name = FONT_HEADING
            pq.font.size = Pt(11)
            pq.font.bold = True
            pq.font.color.rgb = BADGE_TXT_BLUE

            pa = tf.add_paragraph()
            pa.text = a
            pa.font.name = FONT_BODY
            pa.font.size = Pt(8.5)
            pa.font.color.rgb = TEXT_MAIN
            idx += 1

    add_callout(slide, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.0), "ORAL DEFENSE STRATEGY",
                "Tip for the student: 'If the professor asks any technical question, keep calm, refer to this slide, and connect your answer to: 1. Ravichandiran's book principles, 2. The physical realities of traffic in Ingolstadt, or 3. Our exact code implementation in src/.'")

    set_speaker_notes(slide, """
SPEAKER SCRIPT FOR SLIDE 18 (BACKUP REFERENCE):
"Keep this slide ready as backup during the Q&A session. 
If the professor asks:
- 'Why did you use Tanh instead of ReLU?' -> Point to Q3: Tanh provides smooth bounded gradients that prevent policy collapse in continuous-control Actor-Critic networks.
- 'How do you prevent collisions?' -> Point to Q4: The deterministic SafetyLayer decouples policy output from physical hardware commands, enforcing yellow and all-red clearance intervals.
- 'Why not Centralized DQN?' -> Point to Q1: 2^98 action space explosion and replay buffer invalidation."
""")


def main():
    print(f"Creating presentation at: {OUTPUT_FILE}")
    prs, blank_layout = create_presentation()

    build_slide_1(prs, blank_layout)
    build_slide_2(prs, blank_layout)
    build_slide_3(prs, blank_layout)
    build_slide_4(prs, blank_layout)
    build_slide_5(prs, blank_layout)
    build_slide_6(prs, blank_layout)
    build_slide_7(prs, blank_layout)
    build_slide_8(prs, blank_layout)
    build_slide_9(prs, blank_layout)
    build_slide_10(prs, blank_layout)
    build_slide_11(prs, blank_layout)
    build_slide_12(prs, blank_layout)
    build_slide_13(prs, blank_layout)
    build_slide_14(prs, blank_layout)
    build_slide_15(prs, blank_layout)
    build_slide_16(prs, blank_layout)
    build_slide_17(prs, blank_layout)
    build_slide_18(prs, blank_layout)
    build_slide_19(prs, blank_layout)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_FILE))
    print(f"Presentation successfully saved! Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
