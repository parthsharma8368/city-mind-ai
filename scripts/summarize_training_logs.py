"""Summarise SB3 training logs: how far did the policy move, and is it learning?  Usage: summarize_training_logs.py LOG..."""
import re
import sys
from pathlib import Path

KEYS = ["total_timesteps", "ep_rew_mean", "clip_fraction", "approx_kl", "entropy_loss", "explained_variance",
        "value_loss", "grad_norm_actor", "grad_norm_critic"]


def blocks(text: str):
    for b in re.split(r"-{20,}", text):
        row = {}
        for k in KEYS:
            m = re.search(r"\|\s+" + k + r"\s+\|\s+([-\d.e+]+)", b)
            if m:
                row[k] = float(m.group(1))
        if "clip_fraction" in row:
            yield row


if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    sys.exit("usage: summarize_training_logs.py TRAIN_LOG [TRAIN_LOG ...]   (SB3 training logs; prints first / middle / last update)")
for path in sys.argv[1:]:
    rows = list(blocks(Path(path).read_text(errors="ignore")))
    print(f"\n{path}: {len(rows)} updates logged")
    if not rows:
        continue
    print("  " + " ".join(f"{k[:12]:>12}" for k in KEYS))
    for r in (rows[0], rows[len(rows) // 2], rows[-1]):
        print("  " + " ".join(f"{r.get(k, float('nan')):12.4g}" for k in KEYS))
