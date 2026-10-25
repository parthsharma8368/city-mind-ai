"""Evaluate one controller on SUMO with the drain-aware protocol (the project's single evaluation entry point).

  python scripts/evaluate.py --controller sumo_actuated|fixed_time|ippo [--model MODEL.zip] \
      --seeds 42 123 777 --begin 25200 --duration 3600 --out DIR [--min-green-floor SECONDS]

* ``ippo`` needs --model; the observation version, queue reach, timing source and action mode are read from the
  run's config.yaml next to the model, so a model is always evaluated with the inputs it was trained on.
* Any window works; the scheduled demand and the zero-delay reference are rebuilt for it.
* Results: DIR/metrics/<controller>_seed<seed>_1.0x.json (+ trajectories under DIR/traj).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", required=True, choices=["sumo_actuated", "fixed_time", "ippo"])
    p.add_argument("--model", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    p.add_argument("--begin", type=int, default=25200, help="window start, seconds since midnight (default 07:00)")
    p.add_argument("--duration", type=int, default=3600)
    p.add_argument("--out", required=True)
    p.add_argument("--sumo-cfg", default="data/raw/intas/scenario/InTAS_train.sumocfg")
    p.add_argument("--tls-json", default="artifacts/scenario/tls_metadata.json")
    p.add_argument("--eval-config", default="configs/evaluation.yaml")
    p.add_argument("--min-green-floor", type=float, default=None, help="what-if: minimum green for static-program phases")
    a = p.parse_args()
    if a.controller == "ippo" and not a.model:
        p.error("--controller ippo needs --model")

    from src.evaluation.evaluate import EvaluationRunner

    out = Path(a.out)
    cfg = yaml.safe_load(Path(a.eval_config).read_text())          # protocol defaults, incl. ff_calibration
    cfg.update({"seeds": a.seeds, "begin_time": a.begin, "episode_duration": a.duration,
                "output_dir": str(out / "metrics"), "trajectory_dir": str(out / "traj")})
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        cfg_path = f.name
    runner = EvaluationRunner(
        sumo_cfg=a.sumo_cfg, tls_metadata_json=a.tls_json, eval_config_path=cfg_path,
        ippo_model_path=a.model, controllers=[a.controller],
        output_dir=str(out / "metrics"), trajectory_dir=str(out / "traj"),
    )
    runner.safety_min_green = a.min_green_floor
    results = runner.run_all()
    print("DONE", [(r.controller, r.seed, r.trips_completed, r.backlog) for r in results])


if __name__ == "__main__":
    main()
