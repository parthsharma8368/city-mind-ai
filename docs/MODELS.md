# Models: headline results to run directories

Trained weights are **not tracked in git** (`*.zip` and `artifacts/experiments/` are ignored).
This page and `artifacts/models/MANIFEST.json` (tracked) record exactly which checkpoint
produced each headline number, with SHA-256 so a downloaded copy can be verified.

| Controller | Run directory (under `artifacts/experiments/`) | Algorithm | SHA-256 of `model_final.zip` | Size |
|---|---|---|---|---|
| `ppo` | `ppo/run_20260917_174650_seed42` | SB3 PPO, centralized `MultiBinary(98)` | `6ada56dcacc6187d070c6adfc82114e89e87d14ff60f6dc52f0169eb9a2341c2` | 31.97 MB |
| `ippo` | `ippo/run_20260921_194748_seed42` | SB3 PPO, parameter-shared, `Discrete(2)` | `3185128367bfab0f2e490b40687a52b137b2dcb0fd67035d0e4e7c4551452050` | 1.74 MB |
| `ippo_phase_select` | `ippo_phase_select/run_20260924_085542_seed42` | sb3-contrib MaskablePPO, `Discrete(6)` masked | `f46f1b85d34ba40bf80767c8071540e02c1aeb3e2e973473af83a9591572d68a` | 1.76 MB |

How the mapping works: every tracked `artifacts/metrics/<controller>_seed{42,123,777}_1.0x.json`
carries a `source_run_id` field naming the run directory above. Each run directory also holds
`config.yaml` (frozen hyperparameters), `git_commit.txt` (code version used), `checkpoints/`
and `tb_logs/`. The manifest copies the recorded commit for each run
(`b165297...` for PPO, `3706d23...` for IPPO, `5238b9d...` for phase-select; the last was
written by a resumed run, so it reflects the resume commit, not the first launch).

Other run directories (earlier IPPO attempts, the first failed phase-select attempt
`run_20260923_111708_seed42`, diagnostics) are kept locally for history and are not headline results.

## Obtaining and verifying weights

1. Download the three zips from the repository's GitHub Release (asset names are in
   `MANIFEST.json` -> `release_asset`; none has been published yet - maintainers: attach them
   to the first tagged release).
2. Place each at its `model_file` path from the manifest, then verify:
   `shasum -a 256 artifacts/experiments/ippo/run_20260921_194748_seed42/model_final.zip`
3. Evaluate: `python scripts/run_evaluation.py ... --ppo-model <path>`;
   phase-select uses `python scripts/run_ippo_maskable_eval.py --model <path> ...`.

Without the weights you can still reproduce all four non-learning baselines and retrain.

## Compatibility and safety notes

* `ippo` (`--ippo-model` in `run_evaluation.py`) was trained on an older observation layout;
  the current env uses `12 + MAX_GREEN_PHASES` dims, so that checkpoint cannot be re-evaluated
  against the current env (known limitation; its metrics JSONs were produced earlier).
* SB3 `.zip` checkpoints contain pickled objects. **Only load checkpoints whose SHA-256 you have
  verified against the manifest** (see `SECURITY.md`).
* Library versions at manifest time: stable-baselines3 2.9.0, sb3-contrib 2.9.0, torch 2.14.0,
  gymnasium 1.3.0, numpy 1.26.4 (training-time versions were not recorded in the runs).

## Updating the manifest

When a headline run changes: add the run, recompute `sha256` and `size_bytes`, copy
`git_commit.txt`, update the metrics `source_run_id`, and update this table.
