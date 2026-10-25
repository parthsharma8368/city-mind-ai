# Models

Trained weights are **not tracked in git** (`*.zip` and `artifacts/experiments/` / `artifacts/models/*` are ignored).
`artifacts/models/MANIFEST.json` (tracked) records which file produced every reported number, with SHA-256.

| Model | What it is | File | SHA-256 | Size |
|---|---|---|---|---|
| `stage1_seed42` | stage 1: 100 updates, training seed 42 | `artifacts/experiments/obs_compare/v3r_100/run_20261013_192101_seed42/model_final.zip` | `75b2694f9a8942e3874e97a3420ab59eb494fd619894bc231a4b67479c84d2cd` | 1.83 MB |
| `stage1_seed1042` | stage 1: 100 updates, training seed 1042 | `artifacts/experiments/obs_compare/v3r_100_seed1042/run_20261022_182635_seed1042/model_final.zip` | `8481653b61bdbfe2a1c2ec0679da0b8671993413f1ccc6add5960d5308eeba24` | 1.83 MB |
| `final_seed42` | FINAL: stage 1 (seed 42) + 10 fine-tune updates with pending_in_demand | `artifacts/models/ippo_final_seed42/model_final.zip` | `7a94580dd00cd82d3bdcf527399d243f94381408f19ba81c58aabf7145d28e29` | 1.83 MB |
| `final_seed1042` | FINAL: stage 1 (seed 1042) + 20 fine-tune updates with pending_in_demand | `artifacts/models/ippo_final_seed1042/model_final.zip` | `347380e38f81903042c8bb37781e6a4c81e552f255ca159e9dd0957fed84d69d` | 1.83 MB |

**Two final models, both reported:** `final_seed42` (stage 1 seed 42 + 10 fine-tune updates) and `final_seed1042` (stage 1 seed 1042 + 20 fine-tune updates).
Stage 1 is `configs/ippo.yaml` + `configs/environment.yaml` (100 PPO updates); stage 2 continues the same network with `configs/ippo_finetune.yaml` +
`configs/environment_pending.yaml` (`scripts/finetune_ippo.py`, 10 updates per call). Each model directory holds `config.yaml` (frozen settings; the evaluator
reads the observation options from it) and the weights.

Stage 1 reproduces across the two training seeds (mean lag 251.0 and 250.8 s). Stage 2 closes the cars-never-inserted gap for both (10-seed means: 501 and 481 vs 503 for SUMO actuated, i.e. level);
see `docs/TEST_REPORT_10_SEEDS.md`.

## Evaluate

```bash
python scripts/evaluate.py --controller ippo --model artifacts/models/ippo_final_seed42/model_final.zip --out artifacts/eval/ippo_42
python scripts/evaluate.py --controller ippo --model artifacts/models/ippo_final_seed1042/model_final.zip --out artifacts/eval/ippo_1042
python scripts/evaluate.py --controller sumo_actuated --out artifacts/eval/sumo_actuated
```
