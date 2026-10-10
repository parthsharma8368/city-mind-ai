# Evaluation protocol notes (audit D2)

## Which metric to rank on
Per-completed-trip means (`mean_trip_timeloss`, `mean_waiting_time`) and `trips_completed` are
survivor-biased: about 16% of tripinfo rows are zero-length artifact trips, and a controller that never
lets long trips finish or never lets cars in at a congested entry edge looks better on them. Vehicles
blocked at insertion for more than `max-depart-delay` (300 s) are discarded by SUMO and are in no metric.

The drain-aware protocol (`src/evaluation/drain_metrics.py`, protocol id `drain-v1`) scores every run
against the SCHEDULED demand (14,942 cars in 07:00-08:00, 1,864 of them zero-length):

* `lag_per_scheduled_car_s` - mean over ALL scheduled cars of (actual completion, or 08:00 if unfinished or
  never inserted) minus (calibrated zero-delay completion time). Lower is better. Headline metric.
* `due_completion_ratio` - share of zero-delay-feasible, non-zero-length cars that completed by 08:00.
* per-entry-edge completion table (380271165#0, 450072140, 266507542, -233673422#10 + top-10 origin edges),
  completion share per 10-minute departure cohort, `never_inserted`, `teleports` from SUMO's own
  `--statistic-output` (live runs) or the MetricsCollector count (offline rescoring).
* per-trip means exclude zero-length trips (`*_nonzero` fields); the legacy fields are unchanged.

Offline rescoring of the stored artifacts (no simulation):
`python scripts/rescore_drain_metrics.py [--plot]` -> `artifacts/metrics/drain_scores.{json,csv,md}`.
It reproduces the EDA's `per_run_metrics.csv` to floating-point precision.

## Statistics: what the 3 seeds can and cannot support
The seeds (42/123/777) change only SUMO's random draws; each RL controller was trained once, so the
spread is simulation noise and says nothing about training variance. With n = 3 paired seeds the minimum
detectable effect is about 4.4 SD of the paired difference (80% power, alpha 0.05, df = 2). `src/evaluation/statistics.py`
therefore reports paired-by-seed mean differences, t-based CIs (n stated), a bootstrap CI that is flagged
unreliable for n < 6, and the MDE, and deliberately computes no p-values. Do not describe n = 3 results as
"significant" or "reproducible, not noise" without that qualifier.

## Reproducibility of the stored sumo_actuated seed-42 result (audit eval-scripts-4)
Facts established on 2026-10-10:

* The stored tripinfo headers of all 18 canonical runs record `route-steps=100` and no `buildings.poly.xml`,
  i.e. they were produced with `data/raw/intas/scenario/InTAS_train.sumocfg`, NOT `InTAS_buildings.sumocfg`
  (which the README and the old script defaults pointed to; it has route-steps 200 and building polygons and
  different dynamics).
* A standalone `sumo` run (SUMO 1.27.1, `--seed 42 --begin 25200 --end 28800`, InTAS_train.sumocfg, 60 s wall
  clock) reproduces the stored `sumo_actuated_seed42_1.0x.xml` exactly: 11,621 trips, mean timeLoss 283.6717 s,
  identical id/depart/arrival sequence, 6 teleports (SUMO statistic output).
* The re-run that "did not reproduce" (11,604 trips, timeLoss 277.81 s) used InTAS_buildings.sumocfg.
  So the stored numbers are valid for InTAS_train.sumocfg and are NOT a machine/concurrency artifact.
* Consequence: pass `--sumo-cfg data/raw/intas/scenario/InTAS_train.sumocfg` for any run you want to compare with
  the stored results. Every new result JSON records the sumocfg path + sha256 and a `config_hash`; results written
  before this change have no provenance (legacy) and are refused for overwrite unless `--overwrite` is passed.
* The ippo_phase_select results (2026-09-29) also carry the InTAS_train header, so they are comparable with the
  baselines.

## Result files
Canonical window (07:00-08:00): `artifacts/metrics/{controller}_seed{seed}_{demand}.json` (API contract).
Any other window: `artifacts/metrics/runs/b<begin>_d<duration>/..._b<begin>_d<duration>.json`. Each JSON carries
`provenance` (git commit + dirty flag, sumocfg path + sha256, SUMO version, begin/end, step length, seed, model
path + sha256, command line) and `drain_protocol`. Metric-definition versions are `metrics_schema_version`
(src/environment/schema.py); stored pre-2026-10 results are implicitly v1.

## Known gaps
* Old stored results are v1 (pre-unification metrics) and have no provenance; drain scores are computed from
  their tripinfo only, so teleports come from the stored MetricsCollector count and `never_inserted` from
  `total_departed`.
* PPO (centralized) evaluation cannot emit SUMO's statistic output through `make_env` (it builds its own argv);
  the runner appends the output options to the env's `sumo_extra_args` when that attribute is a list.
* Training-seed variance is not measured (one trained model per RL controller).
