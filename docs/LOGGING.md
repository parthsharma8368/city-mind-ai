# Logging convention

Status: convention only. Today most modules use `print`; new and touched code should follow this.

* Libraries (`src/`, `api/`): `log = logging.getLogger(__name__)`; never call `logging.basicConfig`
  at import time and never `print` for diagnostics.
* Entry points (scripts `main()`, API startup) configure logging once, e.g.
  `logging.basicConfig(level=os.getenv("CITYMIND_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")`.
* `print` is acceptable only for deliberate CLI output (tables, final summaries).
* Levels: DEBUG per-step simulation detail; INFO episode/run milestones; WARNING recoverable
  anomalies (teleports, starvation, missing optional data); ERROR failed operations.
* Training runs: also log to `<run_dir>/train.log` and TensorBoard (`tb_logs/`); `*.log` is git-ignored.
* Never log `CITYMIND_API_KEY` or request headers.
* Env var `CITYMIND_LOG_LEVEL` is the single knob (not yet implemented in code; tracked as follow-up).
