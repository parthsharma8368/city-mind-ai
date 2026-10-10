# Contributing

## Setup
```bash
make install && make sumo && source .env.sumo
make lint test                      # must pass before a PR
cd frontend && npm ci && npm run lint && npm test && npm run build
pre-commit install                  # optional: ruff, large-file check
```
Python 3.11 (3.12 allowed by `pyproject.toml`, only 3.11 is tested in CI). `make install-lock`
reproduces CI's pinned dependencies; regenerate the lock with
`uv pip compile pyproject.toml --extra dev --extra sumo -o requirements.lock --python-version 3.11`.

## Style and tooling
* Ruff (config in `pyproject.toml`) is blocking in CI. Existing violations are grandfathered via the
  "baseline debt" ignore list; remove entries as you clean code up, and do not add new violations.
* mypy runs non-strict and non-blocking; do not increase the error count (44 on `src api` when this config was added; 69 after concurrent edits - see CI).
* Logging: follow `docs/LOGGING.md`.
* Keep changes focused; no mass reformatting mixed with behaviour changes.

## Tests
`pytest -q` runs without SUMO scenario data. Tests that need a live SUMO simulation must be
marked/skipped when SUMO is unavailable.

## Results claims
Rank controllers by completed-trip metrics (trips completed, mean trip timeloss, backlog,
teleports), not by in-network mean waiting time alone - that metric rewarded starvation here.
If you retrain a headline model, update `artifacts/models/MANIFEST.json` and `docs/MODELS.md`.

## Pull requests
Use the PR template, describe how you tested, and note changes to metrics or APIs in `CHANGELOG.md`.
Security issues: see `SECURITY.md`.
