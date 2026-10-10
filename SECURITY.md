# Security Policy

## Supported versions
Only the latest commit on `main` (pre-1.0, version 0.1.x) receives fixes.

## Reporting a vulnerability
Please do **not** open a public issue. Use GitHub's private "Report a vulnerability" (Security
advisories) on this repository, or email the maintainer (address in the git log / GitHub profile).
We aim to acknowledge within 7 days. This is a research project without a bug bounty.

## Threat model summary
* **Network exposure**: the API is intended to run on loopback (`127.0.0.1`). It has no user
  accounts. Simulation-control endpoints (launch SUMO-GUI, etc.) are guarded by `CITYMIND_API_KEY`;
  when the key is unset they are open and the API logs a warning. Set the key before binding to any
  non-loopback address. Docker Compose publishes ports on host loopback only.
* **CORS** is restricted to local dev origins by default (`CITYMIND_CORS_ORIGINS`).
* **Pickle / model loading**: Stable-Baselines3 `.zip` checkpoints contain pickled Python objects;
  loading a malicious checkpoint can execute arbitrary code. Only load checkpoints you trained or
  whose SHA-256 matches `artifacts/models/MANIFEST.json` (see `docs/MODELS.md`).
* **Processes**: the API spawns SUMO/SUMO-GUI with fixed argument lists (no shell). File-path
  inputs are validated by the API (see api/security.py).
* **Dependencies**: pinned in `requirements.lock`; CI runs a non-blocking `pip-audit` / `npm audit`
  report nightly. Known: dev-only `source-map-js` advisory in the frontend build tooling.
* **Secrets**: never commit `.env`; `.env.example` documents all variables.
