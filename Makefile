.DEFAULT_GOAL := help
PY      ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
API_PORT ?= 8000

.PHONY: help install install-lock sumo lint typecheck test run-api run-frontend eval audit docker-up
help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## create .venv and install package + dev extras
	uv venv .venv --python 3.11 || python3.11 -m venv .venv
	uv pip install --python .venv/bin/python -e ".[dev]"

install-lock:  ## reproducible install pinned by requirements.lock
	uv pip install --python .venv/bin/python -c requirements.lock -e ".[dev]"

sumo:  ## check SUMO, fetch pinned InTAS, install training sumocfg
	./scripts/setup_sumo.sh

lint:  ## ruff
	$(PY) -m ruff check .

typecheck:  ## mypy (non-strict baseline; may report errors)
	$(PY) -m mypy src api

test:  ## backend tests
	$(PY) -m pytest -q

run-api:  ## API on loopback
	$(PY) -m uvicorn api.main:app --host 127.0.0.1 --port $(API_PORT) --reload

run-frontend:  ## Vite dev server
	cd frontend && npm install && npm run dev

eval:  ## baselines + RL eval (needs SUMO + InTAS + models; see docs/MODELS.md)
	$(PY) scripts/run_evaluation.py --sumo-cfg data/raw/intas/scenario/InTAS_buildings.sumocfg \
	  --tls-json artifacts/scenario/tls_metadata.json --output-dir artifacts/metrics --plot

audit:  ## dependency vulnerability report
	$(PY) -m pip_audit -r requirements.lock --no-deps --disable-pip || true
	cd frontend && npm audit --omit=dev || true

docker-up:  ## build and start api + frontend containers
	docker compose up --build
