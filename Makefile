# Convenience targets. Every target is a thin wrapper around `osb`.
PY ?= python

.PHONY: install dev test smoke lint format data-online data-offline doctor

install:            ## Core + training dependencies.
	$(PY) -m pip install -e ".[train,lora,llm,xlsx]"

dev:                ## Everything including linters/tests.
	$(PY) -m pip install -e ".[all]"

test:               ## Unit tests + synthetic end-to-end smoke test.
	$(PY) -m pytest -q

smoke:              ## Synthetic end-to-end pipeline (CPU, ~1 min).
	osb smoke --workdir smoke_run

lint:               ## Google-style lint (pylint) + format check (pyink).
	pyink --check src tests
	isort --check-only src tests
	pylint src/opensrbench

format:
	isort src tests
	pyink src tests

doctor:
	osb doctor --network

data-online:        ## Network stages (login node).
	osb data --stages online

data-offline:       ## Offline stages (compute node).
	osb data --stages offline
