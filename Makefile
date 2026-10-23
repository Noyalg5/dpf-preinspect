# Pipeline entry points. CONFIG selects the profile: tiny, demo or full.
CONFIG ?= demo
VENV ?= .venv
PY := $(VENV)/bin/python
CLI := $(VENV)/bin/preinspect
UV := $(shell command -v uv 2>/dev/null)
export PYTORCH_ENABLE_MPS_FALLBACK := 1

.PHONY: setup data train evaluate validate simulate report api app replay smoke demo full test lint format clean

setup:
ifdef UV
	test -x $(PY) || uv venv --python-preference only-managed --python 3.11 $(VENV)
	VIRTUAL_ENV=$(VENV) uv pip install -e ".[dev,app]"
else
	test -x $(PY) || python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,app]"
endif

data:
	$(CLI) data --config $(CONFIG)

train:
	$(CLI) train --config $(CONFIG)

evaluate:
	$(CLI) evaluate --config $(CONFIG)

validate:
	$(CLI) validate --config $(CONFIG)

simulate:
	$(CLI) simulate --config $(CONFIG)

report:
	$(CLI) report --config $(CONFIG)

api:
	$(CLI) api --config $(CONFIG)

app:
	$(CLI) app

replay:
	$(CLI) replay --config $(CONFIG)

smoke:
	$(CLI) smoke --config $(CONFIG)

demo:
	$(MAKE) data train evaluate validate simulate report smoke CONFIG=demo

full:
	$(MAKE) data train evaluate validate simulate report smoke CONFIG=full

test:
	$(PY) -m pytest

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

format:
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

clean:
	rm -rf runs data .pytest_cache .ruff_cache
