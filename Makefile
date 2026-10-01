.PHONY: check format test experiment1

check:
	./scripts/check.sh

format:
	./scripts/format.sh

test: 
	uv run pytest

# Resumable: run it again to continue; WORKERS=1 for the cleanest timings.
WORKERS ?= 8
experiment1:
	uv run python -m src.Experiments.Experimental_cases.run_experiment --workers $(WORKERS)
