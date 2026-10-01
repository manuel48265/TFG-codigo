.PHONY: check format test experiment1 experiment2 experiment3

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

experiment2:
	uv run python -m src.Experiments.Experimental_cases.run_pc_experiment --workers $(WORKERS)

experiment3:
	uv run python -m src.Experiments.Experimental_cases.run_real_experiment --workers $(WORKERS)
