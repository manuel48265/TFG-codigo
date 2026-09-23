.PHONY: check format test

check:
	./scripts/check.sh

format:
	./scripts/format.sh

test: 
	uv run pytest
