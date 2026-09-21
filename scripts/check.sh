#!/usr/bin/env bash
set -e
uv run ruff check .
uv run ruff format --check .
uv run mypy src test
uv run pytest
