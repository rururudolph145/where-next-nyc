#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/ci-setup.sh"

uv run ruff check .
uv run ruff format --check .
uv run mypy app tests
