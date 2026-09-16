.PHONY: install test test-live lint bench bench01 bench02 bench03 bench04 bench05 \
        models all

install:
	uv sync --extra dev

test:
	uv run pytest -q -m "not live"

test-live:
	uv run pytest -q

lint:
	uv run ruff check shared projects tests scripts
	uv run ruff format --check shared projects tests

bench: bench01 bench02 bench03 bench04 bench05

bench01:
	uv run python -m projects.p01_revision_loops.benchmark

bench02:
	uv run python -m projects.p02_router_misroute.benchmark

bench03:
	uv run python -m projects.p03_parallel_merge.benchmark

bench04:
	uv run python -m projects.p04_checkpoint_resume.benchmark

bench05:
	uv run python -m projects.p05_supervisor_handoff.benchmark






shots:
	npm install
	node scripts/shoot.mjs --project p01
	node scripts/shoot.mjs --project p02
	node scripts/shoot.mjs --project p03
	node scripts/shoot.mjs --project p04
	node scripts/shoot.mjs --project p05

models:
	ollama pull qwen2.5:3b-instruct

all: install lint test
