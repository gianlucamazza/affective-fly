.PHONY: install test lint format demo benchmark figures journal clean

install:
	uv sync --all-extras

test:
	uv run pytest tests/ -v --cov=affective_fly --cov-report=term-missing

lint:
	uv run ruff check src/ tests/ examples/
	uv run mypy src/

format:
	uv run ruff format src/ tests/ examples/
	uv run ruff check --fix src/ tests/ examples/

demo:
	@echo "Running demo loop..."
	uv run python examples/demo_loop.py
	@echo "\nRunning mood launch demo..."
	uv run python examples/demo_mood_launch.py
	@echo "\nRunning swarm demo..."
	uv run python examples/demo_swarm.py
	@echo "\nRunning three-factor learning demo..."
	uv run python examples/demo_td.py
	@echo "\nRunning SQLite persist demo..."
	uv run python examples/demo_persist.py
	@echo "\nRunning delayed-US demo..."
	uv run python examples/demo_cs_us.py
	@echo "\nRunning resonance demo..."
	uv run python examples/demo_resonance.py

benchmark:
	@echo "Running circuit backend benchmark sweep (needs brian extra for Brian2)..."
	uv run python examples/benchmark_brian2_codegen.py

figures:
	@echo "Regenerating docs/figures/ (runs benchmark first; needs viz extra)..."
	uv run python examples/benchmark_brian2_codegen.py
	uv run python examples/make_figures.py

journal:
	uv run python -m affective_fly journal $(JOURNAL)

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf src/**/__pycache__ tests/**/__pycache__ examples/**/__pycache__
	rm -rf *.egg-info build dist
	rm -f .coverage
	rm -f affective_fly.db affective_fly.mood.json journal.jsonl
