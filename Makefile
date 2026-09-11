.PHONY: install test demo lint format clean journal help

# Default target
.DEFAULT_GOAL := help

help:  ## Show this help message
	@echo "Affective Fly: Makefile targets"
	@echo "================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dependencies (uv)
	@echo "Installing affective-fly with uv..."
	uv pip install -e .
	@echo "Installation complete!"

install-dev:  ## Install with development dependencies
	@echo "Installing affective-fly with dev dependencies..."
	uv pip install -e ".[dev]"
	@echo "Dev installation complete!"

install-all:  ## Install with all optional dependencies
	@echo "Installing affective-fly with all extras..."
	uv pip install -e ".[brian2,viz,dev]"
	@echo "Full installation complete!"

test:  ## Run pytest test suite
	@echo "Running tests..."
	pytest -v

test-cov:  ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest --cov=src/affective_fly --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

demo:  ## Run all three demos in sequence
	@echo "Running demos..."
	@echo "\n=== Demo 1: Basic Loop ==="
	python examples/demo_loop.py
	@echo "\n=== Demo 2: Mood Launch ==="
	python examples/demo_mood_launch.py
	@echo "\n=== Demo 3: Swarm ==="
	python examples/demo_swarm.py
	@echo "\nAll demos complete!"

demo-loop:  ## Run basic affective loop demo
	python examples/demo_loop.py

demo-launch:  ## Run mood-conditioned launch demo
	python examples/demo_mood_launch.py

demo-swarm:  ## Run swarm demo
	python examples/demo_swarm.py

journal:  ## Run demo and export journal to JSON
	@echo "Running demo and exporting journal..."
	python examples/demo_loop.py
	@echo "Journal exported to: demo_journal.json"
	@echo "View with: cat demo_journal.json | jq ."

lint:  ## Run ruff linter
	@echo "Running ruff linter..."
	ruff check src/ tests/ examples/

lint-fix:  ## Run ruff linter with auto-fix
	@echo "Running ruff linter with auto-fix..."
	ruff check --fix src/ tests/ examples/

format:  ## Format code with ruff
	@echo "Formatting code..."
	ruff format src/ tests/ examples/

type-check:  ## Run mypy type checker
	@echo "Running mypy..."
	mypy src/affective_fly

clean:  ## Clean build artifacts and caches
	@echo "Cleaning..."
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f demo_journal.json
	@echo "Clean complete!"

build:  ## Build package (wheel and sdist)
	@echo "Building package..."
	python -m build

docs-serve:  ## Serve documentation locally (requires mkdocs)
	@echo "Serving docs at http://127.0.0.1:8000"
	mkdocs serve

check-all: lint type-check test  ## Run all checks (lint, type-check, test)
	@echo "All checks passed!"
