.PHONY: lint lint-fix format-check format check test test-envs-fast test-envs-real cov cov-html

# Lint: check for code quality issues
lint:
	uv run --extra dev ruff check src/ tests/

# Auto-fix linter issues
lint-fix:
	uv run --extra dev ruff check --fix src/ tests/

# Check formatting (no changes)
format-check:
	uv run --extra dev ruff format --check src/ tests/

# Auto-format code
format:
	uv run --extra dev ruff format src/ tests/

# Lint + format check (CI-friendly, no modifications)
check: format-check lint

# Run tests (CPU only, no GPU required)
test:
	uv run --extra test pytest tests/ -v

# Run only the fast env-decision tests (subset of `test`, no GPU)
test-envs-fast:
	uv run --extra test pytest tests/test_envs_fast.py -v

# Run env-real tests (opt-in: requires CUDA + downloads diffusers model weights
# on first run). To also constrain host RAM, wrap with:
#   systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0 make test-envs-real
test-envs-real:
	DIFFUSERS_MM_RUN_GPU_TESTS=1 uv run --extra test pytest tests/test_envs_real.py -v -s

# Run tests with coverage (terminal report)
cov:
	uv run --extra test pytest tests/ --cov=diffusers_mm --cov-report=term-missing

# Run tests with HTML coverage report in htmlcov/
cov-html:
	uv run --extra test pytest tests/ --cov=diffusers_mm --cov-report=html
