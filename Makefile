.PHONY: lint lint-fix format-check format check test cov cov-html

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

# Run tests with coverage (terminal report)
cov:
	uv run --extra test pytest tests/ --cov=diffusers_mm --cov-report=term-missing

# Run tests with HTML coverage report in htmlcov/
cov-html:
	uv run --extra test pytest tests/ --cov=diffusers_mm --cov-report=html
