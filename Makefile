.PHONY: install lint format typecheck docstrings check test ci-docker docs build clean

PYTHON_VERSIONS ?= 3.12 3.13 3.14

install:        ## Sync all dependency groups into the uv environment
	uv sync --all-groups

lint:           ## Ruff lint and format check
	uv run ruff check .
	uv run ruff format --check .

format:         ## Ruff format (writes changes)
	uv run ruff format .

typecheck:      ## Pyright on the package source
	uv run pyright

docstrings:     ## Pydoclint on the package source
	uv run pydoclint src/geoestimate

check:          ## Everything CI runs
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pydoclint src/geoestimate
	uv run pytest
	uvx preen check --strict

test:           ## Run the test suite
	uv run pytest

ci-docker:      ## Test each supported Python version in a standard uv image
	@for version in $(PYTHON_VERSIONS); do \
		docker run --rm \
			-v "$$(pwd):/workspace" -w /workspace \
			-e UV_PROJECT_ENVIRONMENT=/tmp/geoestimate-venv \
			-e UV_PYTHON=$$version -e UV_PYTHON_DOWNLOADS=never \
			"ghcr.io/astral-sh/uv:python$$version-bookworm" \
			sh -c 'uv sync --all-groups --frozen && uv run pytest' || exit 1; \
	done

docs:           ## Build the HTML documentation
	cd docs && make html

build:          ## Build sdist + wheel
	uv build

clean:          ## Remove build/doc artifacts
	rm -rf dist build docs/_build
