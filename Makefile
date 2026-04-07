# Prevent the host shell's VIRTUAL_ENV from leaking into uv subprocesses
unexport VIRTUAL_ENV

.PHONY: build preflight lint format typecheck test evals retrieval-eval ui clean

build:
	uv venv
	uv sync --all-groups

preflight: format lint typecheck test

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy .

test:
	uv run pytest -v

evals:
	uv run python -m evals.run_evals --no-llm

retrieval-eval:
	uv run python -m evals.retrieval_eval

ui:
	uv run python app.py

clean:
	rm -rf .venv .mypy_cache .ruff_cache __pycache__ .pytest_cache .mosaic.db
