PYTHON := uv run python

.PHONY: install lock fmt lint test clean \
	data-small data-medium \
	train eval-probe eval-rollout eval-collapse

install:
	uv sync --dev

lock:
	uv lock

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest

clean:
	rm -rf .pytest_cache .ruff_cache

data-small:
	$(PYTHON) scripts/generate_data.py --config config/data/small.yaml

data-medium:
	$(PYTHON) scripts/generate_data.py --config config/data/medium.yaml

train:
	$(PYTHON) scripts/train.py --config config/base.yaml

eval-probe:
	$(PYTHON) scripts/eval_probe.py --run_dir $(RUN_DIR)

eval-rollout:
	$(PYTHON) scripts/eval_rollout.py --run_dir $(RUN_DIR)

eval-collapse:
	$(PYTHON) scripts/eval_collapse.py --run_dir $(RUN_DIR)
