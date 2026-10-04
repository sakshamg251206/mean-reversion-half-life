# Reproduce everything: make all   (Python 3.13 via uv; see README)
PY = uv run python

.PHONY: env test data run holdout notebooks paper all

env:
	uv sync

test:
	uv run pytest -q

data:
	$(PY) experiments/01_data.py

run:
	$(PY) experiments/02_main_run.py
	$(PY) experiments/03_replication.py
	$(PY) experiments/04_halflife.py
	$(PY) experiments/05_predictive.py
	$(PY) experiments/06_robustness.py

holdout:   # requires experiments/frozen.yaml committed and unchanged
	$(PY) experiments/07_holdout.py

notebooks:
	$(PY) notebooks/build_notebooks.py
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_pair_walkthrough.ipynb notebooks/02_results_tour.ipynb

paper:
	cd paper && pdflatex -interaction=nonstopmode main && bibtex main && pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main

all: env test data run holdout notebooks paper
