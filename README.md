# Flood defence condition grading

This is the code artifact for my MSc Final project - predicting Environment Agency flood defence condition grades with machine learning, and comparing them against Halcrow condition deterioration curves.

This repo contains dvc reciepts to raw and processed data, so transfer is easy when requested. 

This repo contains notebooks for original EDA, source code for repeated data preperation, modelling, and evaluation processes, notebooks for investigating the heuristics, experiments for training models, and notebooks to evaluate the results. 

```
src/            shared modelling code
experiments/    one package per experiment, plus its outputs
notebooks/      numbered analysis, .ipynb plus a couple of R .qmd
report/         Quarto book
scripts/        one off feature builds and checks
data/           DVC tracked, not in the repo
```

The project has been given an MIT licence, making reproduction and extention free and unrestricted. 

Python is used for the data work and the modelling. R is used inside Quarto documents for some of the plots and statistical output, with Parquet files as handoffs between the two.

## Environments

Two separate environments, each with its own lockfile. Both are committed, so an install should reproduce exactly what was used for the results.

### Python

Managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml` holds the dependency ranges and `uv.lock` pins the resolved versions, and `.python-version` pins Python to 3.13.

```
uv sync
```

That creates `.venv` and installs everything from the lock. Run things through `uv run` rather than activating the venv:

```
uv run python -m experiments.final_grid.runner.py
```

To add a package use `uv add polars`, which updates both the pyproject and the lock. Editing `pyproject.toml` by hand then running `uv sync` works too.
`uv lock --upgrade` re-resolves everything, which is worth avoiding mid-experiment.

The project itself is installed in editable mode, which is what makes `from src... import` work from anywhere in the repo.

`pyproject.toml` also carries a `[tool.project-paths]` table. Those are read by `project_paths` and are how the code finds data and output directories, rather than hardcoding relative paths:

```python
from project_paths import paths

paths.processed_data / "unified_aims_eir_bgs.parquet"
```

### R

Managed with [renv](https://rstudio.github.io/renv/). `renv.lock` pins the package versions against R 4.4.1, and `.Rprofile` activates the project library whenever R starts in this directory.

Open R in the repo root and run:

```r
renv::restore()
```

Then rendering the Quarto documents (`quarto render report`, or a single `.qmd`) picks the packages up automatically. renv only tracks packages it can see being used, so after adding a `library()` call somewhere run `renv::snapshot()` to update the lock. `.renvignore` keeps it out of `src/`, `experiments/`, `data/` and the `.ipynb` notebooks, so it only scans the `.qmd` files and does not try to resolve Python imports as R packages.

### Data

Data files are tracked with DVC rather than git, and are not in the repo. Most of it is open (AIMS, EIR, BGS), but the joined `data/processed/unified_aims_eir_bgs.parquet` is what nearly everything downstream reads.

## Running an experiment

Each experiment lives in its own package under `experiments/`, and each is self contained: the
feature compositions, model grid, runner and result loading for that experiment only. Once run
they are not modified, so earlier results stay reproducible.

```
uv run python -m experiments.final_grid.runner --smoke   # short pass, writes to *_smoke
uv run python -m experiments.final_grid.runner           # the real thing
```

The grid experiments write one file per cell into the experiment directory:

```
folds.parquet               fold assignment, shared by every cell
oof/{cell_id}.parquet       out of fold predicted probabilities
shap_agg/{cell_id}.parquet  mean absolute shap per feature x grade x fold
meta/{cell_id}.json         per cell record
runs.jsonl                  one line per invocation
```

A cell whose output already exists is skipped, so a run can be interrupted and restarted without losing what it had finished. Deleting the output directory forces a clean run.

No metrics are computed during the run. Everything scored comes from the OOF probabilities afterwards, in the numbered notebook that matches the experiment(`21_final_grid_experiment_results.ipynb` for `final_grid`)

## Writing a new experiment

`src/` holds the parts that are not specific to any one experiment:

| module | what it gives you |
| --- | --- |
| `features.py` | `prepare_features` (target, filters, enums), `create_arrays`, and `LEAKY_COLS` |
| `cross_validate.py` | `produce_spatial_blocks`, `fold_ids_from_splitter`, `run_oof` |
| `metrics.py` | scorers - quadratic weighted kappa, PR-AUC for grade >= 4, per grade precision/recall/F1 |
| `heuristic.py` | the Halcrow curves, for the heuristic baseline |
| `frank_hall.py` | ordinal classifier built from binary cut models |
| `oof_shap.py` | TreeExplainer SHAP shaped to match the OOF frame |

The usual shape of a new experiment is a package with `features.py`, `models.py`, `runner.py` and
`reporting.py`, where:

- `features.py` defines the column lists and calls `create_arrays` to build the matrices
- `models.py` defines the grid and a `make_model(spec)`
- `runner.py` loads the parquet, gates it, builds folds once, then calls `run_oof` per cell
- `reporting.py` loads the artefacts back for the analysis notebook

`experiments/baseline/baseline.py` is the smallest example and a reasonable starting point.
`experiments/final_grid/` is the fullest one, with parallelism, resume and SHAP.
