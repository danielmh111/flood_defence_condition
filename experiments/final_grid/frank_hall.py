# frank-hall as a paired comparison, not a grid axis (spec 6). four binary models per cell means
# ~4x the cost, so gridding it would quadruple the main run to answer a question that is answerable
# at a handful of configs.
#
# runs after the main grid, on the same folds, writing into the same directory. cell_id hashes the
# model name, so a frank-hall cell never collides with the hgb cell at identical params.
#
# no shap: src/oof_shap.py assumes one model emitting a 3-D multiclass array, and four binary models
# need four explainers and a different aggregation. what these cells log instead is the per-fold
# inversion rate - see models.frank_hall_diagnostics.
#
#   uv run python -m experiments.final_grid.frank_hall
#   uv run python -m experiments.final_grid.frank_hall --smoke

# runner first, before numpy or sklearn reaches any import: importing it executes its thread-pinning
# block, and under spawn this module is re-executed as __mp_main__ in every worker. this ordering is
# load-bearing, not style.
import sys

from loguru import logger

from experiments.final_grid.models import REFERENCE_CONFIG, build_cells, check_unique
from experiments.final_grid.runner import (
    N_SPLITS,
    N_WORKERS,
    OUT,
    SMOKE_OUT,
    SMOKE_SPLITS,
    SMOKE_WORKERS,
    prepare,
    run_pass,
)

# three configs, each running on both arms - six cells. the reference config is here because it is
# the cell shared with the fine grid; the selected config because it is the model the chapter
# reports; and the qwk-strongest because frank-hall's whole rationale is ordinal structure, and
# configs that maximise pr_auc_ge4 land around rank 1,000 of 1,440 on qwk in both arms. testing the
# decomposition only where pr_auc is maximised would test it where it is least likely to show
# anything.
FRANK_HALL_CONFIGS = [
    dict(REFERENCE_CONFIG),
    {
        "capacity": 2.5,
        "learning_rate": 0.0125,
        "max_leaf_nodes": 16,
        "min_samples_leaf": 5,
        "max_bins": 255,
        "l2_regularization": 0.0,
    },
    {
        "capacity": 4.0,
        "learning_rate": 0.05,
        "max_leaf_nodes": 8,
        "min_samples_leaf": 5,
        "max_bins": 255,
        "l2_regularization": 1.0,
    },
]


def main(smoke: bool = False) -> None:
    out_dir = SMOKE_OUT if smoke else OUT
    n_splits = SMOKE_SPLITS if smoke else N_SPLITS
    n_workers = SMOKE_WORKERS if smoke else N_WORKERS

    context = prepare(out_dir, n_splits)

    cells = check_unique(
        build_cells(
            context["cat_idx_by_composition"], FRANK_HALL_CONFIGS, model="frank_hall"
        )
    )
    logger.info(
        f"{len(FRANK_HALL_CONFIGS)} configs x {len(cells) // len(FRANK_HALL_CONFIGS)} arms"
    )

    run_pass(
        cells,
        out_dir,
        min(n_workers, len(cells)),
        pass_name="frank_hall",
        record={
            "smoke": smoke,
            "n_splits": n_splits,
            "n_assets": context["n_assets"],
            "n_configs": len(FRANK_HALL_CONFIGS),
            "frank_hall_configs": FRANK_HALL_CONFIGS,
            "compositions": context["names"],
        },
    )


if __name__ == "__main__":
    main()
