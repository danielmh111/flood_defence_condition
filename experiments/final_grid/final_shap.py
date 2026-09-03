# uncapped per-row shap at the selected config, both arms, all 10 folds (spec 7). this is the
# beeswarm material for the final chapter and is the reason the pass exists - the main grid only
# writes shap_agg for the reference config, because at max_iter up to 960 TreeExplainer cost scales
# with tree count and explaining all 2,880 cells could exceed the fit time entirely.
#
# runs after notebook 21 identifies the config, which is set below as a module constant rather than
# passed on the command line - a run that could be pointed at a different config would not be the
# experiment.
#
#   uv run python -m experiments.final_grid.final_shap
#   uv run python -m experiments.final_grid.final_shap --smoke

# runner first, before numpy or sklearn reaches any import - see the note in frank_hall.py
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

import sys

from loguru import logger

from experiments.final_grid.models import REFERENCE_CONFIG, build_configs, build_cells, check_unique

# selected in notebook 21, not the argmax. the argmax of mean pr_auc sits on three axis boundaries
# at once - 31 leaves, min_samples_leaf 5, 255 bins - which is where an argmax over 2,880 cells
# drifts, and 126 cells are within one paired standard error of it. this config ranks 6 of 1,440 on
# identity_lidar_climate_raw and 3 of 1,440 on identity_lidar_full, about a fifth of that standard
# error behind, and it is the only near-top config that is near-top on both arms - which this pass
# needs, since one config is explained across both.
FINAL_SHAP_CONFIG = {
    "capacity": 2.5,
    "learning_rate": 0.0125,
    "max_leaf_nodes": 16,
    "min_samples_leaf": 5,
    "max_bins": 255,
    "l2_regularization": 0.0,
}


def main(smoke: bool = False) -> None:
    out_dir = SMOKE_OUT if smoke else OUT
    n_splits = SMOKE_SPLITS if smoke else N_SPLITS
    n_workers = SMOKE_WORKERS if smoke else N_WORKERS

    if FINAL_SHAP_CONFIG not in build_configs():
        raise ValueError(f"FINAL_SHAP_CONFIG is not a member of the grid: {FINAL_SHAP_CONFIG}")

    context = prepare(out_dir, n_splits)

    cells = check_unique(
        build_cells(
            context["cat_idx_by_composition"],
            [FINAL_SHAP_CONFIG],
            shap_for=FINAL_SHAP_CONFIG,
            shap_tier="full",
        )
    )
    logger.info(f"{len(cells)} cells at {FINAL_SHAP_CONFIG}")

    # these cells already have an oof from the main grid, so the inherited completion flag would
    # skip every one of them. the per-row artefact is what this pass is for, so that is what marks
    # it done. the refit rewrites oof and meta - the oof content is identical, the fit being
    # deterministic, and the meta's seconds_* then describe this pass, which `pass` disambiguates.
    run_pass(
        cells,
        out_dir,
        min(n_workers, len(cells)),
        pass_name="final_shap",
        record={
            "smoke": smoke,
            "n_splits": n_splits,
            "n_assets": context["n_assets"],
            "n_configs": 1,
            "final_shap_config": FINAL_SHAP_CONFIG,
            "compositions": context["names"],
        },
        done_dir="shap_full",
    )


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
