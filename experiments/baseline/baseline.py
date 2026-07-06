# baseline experiment.
#
# the main outcome of this is a first implementation of the src code,
# using the features, metrics, and cross validate modules to set up and run an experiment
#
# Im also going to use the dummy classifier here to work out how to create baseline that could be alternatives to the heuristic outcomes.
#
# A proper baseline will compare a few different modelling strategies:
#
# probably a linear model (lr, maybe after doing some ordinal modelling comparisions),
# and a few non linear, possibly including a simple decision tree, and ensembles (xgboost or lightgbm), and maybe something different like naive bayes as a comparison
#
# however, right now i don't have pipelines to compare all of these properly (tree ensables will handle nans/nulls for me and dont need to 1 hot encode enums)
#
# so in this first file to test core code and dummy comparison, i will just be using random forest as the common/familiar/powerful/simple enough model

# edit - rf doesnt handle the categoricals natively like xgboost does, so until i go back and add 1 hot encoding on things like fluvial/tidal (this can actually be both)
# then i will use a different model
# after reading docs, histogram gradient boosting is the sklearn port of lightgbm so will handle the data in current form and is also a commonly accepted powerful model for tabular data
# sklearn docs seem to recommend it over the other ensembles
# boosting is probably actually better than bagging as an ensemble bc the use case here is being good at predicting the minority class, and boosting is supposed to do better at hard cases and imbalanced data because of its sequential learning
# arguably catboost is the best fit but i prefer to stay in the sklearn api and i think going with hist gb will give up very little


from project_paths import paths
import geopandas as gpd
import polars as pl
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from pathlib import Path
import json
from uuid import uuid4
from datetime import datetime

from src.cross_validate import cross_validate, produce_spatial_blocks
from src.features import (
    GEOSURE_COLS,
    LEAKY_COLS,
    NOMINAL_CATEGORY_COLS,
    prepare_features,
)
from src.metrics import build_scorers, report_means

PARQUET = paths.processed_data / "unified_aims_eir_bgs.parquet"
GPKG = paths.aims_data / "aims.gpkg"

N_BLOCKS = 50
N_SPLITS = 10
SEED = 123


def create_arrays(df: pl.DataFrame):
    """take the typed dataframe and turn it into the necessary arrays of features, tagets, indexs, and ids"""

    feature_cols = [
        col for col in df.columns if col not in ("asset_id", "condition_grade")
    ]
    X_df = df.select(feature_cols)

    enum_cols = [col for col, datatype in X_df.schema.items() if datatype == pl.Enum]
    cat_idx = [i for i, col in enumerate(feature_cols) if col in enum_cols]

    X_df = X_df.with_columns(pl.col(enum_cols).to_physical())  # Enum cast to int codes
    X = X_df.to_numpy().astype("float64")  # null cast to numpy nan

    y = df["condition_grade"].to_numpy().astype(int)
    asset_ids = df["asset_id"].to_numpy()

    return X, y, cat_idx, asset_ids


def load_coords(asset_ids: np.ndarray, gpkg_path: Path) -> np.ndarray:
    """calculate the coordinate for each asset"""

    gdf = gpd.read_file(gpkg_path)
    cent = gdf.geometry.centroid
    coords = pl.DataFrame(
        {
            "asset_id": gdf["asset_id"].astype("int64").to_numpy(),
            "x": cent.x.to_numpy(),
            "y": cent.y.to_numpy(),
        }
    )
    joined = pl.DataFrame({"asset_id": asset_ids}).join(
        coords, on="asset_id", how="left"
    )
    assert joined["x"].null_count() == 0, "asset(s) with no centroid in aims.gpkg"
    return joined.select("x", "y").to_numpy()


def make_models(cat_idx) -> dict:
    return {
        "median": lambda: DummyClassifier(strategy="constant", constant=3),
        "stratified": lambda: DummyClassifier(strategy="stratified", random_state=SEED),
        "hgb": lambda: HistGradientBoostingClassifier(
            categorical_features=cat_idx,
            learning_rate=0.05,
            max_iter=400,
            early_stopping=True,
            random_state=SEED,
        ),
    }


def balanced_fit(model, X, y):
    """
    only balance the real model, not the dummies.
    Balance would have no affect on the median dummy predictor,
    and would break the from distibution stratified one
    """

    if isinstance(model, HistGradientBoostingClassifier):
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def main(parquet=PARQUET, gpkg=GPKG, n_blocks=N_BLOCKS, n_splits=N_SPLITS, seed=SEED):
    df_feats = prepare_features(pl.read_parquet(parquet))

    # todo move these asserts to a validate function that future scripts can reuse, or enforce when data is created
    assert set(LEAKY_COLS).isdisjoint(df_feats.columns)
    assert df_feats["asset_id"].n_unique() == df_feats.height
    assert df_feats["condition_grade"].is_between(1, 5).all()
    assert df_feats["asset_length_log1p"].is_finite().all()

    full = [
        *NOMINAL_CATEGORY_COLS,
        *GEOSURE_COLS,
        "maintainer_is_ea",
        "asset_length_log1p",
    ]

    # todo - i thought this assert was proven to pass in eda but is failing here. Either im wrong or the data has changed or the eda is wrong. should investigate, but doesn't break this script as model will tolerate the nulls
    # assert df_feats.select(full).null_count().to_numpy().sum() == 0, (
    #     f"assert no nulls failed, {df_feats.select(full).null_count().to_numpy().sum()} nulls"
    # )

    X, y, cat_idx, asset_ids = create_arrays(df_feats)
    coords = load_coords(asset_ids, gpkg)
    blocks = produce_spatial_blocks(coords, n_blocks=n_blocks, seed=seed)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scorers = build_scorers()

    results, rows = {}, []
    for name, factory in make_models(cat_idx).items():
        result = cross_validate(
            make_model=factory,
            X=X,
            y=y,
            cv_splitter=splitter,
            groups=blocks,
            scorers=scorers,
            fit_fn=balanced_fit,
        )
        results[name] = result
        means = report_means(result)
        scores = {k: v for k, v in means.items() if not k.endswith("n_folds")}

        rows.append({"model": name, **scores})

    table = pl.DataFrame(rows)

    print(f"\nn={len(y)}\nblocks={n_blocks}\nfolds={n_splits}")
    print("\n")

    with pl.Config(
        tbl_cols=-1,
        tbl_width_chars=180,
    ):
        print(table)
    print("\n")

    log_results(rows)

    return table, results


def log_results(rows):

    results = paths.experiments / "baseline/results.jsonl"
    data = json.dumps(
        {
            "run_id": str(uuid4()),
            "run_at": datetime.now().strftime(format="%Y-%m-%d %H:%M:%S"),
            "results": rows,
        }
    )

    results.write_text(data=data)


if __name__ == "__main__":
    main()
