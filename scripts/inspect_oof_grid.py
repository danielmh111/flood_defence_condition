# exercises the whole src/reporting.py surface against the real oof_grid artefacts. this is
# proof the stored OOF frame supports every spec §9 metric - it is not the analysis (that is
# a future notebook's job; out of scope for this session, see experiment_design.md §9).

import polars as pl
from project_paths import paths

from experiments.oof_grid import reporting as R

OUT = paths.experiments / "oof_grid"


def main():
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_cols(15)
    pl.Config.set_tbl_width_chars(200)

    oof = R.load_oof(OUT / "oof_predictions.parquet")
    print(
        f"load_oof: {oof.height} rows, {oof['cell_id'].n_unique()} cells - schema ok\n"
    )

    per_fold = R.metrics_per_fold(oof)
    summary = R.metrics_summary(per_fold)
    print("=== metrics_summary ===")
    print(summary)

    hgb_base = summary.filter(
        (pl.col("cell_id") == "hgb_default__base")
        & (pl.col("metric").is_in(["qwk", "pr_auc_ge4"]))
    )
    print("\nsanity check vs baseline_2 (QWK ~ 0.26, PR-AUC(>=4) ~ 0.17):")
    print(hgb_base)

    floor = summary.filter(
        (pl.col("cell_id") == "dummy_stratified__base")
        & (pl.col("metric").is_in(["qwk", "pr_auc_ge4"]))
    )
    print(
        "\nsanity check floor (dummy_stratified: QWK ~ 0, PR-AUC(>=4) ~ prevalence 0.096):"
    )
    print(floor)

    print("\n=== per_grade_metrics (hgb_default__base) ===")
    pgm = R.per_grade_metrics(oof)
    print(pgm.filter(pl.col("cell_id") == "hgb_default__base"))

    print("\n=== confusion_wide (hgb_default__base) ===")
    conf = R.confusion(oof)
    print(R.confusion_wide(conf).filter(pl.col("cell_id") == "hgb_default__base"))

    print("\n=== paired_deltas (default_contrasts) ===")
    contrasts = R.default_contrasts(oof["cell_id"].unique().to_list())
    deltas = R.paired_deltas(per_fold, contrasts)
    print(deltas)

    print("\n=== recommend_operating_point (max_f1) ===")
    sweep = R.threshold_sweep(oof)
    print(R.recommend_operating_point(sweep, rule="max_f1"))

    print("\ninspect_oof_grid: all reporting.py functions ran without error")


if __name__ == "__main__":
    main()
