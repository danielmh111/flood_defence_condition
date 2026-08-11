# materialises the nb08 halcrow heuristic as an artefact so the coarse grid can score it on the
# same folds and metrics as every fitted cell. nb08 computed its predictions inline and never
# persisted them.
#
# run: uv run python -m scripts.build_heuristic_national

import polars as pl
from project_paths import paths

from src.heuristic import DEFAULT_RATE, DEFAULT_REGIME, score

PARQUET = paths.processed_data / "unified_aims_eir_bgs.parquet"
OUT = paths.processed_data / "heuristic_national.parquet"


def main(parquet=PARQUET, out=OUT, rate=DEFAULT_RATE, regime=DEFAULT_REGIME):
    scored = score(pl.read_parquet(parquet), rate=rate, regime=regime)

    if scored["asset_id"].n_unique() != scored.height:
        raise ValueError("asset_id is not unique in the heuristic output")

    coverage = (
        scored.group_by("sub_type", "curve_status")
        .agg(pl.len().alias("n"))
        .sort("sub_type", "n", descending=[False, True])
    )
    status = (
        scored.group_by("curve_status")
        .agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / scored.height).round(4).alias("pct"))
        .sort("n", descending=True)
    )

    with pl.Config(tbl_rows=60):
        print(f"rate={rate}, regime={regime}, {scored.height} assets\n")
        print(status)
        print("\nby sub type:")
        print(coverage)
        print("\npredicted grade distribution (scored only):")
        print(
            scored.filter(pl.col("curve_status").eq("scored"))
            .group_by("pred_grade")
            .agg(pl.len().alias("n"))
            .sort("pred_grade")
        )

    scored.select("asset_id", "pred_grade").write_parquet(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
