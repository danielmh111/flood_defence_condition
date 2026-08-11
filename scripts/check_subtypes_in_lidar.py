import polars as pl
from project_paths import paths

subtypes = pl.read_parquet(paths.processed_data / "lidar_stations.parquet")
print(f"{subtypes.height} stations, {subtypes['asset_id'].n_unique()} assets")

# does prominence separate pinned shoulders from real crests?

per_asset = (
    subtypes.group_by("asset_id", "tier")
    .agg(
        pl.col("crest_off").abs().median().alias("offset_median"),
        pl.col("prominence").median().alias("prom_med"),
        pl.col("relheight").median().alias("relheight_med"),
        pl.len().alias("n_stations"),
    )
    .with_columns((pl.col("offset_median") >= 9).alias("pinned"))
    .group_by("tier", "pinned")
    .agg(
        pl.len().alias("n"),
        pl.col("prom_med").median().round(2).alias("prom_p50"),
        pl.col("prom_med").quantile(0.25).round(2).alias("prom_p25"),
        pl.col("prom_med").quantile(0.75).round(2).alias("prom_p75"),
        (pl.col("prom_med") <= 0).mean().round(2).alias("frac_nonpeak"),
    )
    .sort("tier", "pinned")
)

with pl.Config(tbl_cols=-1, tbl_rows=-1):
    print(per_asset)
