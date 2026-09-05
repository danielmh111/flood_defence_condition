import polars as pl
from project_paths import paths
from src import heuristic

df = pl.read_parquet(paths.processed_data / "unified_aims_eir_bgs.parquet")
print(f"height: {df.height}")

h = heuristic.score(df)

coverage = h.group_by("curve_status").len().sort("len", descending=True)
print(coverage)

by_type = (
    h.group_by("sub_type", "curve_status")
    .len()
    .pivot(on="curve_status", index="sub_type", values="len")
    .fill_null(0)
)

print(by_type)

df = df.filter(
    pl.col("eir__condition_grade").ne(0)
    & pl.col("aims__asset_sub_type").ne("Natural High Ground")
)
assert df.height == 23304

status = heuristic.score(df).group_by("curve_status").len().sort("len", descending=True)

print(status)

# by_type.write_parquet(paths.exports / "nb_22_heuristic_coverage.parquet")
