"""Stage 1 - reduce the per-cell daily cache (src/climate_fetch.py) to monthly-additive
and annual run-length metrics. Pure Polars over the cache; cheap to re-run when a metric
definition changes, which is the point of persisting the daily series (design doc §8).

ftc/fdd live in "temp" cell-id space (tasmin+tasmax); drydays/r10mm/r20mm/dryspell*/
rewet_15/rain_max_day live in "rain" cell-id space (rainfall). METRIC_GROUP records which,
since climate_aggregate.py must join each metric's normals/monthly frame on the matching
cell_id_{group} column from cells_{res}m.parquet.
"""

import calendar
from datetime import date

import numpy as np
import polars as pl
from project_paths import paths

from src.climate_fetch import CACHE_DIR, RESOLUTION_M

NORMAL_START, NORMAL_END = 1991, 2020  # WMO standard period; inside the FETCH_START floor

ADDITIVE_METRICS_TEMP = ["ftc", "fdd"]
ADDITIVE_METRICS_RAIN = ["drydays", "r10mm", "r20mm"]
ANNUAL_METRICS_RAIN = [
    "dryspell_max",
    "dryspell_count_10",
    "dryspell_count_15",
    "dryspell_count_21",
    "rain_max_day",
    "rewet_15",
]

METRIC_GROUP = {
    **{m: "temp" for m in ADDITIVE_METRICS_TEMP},
    **{m: "rain" for m in ADDITIVE_METRICS_RAIN},
    **{m: "rain" for m in ANNUAL_METRICS_RAIN},
}

MONTHLY_TEMP_PATH = paths.processed_data / f"climate_cell_monthly_temp_{RESOLUTION_M}m.parquet"
MONTHLY_RAIN_PATH = paths.processed_data / f"climate_cell_monthly_rain_{RESOLUTION_M}m.parquet"
ANNUAL_RAIN_PATH = paths.processed_data / f"climate_cell_annual_rain_{RESOLUTION_M}m.parquet"
NORMALS_PATH = paths.processed_data / f"climate_cell_normals_{RESOLUTION_M}m.parquet"


class ExtractError(Exception): ...


def load_daily(var: str) -> pl.DataFrame:
    """cell_id, date, value - concatenated from consolidated years + any un-consolidated monthly shards."""
    var_dir = CACHE_DIR / var
    frames = [pl.read_parquet(p) for p in sorted(var_dir.glob("*.parquet"))]
    if not frames:
        raise ExtractError(f"no cached daily data for {var} in {var_dir} - run climate_fetch first")
    df = pl.concat(frames)
    dup = df.select("cell_id", "date").is_duplicated().sum()
    if dup:
        raise ExtractError(f"{var}: {dup} duplicate (cell_id, date) rows - monthly/yearly overlap?")
    return df.sort("cell_id", "date")


def validate_daily_cache(var: str) -> None:
    """C3: per-(cell, year) day counts match the calendar, except the in-progress current year."""
    df = load_daily(var)
    n_cells = df["cell_id"].n_unique()
    today = date.today()
    per_year = df.with_columns(pl.col("date").dt.year().alias("year")).group_by("year").agg(
        pl.len().alias("n")
    )
    for year, n in per_year.iter_rows():
        if year == today.year:
            continue  # partial year, short count expected
        expected = n_cells * (366 if calendar.isleap(year) else 365)
        if n != expected:
            raise ExtractError(f"{var} {year}: expected {expected} rows ({n_cells} cells), got {n}")


# --- monthly-additive metrics ---


def temp_monthly() -> pl.DataFrame:
    """cell_id (temp space), year, month -> ftc, fdd."""
    tasmin = load_daily("tasmin").rename({"value": "tasmin"})
    tasmax = load_daily("tasmax").rename({"value": "tasmax"})
    joined = tasmin.join(tasmax, on=["cell_id", "date"], how="inner")
    if joined.height != tasmin.height or joined.height != tasmax.height:
        raise ExtractError("tasmin/tasmax daily series misaligned - missing dates on one side")

    return (
        joined.with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            ((pl.col("tasmin") < 0) & (pl.col("tasmax") > 0)).alias("_ftc_day"),
            (-(pl.col("tasmin") + pl.col("tasmax")) / 2).clip(lower_bound=0).alias("_fdd_day"),
        )
        .group_by("cell_id", "year", "month")
        .agg(
            pl.col("_ftc_day").sum().cast(pl.Int32).alias("ftc"),
            pl.col("_fdd_day").sum().alias("fdd"),
        )
        .sort("cell_id", "year", "month")
    )


def rain_monthly() -> pl.DataFrame:
    """cell_id (rain space), year, month -> drydays, r10mm, r20mm."""
    rain = load_daily("rainfall")
    return (
        rain.with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        )
        .group_by("cell_id", "year", "month")
        .agg(
            (pl.col("value") < 1.0).sum().cast(pl.Int32).alias("drydays"),
            (pl.col("value") >= 10.0).sum().cast(pl.Int32).alias("r10mm"),
            (pl.col("value") >= 20.0).sum().cast(pl.Int32).alias("r20mm"),
        )
        .sort("cell_id", "year", "month")
    )


# --- annual run-length metrics (rainfall only) ---


def _find_spells(is_dry: np.ndarray) -> list[tuple[int, int]]:
    """(start_idx, length) for each maximal True-run in a contiguous daily `is_dry` array."""
    if len(is_dry) == 0:
        return []
    change = np.diff(is_dry.astype(np.int8))
    starts = np.flatnonzero(change == 1) + 1
    if is_dry[0]:
        starts = np.concatenate(([0], starts))
    ends = np.flatnonzero(change == -1) + 1
    if is_dry[-1]:
        ends = np.concatenate((ends, [len(is_dry)]))
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def rain_annual() -> pl.DataFrame:
    """cell_id (rain space), year -> dryspell_max, dryspell_count_{10,15,21}, rain_max_day, rewet_15.

    each spell is assigned to the year it *starts* - correct treatment of a spell straddling
    31 Dec, only possible because the daily cache lets us run rle() over the whole per-cell
    series rather than truncating to calendar years first.
    """
    rain = load_daily("rainfall")
    rows: list[dict] = []

    for grp in rain.sort("date").partition_by("cell_id", maintain_order=True):
        cell_id = grp["cell_id"][0]
        dates = grp["date"].to_numpy()
        values = grp["value"].to_numpy()

        gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
        if len(gaps) and not np.all(gaps == 1):
            raise ExtractError(f"cell {cell_id}: gap in daily rainfall series")

        is_dry = values < 1.0
        spells = _find_spells(is_dry)
        spell_years = np.array(
            [dates[s].astype("datetime64[Y]").astype(int) + 1970 for s, _ in spells]
        )
        years = dates.astype("datetime64[Y]").astype(int) + 1970

        for year in np.unique(years):
            in_year = spell_years == year
            lengths = np.array([l for (_, l), keep in zip(spells, in_year) if keep])
            rewet = 0
            for (s, l), keep in zip(spells, in_year):
                if not keep or l < 15:
                    continue
                end = s + l
                if values[end : min(end + 3, len(values))].sum() >= 10.0:
                    rewet += 1
            rows.append(
                {
                    "cell_id": int(cell_id),
                    "year": int(year),
                    "dryspell_max": int(lengths.max()) if lengths.size else 0,
                    "dryspell_count_10": int((lengths >= 10).sum()),
                    "dryspell_count_15": int((lengths >= 15).sum()),
                    "dryspell_count_21": int((lengths >= 21).sum()),
                    "rain_max_day": float(values[years == year].max()),
                    "rewet_15": rewet,
                }
            )

    return pl.DataFrame(rows).sort("cell_id", "year")


# --- 1991-2020 normals for additive metrics ---


def compute_normals(monthly: pl.DataFrame, metrics: list[str]) -> pl.DataFrame:
    """cell_id, metric, mu, sd of the annual total over NORMAL_START-NORMAL_END."""
    annual = (
        monthly.filter(pl.col("year").is_between(NORMAL_START, NORMAL_END))
        .group_by("cell_id", "year")
        .agg(*(pl.col(m).sum().cast(pl.Float64).alias(m) for m in metrics))
    )
    frames = [
        annual.group_by("cell_id")
        .agg(pl.col(m).mean().alias("mu"), pl.col(m).std().alias("sd"))
        .with_columns(pl.lit(m).alias("metric"))
        for m in metrics
    ]
    return pl.concat(frames).select("cell_id", "metric", "mu", "sd")


def main():
    for var in ("tasmin", "tasmax", "rainfall"):
        validate_daily_cache(var)

    tm = temp_monthly()
    rm = rain_monthly()
    ra = rain_annual()

    normals = pl.concat(
        [
            compute_normals(tm, ADDITIVE_METRICS_TEMP).with_columns(pl.lit("temp").alias("group")),
            compute_normals(rm, ADDITIVE_METRICS_RAIN).with_columns(pl.lit("rain").alias("group")),
        ]
    )

    tm.write_parquet(MONTHLY_TEMP_PATH)
    rm.write_parquet(MONTHLY_RAIN_PATH)
    ra.write_parquet(ANNUAL_RAIN_PATH)
    normals.write_parquet(NORMALS_PATH)

    print(
        f"wrote monthly temp {tm.shape}, monthly rain {rm.shape}, "
        f"annual rain {ra.shape}, normals {normals.shape}"
    )


if __name__ == "__main__":
    main()
