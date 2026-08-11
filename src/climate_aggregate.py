"""Stage 2 - aggregate the reduced climate metrics (src/climate_extract.py) into
asset_id-keyed windowed features. The exposure_aggregate.py analogue; design doc §6-§7.

Key divergence from exposure (deliberate, §6): accumulation is *clipped* to the
observable span [observable_floor, t_eff), not left as a raw window count, so `count`
and `rate` stay mutually consistent for a dense (climate) rather than sparse
(flood-event) field.

Two granularities, both reduced with the same masked dense-grid trick (`_masked_reduce`):
- additive metrics (ftc, fdd, drydays, r10mm, r20mm): `count`/`rate` at MONTH granularity
  (>=1mo edge error), `max_in_year`/`z` at YEAR granularity (fully-observable calendar
  years only, to avoid a partial-year understatement artefact - a refinement on top of
  the design doc, agreed in the planning session).
- annual run-length metrics (dryspell_*, rain_max_day, rewet_15): {mean, max} or
  {sum, mean, max} at YEAR granularity, same fully-observable-year definition.
"""

import calendar
import json
from datetime import date, timedelta

import numpy as np
import polars as pl
import polars.selectors as cs
from project_paths import paths

from src.climate_extract import (
    ADDITIVE_METRICS_RAIN,
    ADDITIVE_METRICS_TEMP,
    ANNUAL_RAIN_PATH,
    MONTHLY_RAIN_PATH,
    MONTHLY_TEMP_PATH,
    NORMALS_PATH,
)
from src.climate_fetch import CELLS_PATH, FETCH_START, RESOLUTION_M
from src.exposure_aggregate import load_anchors

ADDITIVE_METRICS = ADDITIVE_METRICS_TEMP + ADDITIVE_METRICS_RAIN
ANNUAL_EXTREME_METRICS = ["dryspell_max", "rain_max_day"]
ANNUAL_COUNT_METRICS = ["dryspell_count_10", "dryspell_count_15", "dryspell_count_21", "rewet_15"]
ANNUAL_METRICS = ANNUAL_EXTREME_METRICS + ANNUAL_COUNT_METRICS

METRIC_GROUP = {
    **{m: "temp" for m in ADDITIVE_METRICS_TEMP},
    **{m: "rain" for m in ADDITIVE_METRICS_RAIN + ANNUAL_METRICS},
}

TRUE_RECORD_FLOOR = {"temp": date(1931, 1, 1), "rain": date(1891, 1, 1)}
RECORD_FLOOR = {g: max(f, FETCH_START) for g, f in TRUE_RECORD_FLOOR.items()}

# the fetch floor (1991) dominates both true record floors today, so a single
# group-agnostic observable_floor/window_truncated is correct (see with_window_cols).
# fails loud rather than silently, if a future FETCH_START change invalidates this.
if len(set(RECORD_FLOOR.values())) != 1:
    raise RuntimeError(
        "per-group record floors now differ - with_window_cols assumes a single "
        "effective floor and must be made group-aware"
    )
EFFECTIVE_RECORD_FLOOR = next(iter(set(RECORD_FLOOR.values())))

WINDOWS: list[timedelta] = [timedelta(days=365.25 * w) for w in (2, 5, 15, 30)]

OUT_PATH = paths.processed_data / "climate_features.parquet"
MANIFEST_PATH = paths.processed_data / "climate_manifest.json"


class AggregateError(Exception): ...


def window_label(window: timedelta) -> str:
    return f"{round(window.days / 365.25)}y"


# --- month/year ordinal bounds (the "fully-included" edge rule) ---
#
# a month [start, end] is included iff start >= observable_floor and end < t_eff.
# the month containing t_eff always fails end < t_eff (t_eff falls inside it), so the
# last included month is always ordinal(t_eff) - 1, and symmetrically for years.
# the first included month is ordinal(observable_floor) itself only if observable_floor
# is exactly a month start, else the next month. same pattern for years/Jan-1.


def month_ordinal(year: np.ndarray, month: np.ndarray) -> np.ndarray:
    return year * 12 + (month - 1)


def _nat_mask(*arrs: np.ndarray) -> np.ndarray:
    out = np.isnat(arrs[0])
    for a in arrs[1:]:
        out = out | np.isnat(a)
    return out


def _month_bounds(observable_floor: np.ndarray, t_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """first/last fully-included month ordinal per row.

    NaT does NOT propagate to NaN through a plain datetime64->int64->float cast (NaT's
    int64 sentinel becomes a huge finite float, not NaN) - unassessable rows are masked
    explicitly here rather than relied on to cancel out downstream.
    """
    nat = _nat_mask(observable_floor, t_eff)

    of_month_start = observable_floor.astype("datetime64[M]")
    of_year = of_month_start.astype("datetime64[Y]").astype(float) + 1970
    of_month = (of_month_start.astype("datetime64[M]").astype("int64") % 12) + 1
    of_is_month_start = observable_floor.astype("datetime64[D]") == of_month_start.astype(
        "datetime64[D]"
    )
    first_ord = of_year * 12 + (of_month - 1) + (~of_is_month_start).astype(float)

    te_month_start = t_eff.astype("datetime64[M]")
    te_year = te_month_start.astype("datetime64[Y]").astype(float) + 1970
    te_month = (te_month_start.astype("datetime64[M]").astype("int64") % 12) + 1
    last_ord = te_year * 12 + (te_month - 1) - 1

    first_ord[nat] = np.nan
    last_ord[nat] = np.nan
    return first_ord, last_ord


def _year_bounds(observable_floor: np.ndarray, t_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nat = _nat_mask(observable_floor, t_eff)

    of_year_start = observable_floor.astype("datetime64[Y]")
    of_year = of_year_start.astype(float) + 1970
    of_is_year_start = observable_floor.astype("datetime64[D]") == of_year_start.astype(
        "datetime64[D]"
    )
    first_year = of_year + (~of_is_year_start).astype(float)

    te_year_start = t_eff.astype("datetime64[Y]")
    te_year = te_year_start.astype(float) + 1970
    last_year = te_year - 1

    first_year[nat] = np.nan
    last_year[nat] = np.nan
    return first_year, last_year


# --- dense (cell x index) grids + masked range-reduce ---


def dense_grid(
    df: pl.DataFrame, index_col: str, value_col: str, index_min: int, index_max: int
) -> tuple[np.ndarray, dict[int, int]]:
    """(n_cells, n_idx) dense array from a long (cell_id, index_col, value_col) frame.

    NaN where a cell has no row for that index. cell_pos maps cell_id -> row position.
    """
    cells = sorted(df["cell_id"].unique().to_list())
    cell_pos = {c: i for i, c in enumerate(cells)}
    n_idx = index_max - index_min + 1
    grid = np.full((len(cells), n_idx), np.nan)

    cell_arr = df["cell_id"].to_numpy()
    idx_arr = df[index_col].to_numpy() - index_min
    val_arr = df[value_col].to_numpy().astype(np.float64)
    in_range = (idx_arr >= 0) & (idx_arr < n_idx)
    row_pos = np.array([cell_pos[c] for c in cell_arr[in_range]], dtype=np.int64)
    grid[row_pos, idx_arr[in_range]] = val_arr[in_range]
    return grid, cell_pos


def _masked_reduce(
    cell_id: np.ndarray,
    first_idx: np.ndarray,
    last_idx: np.ndarray,
    index_min: int,
    grid: np.ndarray,
    cell_pos: dict[int, int],
    reduction: str,
) -> np.ndarray:
    """per-row reduction of `grid` over columns in [first_idx, last_idx] inclusive.

    `first_idx`/`last_idx` are ABSOLUTE ordinals (month or year, same convention as
    `_month_bounds`/`_year_bounds`); `index_min` converts them to `grid`'s own 0-based
    column index (same convention as `dense_grid`'s `index_min`) - do not pass already-
    relative indices here.

    NaN first_idx/last_idx (unassessable rows) and empty ranges both correctly yield
    NaN (never 0) via the `valid.any(axis=1)` guard - "no observable data" != "measured
    zero". `count`/`sum` reductions still return 0 for a real, non-empty, all-zero span.
    """
    if cell_id.dtype != object:
        pos = np.fromiter((cell_pos[int(c)] for c in cell_id), dtype=np.int64, count=len(cell_id))
    else:
        pos = np.fromiter((cell_pos[c] for c in cell_id), dtype=np.int64, count=len(cell_id))

    per_row = grid[pos]  # (n_rows, n_idx)
    n_idx = grid.shape[1]
    first_rel = first_idx - index_min
    last_rel = last_idx - index_min
    idx_grid = np.arange(n_idx)
    mask = (idx_grid[None, :] >= first_rel[:, None]) & (idx_grid[None, :] <= last_rel[:, None])
    valid = mask & ~np.isnan(per_row)
    any_valid = valid.any(axis=1)

    if reduction == "max":
        out = np.where(valid, per_row, -np.inf).max(axis=1)
    elif reduction in ("sum", "count"):
        out = np.where(valid, per_row, 0.0).sum(axis=1)
    elif reduction == "mean":
        summed = np.where(valid, per_row, 0.0).sum(axis=1)
        counts = valid.sum(axis=1)
        out = np.divide(summed, counts, out=np.full(len(cell_id), np.nan), where=counts > 0)
    else:
        raise AggregateError(f"unknown reduction {reduction!r}")

    out = out.astype(np.float64)
    if reduction in ("sum", "count"):
        out[~any_valid] = np.nan  # empty range: null, not 0 (distinct from a measured zero)
    else:
        out[~any_valid] = np.nan
    return out


# --- data ceiling, anchor frame ---


def data_ceiling(monthly_temp: pl.DataFrame, monthly_rain: pl.DataFrame) -> date:
    """last calendar day with data common to both variable groups."""

    def last_month(df: pl.DataFrame) -> date:
        row = df.sort("year", "month").tail(1)
        y, m = int(row["year"][0]), int(row["month"][0])
        return date(y, m, calendar.monthrange(y, m)[1])

    return min(last_month(monthly_temp), last_month(monthly_rain))


def build_anchor_frame(anchors: pl.DataFrame, ceiling: date) -> pl.DataFrame:
    cells = pl.read_parquet(CELLS_PATH)
    out = anchors.join(cells, on="asset_id", how="left")

    unassessable = pl.col("t_anchor").is_null()
    t_eff = pl.when(unassessable).then(None).otherwise(
        pl.min_horizontal(pl.col("t_anchor"), pl.lit(ceiling))
    )
    out = out.with_columns(t_eff.alias("t_eff")).with_columns(
        (pl.col("t_anchor") - pl.col("t_eff")).dt.total_days().cast(pl.Int32).alias(
            "climate__ceiling_lag_days"
        ),
        (pl.col("fallback_temp") | pl.col("fallback_rain")).alias("climate__oncoast_fallback"),
    )
    return out


def with_window_cols(df: pl.DataFrame, window: timedelta) -> pl.DataFrame:
    label = window_label(window)
    unassessable = pl.col("t_eff").is_null()
    window_start = pl.col("t_eff") - pl.duration(days=window.days)

    observable_floor = pl.when(unassessable).then(None).otherwise(
        pl.max_horizontal(
            window_start, pl.col("asset_start_date"), pl.lit(EFFECTIVE_RECORD_FLOOR)
        )
    )
    df = df.with_columns(observable_floor.alias(f"_obs_floor_{label}"))

    observable_years = pl.when(unassessable).then(None).otherwise(
        (pl.col("t_eff") - pl.col(f"_obs_floor_{label}")).dt.total_days() / 365.25
    )
    df = df.with_columns(observable_years.alias(f"_obs_years_{label}"))

    truncated = pl.when(unassessable).then(None).otherwise(
        pl.col(f"_obs_years_{label}") < (window.days / 365.25)
    )
    return df.with_columns(truncated.alias(f"climate__{label}__window_truncated"))


# --- assembling the additive + annual stat blocks ---


def build_features(
    anchor_frame: pl.DataFrame,
    monthly_temp: pl.DataFrame,
    monthly_rain: pl.DataFrame,
    annual_rain: pl.DataFrame,
    normals: pl.DataFrame,
    ceiling: date,
) -> pl.DataFrame:
    n = anchor_frame.height
    t_eff = anchor_frame["t_eff"].to_numpy()

    # --- month-level dense grids (additive metrics: count/rate) ---
    month_min = FETCH_START.year * 12 + (FETCH_START.month - 1)
    month_max = ceiling.year * 12 + (ceiling.month - 1)

    # monthly_temp/monthly_rain both use a plain "cell_id" column (their own group's local
    # id space, per climate_extract.py); only the anchor_frame distinguishes cell_id_temp
    # vs cell_id_rain for the join. METRIC_GROUP says which monthly source each metric reads.
    monthly_src = {
        "ftc": monthly_temp,
        "fdd": monthly_temp,
        "drydays": monthly_rain,
        "r10mm": monthly_rain,
        "r20mm": monthly_rain,
    }
    month_grids: dict[str, tuple[np.ndarray, dict[int, int]]] = {}
    for metric, src in monthly_src.items():
        ordinals = src["year"] * 12 + (src["month"] - 1)
        long = src.select(pl.col("cell_id"), ordinals.alias("ordinal"), pl.col(metric))
        month_grids[metric] = dense_grid(long, "ordinal", metric, month_min, month_max)

    # --- year-level dense grids: additive-metric annual sums (max_in_year) ---
    year_min, year_max = FETCH_START.year, ceiling.year
    year_grids_additive: dict[str, tuple[np.ndarray, dict[int, int]]] = {}
    for metric, src in monthly_src.items():
        annual_sum = src.group_by("cell_id", "year").agg(pl.col(metric).sum().alias(metric))
        year_grids_additive[metric] = dense_grid(annual_sum, "year", metric, year_min, year_max)

    # --- year-level dense grids: run-length annual metrics (already per-year) ---
    year_grids_annual: dict[str, tuple[np.ndarray, dict[int, int]]] = {}
    for metric in ANNUAL_METRICS:
        long = annual_rain.select(pl.col("cell_id"), pl.col("year"), pl.col(metric))
        year_grids_annual[metric] = dense_grid(long, "year", metric, year_min, year_max)

    # --- normals lookup: (group, metric) -> {cell_id: (mu, sd)} ---
    normal_lookup: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}
    for (group, metric), grp in normals.group_by(["group", "metric"]):
        normal_lookup[(group, metric)] = {
            int(r["cell_id"]): (r["mu"], r["sd"]) for r in grp.iter_rows(named=True)
        }

    cell_id_temp = anchor_frame["cell_id_temp"].to_numpy()
    cell_id_rain = anchor_frame["cell_id_rain"].to_numpy()
    cell_id_of = {"temp": cell_id_temp, "rain": cell_id_rain}

    out_cols: dict[str, np.ndarray] = {}

    for window in WINDOWS:
        label = window_label(window)
        obs_floor = anchor_frame[f"_obs_floor_{label}"].to_numpy()
        obs_years = anchor_frame[f"_obs_years_{label}"].to_numpy()

        first_month, last_month = _month_bounds(obs_floor, t_eff)
        first_year, last_year = _year_bounds(obs_floor, t_eff)

        for metric in ADDITIVE_METRICS:
            group = METRIC_GROUP[metric]
            cell_id = cell_id_of[group]

            grid_m, pos_m = month_grids[metric]
            count = _masked_reduce(
                cell_id, first_month, last_month, month_min, grid_m, pos_m, "count"
            )
            rate = np.where(obs_years > 0, count / np.where(obs_years > 0, obs_years, 1), np.nan)

            grid_y, pos_y = year_grids_additive[metric]
            max_in_year = _masked_reduce(
                cell_id, first_year, last_year, year_min, grid_y, pos_y, "max"
            )

            mu = np.full(n, np.nan)
            sd = np.full(n, np.nan)
            lut = normal_lookup.get((group, metric), {})
            for i, c in enumerate(cell_id):
                if int(c) in lut:
                    mu[i], sd[i] = lut[int(c)]
            z = np.where(sd > 0, (rate - mu) / np.where(sd > 0, sd, 1), np.nan)

            out_cols[f"climate__{metric}__{label}__count"] = count
            out_cols[f"climate__{metric}__{label}__rate"] = rate
            out_cols[f"climate__{metric}__{label}__max_in_year"] = max_in_year
            out_cols[f"climate__{metric}__{label}__z"] = z

        for metric in ANNUAL_EXTREME_METRICS:
            cell_id = cell_id_of[METRIC_GROUP[metric]]
            grid_y, pos_y = year_grids_annual[metric]
            out_cols[f"climate__{metric}__{label}__mean"] = _masked_reduce(
                cell_id, first_year, last_year, year_min, grid_y, pos_y, "mean"
            )
            out_cols[f"climate__{metric}__{label}__max"] = _masked_reduce(
                cell_id, first_year, last_year, year_min, grid_y, pos_y, "max"
            )

        for metric in ANNUAL_COUNT_METRICS:
            cell_id = cell_id_of[METRIC_GROUP[metric]]
            grid_y, pos_y = year_grids_annual[metric]
            out_cols[f"climate__{metric}__{label}__sum"] = _masked_reduce(
                cell_id, first_year, last_year, year_min, grid_y, pos_y, "sum"
            )
            out_cols[f"climate__{metric}__{label}__mean"] = _masked_reduce(
                cell_id, first_year, last_year, year_min, grid_y, pos_y, "mean"
            )
            out_cols[f"climate__{metric}__{label}__max"] = _masked_reduce(
                cell_id, first_year, last_year, year_min, grid_y, pos_y, "max"
            )

    # numpy NaN (from _masked_reduce/rate/z) is a *value*, not a Polars null - pl.DataFrame
    # would otherwise store it as a present-but-NaN float, breaking drop_nulls()-based
    # asserts and the "never fill with 0, preserve nulls" contract. Convert once, here.
    features = pl.DataFrame(out_cols).with_columns(cs.float().fill_nan(None))
    features = features.with_columns(
        anchor_frame["asset_id"],
        *(anchor_frame[f"climate__{window_label(w)}__window_truncated"] for w in WINDOWS),
        anchor_frame["climate__oncoast_fallback"],
        anchor_frame["climate__ceiling_lag_days"],
    )
    return features.select("asset_id", *[c for c in features.columns if c != "asset_id"])


def main():
    anchors = load_anchors()
    monthly_temp = pl.read_parquet(MONTHLY_TEMP_PATH)
    monthly_rain = pl.read_parquet(MONTHLY_RAIN_PATH)
    annual_rain = pl.read_parquet(ANNUAL_RAIN_PATH)
    normals = pl.read_parquet(NORMALS_PATH)

    ceiling = data_ceiling(monthly_temp, monthly_rain)
    anchor_frame = build_anchor_frame(anchors, ceiling)
    for window in WINDOWS:
        anchor_frame = with_window_cols(anchor_frame, window)

    features = build_features(
        anchor_frame, monthly_temp, monthly_rain, annual_rain, normals, ceiling
    )

    # --- assert gates (C5-C6; see plan) ---
    assert features["asset_id"].sort().equals(anchors["asset_id"].sort()), "asset_id mismatch"

    count_cols = [c for c in features.columns if c.endswith("__count")]
    for col in count_cols:
        vals = features[col].drop_nulls()
        assert (vals >= 0).all(), f"{col}: negative count"
        assert vals.is_finite().all(), f"{col}: non-finite count"

    for w_small, w_large in zip(WINDOWS, WINDOWS[1:]):
        for metric in ADDITIVE_METRICS:
            small = features[f"climate__{metric}__{window_label(w_small)}__count"]
            large = features[f"climate__{metric}__{window_label(w_large)}__count"]
            both = small.is_not_null() & large.is_not_null()
            assert (large.filter(both) >= small.filter(both)).all(), (
                f"{metric}: count not monotone increasing {window_label(w_small)} -> "
                f"{window_label(w_large)}"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(OUT_PATH)

    manifest = {
        "resolution_m": RESOLUTION_M,
        "fetch_start": FETCH_START.isoformat(),
        "data_ceiling": ceiling.isoformat(),
        "n_assets": features.height,
        "n_columns": features.width - 1,
        "oncoast_fallback_rate": float(features["climate__oncoast_fallback"].mean()),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print(f"wrote {features.height} assets x {features.width - 1} feature columns to {OUT_PATH}")


if __name__ == "__main__":
    main()
