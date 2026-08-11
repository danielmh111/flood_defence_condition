"""Stage 0 - HadUK-Grid fetch and per-cell daily cache.

Design: climate_weathering_feature_extraction.md. Session decisions (2026-07-28):
RESOLUTION_M is a global switch (5km pilot -> 1km production), fetch floor 1991-01-01
(also the WMO 1991-2020 normal period), daily per-cell series persisted so metric
definitions in climate_extract.py are free to iterate without re-downloading.

Only ~5,716 unique 1km cells (1,593 at 5km) cover the scoped assets, so the expensive
step is the download, not the storage - stream one national NetCDF at a time, sample
the scoped cells, discard the file.

Requires a CEDA account + access token in .env as CEDA_TOKEN for months <= DEFINITIVE_END
(register at https://services.ceda.ac.uk/). Months after DEFINITIVE_END are served from
the Met Office provisional stream, which needs no auth but has weaker QC and extra
compression.
"""

import calendar
import os
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import requests
import xarray as xr
from dotenv import load_dotenv
from project_paths import paths

from src.features import prepare_features

load_dotenv()

RESOLUTION_M = 5_000  # flip to 1_000 for production, once the 5km pass clears the notebook audit
RES_LABEL = {1_000: "1km", 5_000: "5km", 12_000: "12km", 25_000: "25km", 60_000: "60km"}[
    RESOLUTION_M
]

TEMP_VARS = ["tasmin", "tasmax"]
RAIN_VARS = ["rainfall"]
ALL_VARS = TEMP_VARS + RAIN_VARS
GROUP_OF = {"tasmin": "temp", "tasmax": "temp", "rainfall": "rain"}
REFERENCE_VAR = {"temp": "tasmin", "rain": "rainfall"}

HADUK_VERSION = "v1.3.2.ceda"
CEDA_BASE = "https://dap.ceda.ac.uk/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid"
PROVISIONAL_BASE = "https://hadleyserver.metoffice.gov.uk/hadobs/hadukgrid/data"

FETCH_START = date(1991, 1, 1)  # 30y cap floor; also the WMO 1991-2020 normal-period start
DEFINITIVE_END = date(2025, 12, 31)  # last month covered by HADUK_VERSION

# BNG / OSGB36 transverse mercator fingerprint (EPSG:27700) - cheaper and more robust
# than parsing a WKT/CRS string out of the grid_mapping variable.
BNG_FALSE_EASTING = 400_000.0
BNG_FALSE_NORTHING = -100_000.0

MAX_FALLBACK_M = 10_000.0  # sea-cell nearest-land search radius
THROTTLE_S = 0.5
MAX_RETRIES = 5
BACKOFF_S = 2.0

OUT_DIR = paths.climate_data
CACHE_DIR = OUT_DIR / "daily" / f"{RESOLUTION_M}m"
CELLS_PATH = OUT_DIR / f"cells_{RESOLUTION_M}m.parquet"
GRID_PATH = OUT_DIR / f"cell_grid_{RESOLUTION_M}m.parquet"
SCRATCH_DIR = OUT_DIR / "_scratch"


class FetchError(Exception): ...


# --- month/file bookkeeping ---


def month_range(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            m, y = 1, y + 1


def file_name(var: str, year: int, month: int) -> str:
    _, last_day = calendar.monthrange(year, month)
    return f"{var}_hadukgrid_uk_{RES_LABEL}_day_{year}{month:02d}01-{year}{month:02d}{last_day:02d}.nc"


_VERSION_DIR_RE = re.compile(r'href="(v\d{8})/?"')
_version_dir_cache: dict[str, str] = {}


def _resolve_version_dir(var: str, headers: dict) -> str:
    """CEDA dropped the `latest/` symlink for HADUK_VERSION - day/ now holds a single
    dated `vYYYYMMDD/` folder instead. Resolve it once per var and cache it."""
    if var in _version_dir_cache:
        return _version_dir_cache[var]
    listing_url = f"{CEDA_BASE}/{HADUK_VERSION}/{RES_LABEL}/{var}/day/"
    r = requests.get(listing_url, headers=headers, timeout=60)
    if r.status_code != 200:
        raise FetchError(f"{r.status_code} listing {listing_url}\n{r.text[:500]}")
    versions = sorted(set(_VERSION_DIR_RE.findall(r.text)))
    if not versions:
        raise FetchError(f"no vYYYYMMDD/ folder found under {listing_url}")
    _version_dir_cache[var] = versions[-1]
    return versions[-1]


def file_url(var: str, year: int, month: int) -> tuple[str, dict]:
    """(url, extra_headers) for one month - definitive (CEDA, auth) or provisional (no auth)."""
    name = file_name(var, year, month)
    if date(year, month, 1) <= DEFINITIVE_END:
        token = os.getenv("CEDA_TOKEN")
        if not token:
            raise FetchError(
                "CEDA_TOKEN not set in .env - register at https://services.ceda.ac.uk/ "
                "and mint an access token before fetching definitive (pre-2026) months"
            )
        headers = {"Authorization": f"Bearer {token}"}
        version_dir = _resolve_version_dir(var, headers)
        url = f"{CEDA_BASE}/{HADUK_VERSION}/{RES_LABEL}/{var}/day/{version_dir}/{name}"
        return url, headers
    return f"{PROVISIONAL_BASE}/{year}/{name}", {}


def _get(url: str, headers: dict) -> bytes:
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers, timeout=180)
            if r.status_code == 200:
                return r.content
            if r.status_code == 401:
                raise FetchError(f"401 on {url} - CEDA_TOKEN missing/expired, refresh it")
            if r.status_code < 500 and r.status_code != 429:
                raise FetchError(f"{r.status_code} on {url}\n{r.text[:500]}")
            last = FetchError(f"{r.status_code} on {url}")
        except requests.RequestException as e:
            last = e
        time.sleep(BACKOFF_S * (2**attempt))
    raise FetchError(f"exhausted {MAX_RETRIES} retries: {last}")


def download(var: str, year: int, month: int, out: Path) -> None:
    url, headers = file_url(var, year, month)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_get(url, headers))


# --- grid / cell lookup ---


@dataclass
class GridInfo:
    x: np.ndarray  # projection_x_coordinate, ascending
    y: np.ndarray  # projection_y_coordinate, ascending
    mask: np.ndarray  # (len(y), len(x)) bool, True = valid (land) cell


def _assert_grid_contract(ds: xr.Dataset, var: str) -> None:
    if "projection_x_coordinate" not in ds.coords or "projection_y_coordinate" not in ds.coords:
        raise FetchError(f"{var}: missing projection_x/y_coordinate dims")

    grid_mapping_name = ds[var].attrs.get("grid_mapping")
    gm = ds[grid_mapping_name] if grid_mapping_name in ds.variables else None
    if gm is None or gm.attrs.get("grid_mapping_name") != "transverse_mercator":
        raise FetchError(f"{var}: expected grid_mapping_name=transverse_mercator")
    if gm.attrs.get("false_easting") != BNG_FALSE_EASTING or gm.attrs.get(
        "false_northing"
    ) != BNG_FALSE_NORTHING:
        raise FetchError(f"{var}: grid_mapping does not fingerprint as EPSG:27700/BNG")

    x = ds["projection_x_coordinate"].to_numpy()
    y = ds["projection_y_coordinate"].to_numpy()
    dx = np.diff(x)
    dy = np.diff(y)
    if not (np.allclose(dx, RESOLUTION_M) and np.allclose(dy, RESOLUTION_M)):
        raise FetchError(
            f"{var}: coordinate spacing != RESOLUTION_M ({RESOLUTION_M}); "
            f"got dx~{dx[0] if len(dx) else None}, dy~{dy[0] if len(dy) else None}"
        )

    units = ds[var].attrs.get("units", "")
    expected = {"tasmin": "degC", "tasmax": "degC", "rainfall": "mm"}[var]
    if units != expected:
        raise FetchError(f"{var}: expected units={expected!r}, got {units!r}")


def load_grid(var: str, year: int, month: int) -> GridInfo:
    """download one reference month for `var` and return its coordinate grid + land mask."""
    ref = SCRATCH_DIR / file_name(var, year, month)
    if not ref.exists():
        download(var, year, month, ref)

    ds = xr.open_dataset(ref)
    _assert_grid_contract(ds, var)
    x = ds["projection_x_coordinate"].to_numpy()
    y = ds["projection_y_coordinate"].to_numpy()
    mask = np.isfinite(ds[var].isel(time=0).to_numpy())
    ds.close()
    return GridInfo(x=x, y=y, mask=mask)


def nearest_index(coords: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """nearest index into an ascending 1-D grid axis for each coordinate."""
    idx = np.searchsorted(axis, coords)
    idx = np.clip(idx, 1, len(axis) - 1)
    left, right = axis[idx - 1], axis[idx]
    return np.where(coords - left <= right - coords, idx - 1, idx)


def _nearest_valid_cell(
    iy: int, ix: int, mask: np.ndarray, max_radius: int
) -> tuple[int, int] | None:
    """expanding-ring search for the nearest True cell in `mask`; None if none within radius."""
    if mask[iy, ix]:
        return iy, ix
    ny, nx = mask.shape
    for r in range(1, max_radius + 1):
        best, best_d2 = None, None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dy), abs(dx)) != r:
                    continue  # only the new ring at this radius
                yy, xx = iy + dy, ix + dx
                if 0 <= yy < ny and 0 <= xx < nx and mask[yy, xx]:
                    d2 = dy * dy + dx * dx
                    if best_d2 is None or d2 < best_d2:
                        best_d2, best = d2, (yy, xx)
        if best is not None:
            return best
    return None


def scoped_asset_centroids() -> gpd.GeoDataFrame:
    """scoped-asset centroids, epsg:27700, matching prepare_features()."""
    scope = prepare_features(pl.read_parquet(paths.unified_file))["asset_id"].to_list()
    a = gpd.read_file(paths.aims_data / "aims.gpkg", columns=["asset_id"])
    a["asset_id"] = a["asset_id"].astype("int64")
    a = a[a["asset_id"].isin(scope)].reset_index(drop=True)
    if a.crs is None or a.crs.to_epsg() != 27700:
        raise FetchError(f"expected epsg:27700, got {a.crs}")
    a["centroid"] = a.geometry.centroid
    return a[["asset_id", "centroid"]]


def _resolve_group(
    x: np.ndarray, y: np.ndarray, grid: GridInfo, max_radius: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """asset centroids -> resolved (iy, ix) on `grid` + a per-asset fallback flag.

    resolution is cached per raw (iy, ix) pair since assets share cells heavily.
    """
    ix_raw = nearest_index(x, grid.x)
    iy_raw = nearest_index(y, grid.y)

    cache: dict[tuple[int, int], tuple[int, int] | None] = {}
    iy_out = np.empty(len(x), dtype=np.int64)
    ix_out = np.empty(len(x), dtype=np.int64)
    fallback = np.empty(len(x), dtype=bool)

    for j in range(len(x)):
        key = (int(iy_raw[j]), int(ix_raw[j]))
        if key not in cache:
            cache[key] = _nearest_valid_cell(*key, grid.mask, max_radius)
        resolved = cache[key]
        if resolved is None:
            raise FetchError(
                f"asset at raw cell {key} has no valid land cell within "
                f"{MAX_FALLBACK_M}m ({max_radius} cells) - widen MAX_FALLBACK_M or investigate"
            )
        iy_out[j], ix_out[j] = resolved
        fallback[j] = resolved != key

    return iy_out, ix_out, fallback


def build_cells(temp_grid: GridInfo, rain_grid: GridInfo) -> pl.DataFrame:
    """asset_id -> dense cell_id per variable-group (temp/rain), + fallback flags. Writes CELLS_PATH, GRID_PATH."""
    assets = scoped_asset_centroids()
    x = assets["centroid"].x.to_numpy()
    y = assets["centroid"].y.to_numpy()

    max_radius = int(np.ceil(MAX_FALLBACK_M / RESOLUTION_M))

    iy_t, ix_t, fb_t = _resolve_group(x, y, temp_grid, max_radius)
    iy_r, ix_r, fb_r = _resolve_group(x, y, rain_grid, max_radius)

    grid_rows = []

    def densify(iy: np.ndarray, ix: np.ndarray, grid: GridInfo, group: str) -> np.ndarray:
        keys = list(zip(iy.tolist(), ix.tolist()))
        uniq = sorted(set(keys))
        id_of = {k: i for i, k in enumerate(uniq)}
        for k, cid in id_of.items():
            grid_rows.append({"group": group, "cell_id": cid, "iy": k[0], "ix": k[1], "x": float(grid.x[k[1]]), "y": float(grid.y[k[0]])})
        return np.array([id_of[k] for k in keys], dtype=np.int32)

    cell_id_temp = densify(iy_t, ix_t, temp_grid, "temp")
    cell_id_rain = densify(iy_r, ix_r, rain_grid, "rain")

    cells = pl.DataFrame(
        {
            "asset_id": assets["asset_id"].to_numpy().astype(np.int64),
            "cell_id_temp": cell_id_temp,
            "cell_id_rain": cell_id_rain,
            "fallback_temp": fb_t,
            "fallback_rain": fb_r,
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells.write_parquet(CELLS_PATH)
    pl.DataFrame(grid_rows).write_parquet(GRID_PATH)

    print(
        f"cells: {cells.height} assets -> "
        f"{cells['cell_id_temp'].n_unique()} temp cells, {cells['cell_id_rain'].n_unique()} rain cells "
        f"(fallback: temp {cells['fallback_temp'].sum()}, rain {cells['fallback_rain'].sum()})"
    )
    return cells


# --- fetch loop ---


def group_cell_indices(group: str) -> tuple[np.ndarray, np.ndarray]:
    """dense cell_id -> (iy, ix) arrays for a variable-group, ordered by cell_id."""
    rows = pl.read_parquet(GRID_PATH).filter(pl.col("group") == group).sort("cell_id")
    return rows["iy"].to_numpy(), rows["ix"].to_numpy()


def fetch_month(var: str, year: int, month: int, iy: np.ndarray, ix: np.ndarray) -> None:
    shard = CACHE_DIR / var / f"{year}{month:02d}.parquet"
    if shard.exists():
        return

    tmp = SCRATCH_DIR / file_name(var, year, month)
    download(var, year, month, tmp)
    try:
        ds = xr.open_dataset(tmp)
        # advanced (point-wise) indexing: one value per (time, unique cell)
        values = ds[var].to_numpy()[:, iy, ix]  # (n_time, n_cells)
        dates = ds["time"].to_numpy()
        ds.close()
    finally:
        tmp.unlink(missing_ok=True)

    n_time, n_cells = values.shape
    df = pl.DataFrame(
        {
            "cell_id": np.tile(np.arange(n_cells, dtype=np.int32), n_time),
            "date": np.repeat(np.asarray(dates, dtype="datetime64[D]"), n_cells),
            "value": values.astype(np.float32).ravel(),
        }
    )
    shard.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(shard)


def consolidate_year(var: str, year: int) -> None:
    """merge complete-year monthly shards into one {year}.parquet; leaves partial years alone."""
    var_dir = CACHE_DIR / var
    month_shards = [var_dir / f"{year}{m:02d}.parquet" for m in range(1, 13)]
    if not all(p.exists() for p in month_shards):
        return
    out = var_dir / f"{year}.parquet"
    pl.concat([pl.read_parquet(p) for p in month_shards]).sort("cell_id", "date").write_parquet(out)
    for p in month_shards:
        p.unlink()


def main(limit_months: int | None = None):
    if not CELLS_PATH.exists() or not GRID_PATH.exists():
        temp_grid = load_grid(REFERENCE_VAR["temp"], FETCH_START.year, FETCH_START.month)
        rain_grid = load_grid(REFERENCE_VAR["rain"], FETCH_START.year, FETCH_START.month)
        build_cells(temp_grid, rain_grid)

    group_indices = {g: group_cell_indices(g) for g in ("temp", "rain")}
    # provisional months are published "shortly after the end of each month" - the
    # current, still-in-progress month is never available, so stop at the last full month.
    today = date.today()
    fetch_end = date(today.year, today.month, 1) - timedelta(days=1)

    months = list(month_range(FETCH_START, fetch_end))
    if limit_months is not None:
        months = months[:limit_months]

    fetched = skipped = 0
    for var in ALL_VARS:
        iy, ix = group_indices[GROUP_OF[var]]
        for year, month in months:
            shard = CACHE_DIR / var / f"{year}{month:02d}.parquet"
            if shard.exists() or (CACHE_DIR / var / f"{year}.parquet").exists():
                skipped += 1
                continue
            fetch_month(var, year, month, iy, ix)
            fetched += 1
            time.sleep(THROTTLE_S)
            if fetched % 50 == 0:
                print(f"{var}: {fetched} fetched, {skipped} skipped")

        for year in {y for y, _ in months}:
            consolidate_year(var, year)

    print(f"done: {fetched} fetched, {skipped} skipped")


if __name__ == "__main__":
    main()
