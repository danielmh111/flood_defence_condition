# one-off probe: does environment.data.gov.uk tolerate concurrent / unthrottled
# WCS requests, or does it start returning 429/5xx? src/lidar_fetch.py is running
# live and serial (0.5s throttle) while this runs -- this script must not touch
# it or its output dir. It writes to a separate throwaway dir and samples assets
# from the tail of the unfetched list (lidar_fetch works front-to-back) to keep
# collision risk low.

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import polars as pl

from project_paths import paths
from src.features import prepare_features
from src.wcs_utils import WCSClient

PAD_M = 30.0
MAX_PX = 3000
N_WORKERS = 4
N_PROBE = 40

CACHED_DIR = paths.data / "lidar" / "dtm_windows"
PROBE_DIR = paths.data / "lidar" / "_probe_windows"
PROBE_DIR.mkdir(parents=True, exist_ok=True)


def probe_assets(n: int) -> gpd.GeoDataFrame:
    scope = prepare_features(pl.read_parquet(paths.unified_file))["asset_id"].to_list()
    assets = gpd.read_file(paths.aims_data / "aims.gpkg", columns=["asset_id"])
    assets["asset_id"] = assets["asset_id"].astype("int64")
    assets = assets[assets["asset_id"].isin(scope)].reset_index(drop=True)

    cached = {int(p.stem) for p in CACHED_DIR.glob("*.tif")}
    remaining = assets[~assets["asset_id"].isin(cached)].reset_index(drop=True)
    return remaining.iloc[-n:].reset_index(drop=True)


def padded_bbox(geom) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = geom.bounds
    return (xmin - PAD_M, ymin - PAD_M, xmax + PAD_M, ymax + PAD_M)


def fetch_one(asset_id: int, bbox: tuple[float, float, float, float], cov) -> tuple[int, str, float]:
    """Fresh client (own session) per call -- no state shared across threads.
    max_retries=1 so failures surface as a raw status/error, not masked by backoff."""
    client = WCSClient(max_retries=1)
    t0 = time.monotonic()
    try:
        data = client.get_coverage(cov, bbox)
        (PROBE_DIR / f"{asset_id}.tif").write_bytes(data)
        return asset_id, "200", time.monotonic() - t0
    except Exception as e:
        return asset_id, str(e)[:100], time.monotonic() - t0


def main():
    assets = probe_assets(N_PROBE)

    boxes = [(int(row.asset_id), padded_bbox(row.geometry)) for row in assets.itertuples(index=False)]
    boxes = [(aid, b) for aid, b in boxes if max(b[2] - b[0], b[3] - b[1]) <= MAX_PX]

    cov = WCSClient().discover()
    print(f"probing {len(boxes)} assets, {N_WORKERS} workers, no throttle -> {PROBE_DIR}")

    t0 = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(fetch_one, aid, bbox, cov) for aid, bbox in boxes]
        for fut in as_completed(futs):
            results.append(fut.result())
    elapsed = time.monotonic() - t0

    ok = [r for r in results if r[1] == "200"]
    bad = [r for r in results if r[1] != "200"]

    print(
        f"\n{len(ok)}/{len(results)} ok in {elapsed:.1f}s wall "
        f"({elapsed / max(len(results), 1):.2f}s/asset avg, {N_WORKERS}x concurrency)"
    )
    if ok:
        times = sorted(t for _, _, t in ok)
        print(f"per-request latency: p50={times[len(times) // 2]:.2f}s p90={times[int(len(times) * 0.9)]:.2f}s")
    if bad:
        print(f"{len(bad)} failed:")
        for asset_id, status, dt in bad:
            print(f"  asset {asset_id}: {status} (after {dt:.1f}s)")
    else:
        print("no failures -- server tolerated concurrency in this probe")


if __name__ == "__main__":
    main()
