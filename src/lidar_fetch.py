import geopandas as gpd
import polars as pl
from project_paths import paths
import time
import rasterio

from src.features import prepare_features
from src.wcs_utils import WCSClient

PAD_M = 30.0  # transect plus buffer search room around each asset bbox
MAX_PX = 3000  # use max size to avoid requests being denied
THROTTLE_S = 0.5  # rate limit

OUT_DIR = paths.data / "lidar" / "dtm_windows"


def scoped_assets() -> gpd.GeoDataFrame:
    scope = prepare_features(pl.read_parquet(paths.unified_file))["asset_id"].to_list()

    assets = gpd.read_file(paths.aims_data / "aims.gpkg", columns=["asset_id"])
    assets["asset_id"] = assets["asset_id"].astype("int64")
    assets = assets[assets["asset_id"].isin(scope)].reset_index(drop=True)

    if assets.crs is None or assets.crs.to_epsg() != 27700:
        raise ValueError(f"expected epsg:27700, got {assets.crs}")

    return assets


def padded_bbox(geom) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = geom.bounds
    return (xmin - PAD_M, ymin - PAD_M, xmax + PAD_M, ymax + PAD_M)


def run(limit: int | None = None):

    assets = scoped_assets()
    client = WCSClient()
    cov = client.discover()

    print(
        f"coverage={cov.coverage_id} axes=({cov.axis_x},{cov.axis_y}) format={cov.tiff_format}"
    )

    fetched = cached = 0
    oversize: list[tuple[int, int, int]] = []

    for row in assets.itertuples(index=False):
        if limit is not None and fetched >= limit:
            break

        out = OUT_DIR / f"{row.asset_id}.tif"
        if out.exists():
            cached += 1
            continue

        bbox = padded_bbox(row.geometry)
        width, height = (
            round(bbox[2] - bbox[0]),
            round(bbox[3] - bbox[1]),
        )  # 1m -> m == px
        if max(width, height) > MAX_PX:
            oversize.append((row.asset_id, width, height))
            continue

        out.write_bytes(client.get_coverage(cov, bbox))

        fetched += 1

        time.sleep(THROTTLE_S)
        if fetched % 100 == 0:
            print(f"{fetched} fetched, {cached} cached, {len(oversize)} oversize")

    print(f"done: {fetched} fetched, {cached} cached, {len(oversize)} oversize")
    if oversize:
        pl.DataFrame(
            oversize, schema=["asset_id", "w", "h"], orient="row"
        ).write_parquet(OUT_DIR.parent / "oversize_assets.parquet")


if __name__ == "__main__":
    run()
