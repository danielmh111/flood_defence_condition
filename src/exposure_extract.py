import geopandas as gpd
import pandas as pd
import polars as pl
import shapely
from project_paths import paths
from src.features import prepare_features
from dataclasses import dataclass
from enum import Enum


class SchemaError(Exception): ...


class Geometry(Enum):
    LINE = "line"
    POINT = "point"


@dataclass()
class Regime:
    geometry: Geometry
    distance: int


# create regimes of geometries.
# assets that are more like points than lines will join to flood polygons in a radius whereas lines will use a linear buffer corridor
# ive divided the line types into broad and narrow lines and chosen an intuative value for the corridor for now

GEOMETRY_REGIMES = {
    **{
        subtype: Regime(Geometry.LINE, 20)
        for subtype in [
            "Embankment",
            "Engineered High Ground",
            "Natural High Ground",
            "Cliff",
            "Dunes",
            "Beach",
            "Barrier Beach",
        ]
    },
    **{
        subtype: Regime(Geometry.LINE, 5)
        for subtype in ["Wall", "Quay", "Promenade", "Demountable Defence"]
    },
    **{
        subtype: Regime(Geometry.POINT, 5)
        for subtype in ["Flood Gate", "Weir", "Spillway", "Bridge Abutment"]
    },
}

# from data dictionary, (docs/guidance_recorded_flood_outlines) names change in 2020, this dict maps them together
PRE_2020_RENAMES = {
    "outline_co": "rec_out_id",
    "event_co": "rec_grp_id",
    "hfm_ind": "hfm_status",
    "bndry_src": "data_src",
    "fluvial_in": "fluvial_f",
    "coastal_ind": "coastal_f",
    "tidal_ind": "tidal_f",
    # name, start_date, end_date, flood_src, flood_caus unchanged
}

EVENT_SCHEMA = pl.Schema(
    {
        "asset_id": pl.Int64,
        "rec_out_id": pl.String,
        "rec_grp_id": pl.String,
        "name": pl.String,
        "start_date": pl.Date,
        "end_date": pl.Date,
        "flood_src": pl.String,
        "flood_caus": pl.String,
        "src_fluvial": pl.Boolean,
        "src_coastal": pl.Boolean,
        "src_tidal": pl.Boolean,
        "src_surface_water": pl.Boolean,
        "intersected_area_m2": pl.Float64,
    }
)


def load_scoped_assets() -> gpd.GeoDataFrame:
    """aims asset geoms filtered to the prepare_features set"""
    scope = prepare_features(pl.read_parquet(paths.unified_file))["asset_id"]

    assets = gpd.read_file(
        paths.aims_data / "aims.gpkg", columns=["asset_id", "asset_sub_type"]
    )
    assets["asset_id"] = assets["asset_id"].astype("int64")
    assets = assets[assets["asset_id"].isin(scope.to_list())].reset_index(drop=True)

    if assets.crs is None or assets.crs.to_epsg() != 27700:
        raise SchemaError(
            f"expected the input data to use epsg:27700 for crs, found {assets.crs}"
        )

    return assets


def _bool_flag(s: pd.Series) -> pd.Series:
    # flags arrive as 'True'/'False' strings in the current release
    return s if s.dtype == bool else s.str.lower().eq("true")


def load_recorded_flood_outlines() -> tuple[gpd.GeoDataFrame, str]:
    """load recorded flood outlines, standardise to the post 2020 field names so downstream sees post-2020 names only"""

    rfo = gpd.read_file(paths.ea_data / "Recorded_Flood_Outlines.gpkg")
    rfo.columns = [c.lower() for c in rfo.columns]

    has_post = "rec_out_id" in rfo.columns
    has_pre = "outline_co" in rfo.columns
    variant = (
        "mixed" if has_post and has_pre else "post-2020" if has_post else "pre-2020"
    )
    rfo = rfo.rename(
        columns={
            old: new
            for old, new in PRE_2020_RENAMES.items()
            if old in rfo.columns and new not in rfo.columns
        }
    )

    if rfo.crs is None:
        rfo = rfo.set_crs(27700)
    elif rfo.crs.to_epsg() != 27700:
        rfo = rfo.to_crs(27700)

    rfo["src_fluvial"] = _bool_flag(rfo["fluvial_f"])
    rfo["src_coastal"] = _bool_flag(rfo["coastal_f"])
    rfo["src_tidal"] = _bool_flag(rfo["tidal_f"])
    rfo["src_surface_water"] = rfo["flood_caus"].eq("local drainage/surface water")

    keep = [col for col in EVENT_SCHEMA if col != "asset_id" and col in rfo.columns]
    missing_cols = [
        col
        for col in EVENT_SCHEMA
        if col not in ("asset_id", "intersected_area_m2", *keep)
    ]
    if missing_cols:
        raise SchemaError(f"missing column(s): {missing_cols}")

    rfo = rfo[[*keep, "geometry"]].reset_index(drop=True)

    # the national file carries a few exact-duplicate outlines (identical
    # attributes and geometry, some repeated 4-5x) which would inflate
    # downstream area sums; drop them here, keeping the first
    duplicated = rfo.drop(columns="geometry").assign(wkb=rfo.geometry.to_wkb()).duplicated()
    rfo = rfo[~duplicated].reset_index(drop=True)

    return rfo, variant


def build_corridors(
    assets: gpd.GeoDataFrame, regimes: dict[str, Regime]
) -> gpd.GeoDataFrame:
    """per-asset catch-area polygons from the geometry-regime config."""

    unknown = set(assets["asset_sub_type"]) - set(regimes)
    if unknown:
        raise ValueError(f"unknown asset subtype with regime unspecified: {unknown}")

    regime = assets["asset_sub_type"].map(
        {subtype: regime.geometry for subtype, regime in regimes.items()}
    )
    distance = assets["asset_sub_type"].map(
        {subtype: regime.distance for subtype, regime in regimes.items()}
    )

    base = assets.geometry.where(regime != Geometry.POINT, assets.geometry.centroid)

    buffered_assets = assets[["asset_id"]].set_geometry(
        base.buffer(distance), crs=assets.crs
    )

    return buffered_assets


def extract_exposure_events(
    assets: gpd.GeoDataFrame,
    rfo: gpd.GeoDataFrame,
    regimes: dict[str, Regime] = GEOMETRY_REGIMES,
) -> pl.DataFrame:
    """one row per (asset x intersecting outline); zero-area boundary touches dropped."""

    corridors = build_corridors(assets, regimes)

    pairs = gpd.sjoin(corridors, rfo, how="inner", predicate="intersects")
    outline_geom = rfo.geometry.to_numpy()[pairs["index_right"].to_numpy()]
    pairs["intersected_area_m2"] = shapely.area(
        shapely.intersection(pairs.geometry.to_numpy(), outline_geom)
    )

    events = pl.from_pandas(pd.DataFrame(pairs[list(EVENT_SCHEMA)]))

    # id fields are ints or floats in the file, cast all to int64 to get rid of decimals then cast to str
    events = (
        events.with_columns(
            pl.col(c).cast(pl.Int64).cast(pl.String)
            for c in ("rec_out_id", "rec_grp_id")
            if events.schema[c].is_numeric()
        )
        .filter(pl.col("intersected_area_m2") > 0)
        .cast(EVENT_SCHEMA)
        .sort("asset_id", "start_date", "rec_out_id")
    )

    return events


def main():
    assets = load_scoped_assets()
    rfo, schema_variant = load_recorded_flood_outlines()
    events = extract_exposure_events(assets, rfo)

    assert events["asset_id"].is_in(assets["asset_id"].to_list()).all()
    assert (events["intersected_area_m2"] > 0).all()

    out_path = paths.processed_data / "exposure_events.parquet"
    events.write_parquet(out_path)

    print(
        f"wrote {events.height} event rows for {events['asset_id'].n_unique()} assets -> {out_path}"
    )


if __name__ == "__main__":
    main()
