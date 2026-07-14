from project_paths import paths
import polars as pl

import joblib

import geopandas as gpd
from shapely.geometry import Point


def filter():
    # input_data = paths.processed_data / "unified_aims_eir_bgs.parquet"

    # input_df = pl.read_parquet(input_data)

    # # print(input_df.columns)

    # # management_areas = (
    # #     input_df.get_column("aims__water_management_area").unique().to_list()
    # # )

    # # print(management_areas)

    # ### ['North East', 'West Midlands', 'Thames', 'Cumbria and Lancashire', 'East Anglia', 'Greater Manchester Merseyside and Cheshire', 'Kent South London and East Sussex', 'Devon Cornwall and the Isles of Scilly', 'Hertfordshire and North London', 'Yorkshire', 'Wessex', 'Solent and South Downs', 'East Midlands', 'Lincolnshire and Northamptonshire']

    # ### ['North East', 'West Midlands', 'Thames', 'Cumbria and Lancashire', 'East Anglia', 'Greater Manchester Merseyside and Cheshire', 'Kent South London and East Sussex', 'Devon Cornwall and the Isles of Scilly', 'Hertfordshire and North London', 'Yorkshire', 'Wessex', 'Solent and South Downs', 'East Midlands', 'Lincolnshire and Northamptonshire']

    # local_df = input_df.filter(pl.col("aims__water_management_area").eq("Wessex"))

    # local_auths = local_df.get_column("aims__local_authority").unique().to_list()

    # print(local_auths)

    # # ['Bournemouth, Christchurch and Poole', 'Somerset West and Taunton', 'Mendip', 'Bath and North East Somerset', 'Stroud', 'Dorset', 'South Somerset', 'Wiltshire', 'North Somerset', 'Bristol, City of', 'New Forest', 'South Gloucestershire', 'Sedgemoor']

    # local_df = local_df.filter(
    #     pl.col("aims__local_authority").is_in(
    #         ["Somerset West and Taunton", "North Somerset", "Sedgemoor"]
    #     )
    # )

    df: pl.DataFrame = joblib.load(
        paths.processed_data / "assets_for_heuristics_df.joblib"
    )
    local_df = df.filter(
        pl.col("aims__water_management_area").eq("Wessex")
        & pl.col("aims__local_authority").is_in(
            ["Somerset West and Taunton", "North Somerset", "Sedgemoor"]
        )
    )

    print(local_df.head())
    print(local_df.height)

    named_df = local_df.filter(
        pl.col("aims__asset_name").is_not_null()
        & ~pl.col("aims__asset_name").eq("Not Available")
    )

    print(named_df.head())
    print(named_df.height)

    print(local_df.get_column("aims__asset_maintainer").unique().to_list())

    ea_maintained = local_df.filter(
        pl.col("aims__asset_maintainer").eq("Environment Agency")
    )

    print(ea_maintained.head())
    print(ea_maintained.height)

    not_nhg_assets = ea_maintained.filter(
        pl.col("original_sub_type").ne("Natural High Ground")
    )

    print(not_nhg_assets.head())
    print(not_nhg_assets.height)

    not_hg_assets = not_nhg_assets.filter(
        pl.col("original_sub_type").ne("Engineered High Ground")
    )

    print(not_hg_assets.head())
    print(not_hg_assets.height)

    print(
        not_hg_assets.group_by("original_sub_type", "grade")
        .len()
        .sort("original_sub_type", "grade")
    )
    print(
        not_hg_assets.group_by("original_sub_type").len().sort("len", descending=True)
    )

    return not_hg_assets


if __name__ == "__main__":
    df = filter()
    df.select(
        "aims__asset_id",
        "original_sub_type",
        "halcrow_type",
        "tier",
        "environment",
        "aims__protection_type",
        "grade",
        "aims__target_condition",
        "aims__asset_length",
        "age_years",
        "age_valid",
        "aims__asset_start_date",
        "aims__year_last_refurbished",
    ).write_csv(paths.processed_data / "local_assets_export.csv")
