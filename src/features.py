# important points to remember in this fiel
# remove any features that leak info eg target grade
# remove any features that have really high correlation with another feature eg design and actual crest level
# output should be a numpy matrix
# exclude the target
#
# edit - changing my mind on output type
# instead of producing numpy matrix here, write a func that takes in a dataframe and outputs another dataframe,
# using the schema to control the types
# then i can write a script that loads the file, extracts/formats the features, then casts to np and saves the feature matrix

import numpy as np
import polars as pl

LEAKY_COLS = [
    "aims__current_condition",  # dupe of target, using eir originating grade instead
    "aims__target_condition",  # target grade obviously would be a great predictor, but probably has loads of info from last inspection and is more like a leak than a signal in context
    "eir__asset_target_condition",  # dupe of above from different source
    "eir__below_required_condition",  # derived from current vs required
    "eir__repair_deadline",  # exists because the grade is poor so a leak of the grade, only available following an inspection so kinda doesnt make sense if the premise is to replace some inspection with ml prediction
    "eir__agreed_by_engineer",  # metadata of the grading inspection, as above we should avoid using features that come directly from the inspection if we are trying to train a model that we compare to the human inspection
    "eir__inspection_date",  # timestamp of when labelled (used only for age)
    "aims__last_inspection_date",  # recency ( rho 0.063 anyway) / scheduling
    "aims__next_inspection_date",  # partly scheduled off current condition probably?
    "aims__current_sop_date",
    "aims__year_last_refurbished",  # ~0 coverage
    "eir__year_last_refurbished",  # ~0 coverage
]


NOMINAL_CATEGORY_COLS = [
    "aims__asset_sub_type",
    "aims__protection_type",
    "aims__primary_purpose",
]

GEOSURE_COLS = [
    "geosure_collapsible_deposits__class",
    "geosure_compressible_ground__class",
    "geosure_landslides__class",
    "geosure_running_sand__class",
    "geosure_shrink_swell__class",
    "geosure_soluble_rocks__class",
]

SOP_SENTINELS = [9999.0, 10000.0]
AGE_MIN, AGE_MAX = 0.0, 150.0  # rule from  nb 08


def to_enum(s: pl.Series) -> pl.Enum:
    """Enum from a columns sorted,  nonnull unique values."""
    return pl.Enum(sorted(s.drop_nulls().unique().to_list()))


def prepare_features(
    df: pl.DataFrame,
    exclude_subtypes: tuple[str, ...] | None = ("Natural High Ground",),
) -> pl.DataFrame:
    exclude = list(exclude_subtypes or [])

    # target
    df = (
        df.with_columns(
            pl.col("eir__condition_grade")
            .round(0)
            .cast(pl.Int8)
            .alias("condition_grade")
        )
        .filter(pl.col("condition_grade").is_between(1, 5))
        .filter(~pl.col("aims__asset_sub_type").is_in(exclude))
    )

    # category sets
    bedrock_top = (
        df["bedrock_geo__lex_rcs"]
        .drop_nulls()
        .value_counts(sort=True)
        .head(20)["bedrock_geo__lex_rcs"]
        .to_list()
    )
    bedrock_enum = pl.Enum(sorted(bedrock_top) + ["the_rest"])

    age_years = (
        pl.col("eir__inspection_date").cast(pl.Date) - pl.col("aims__asset_start_date")
    ).dt.total_days() / 365.25
    df = df.with_columns(
        # design_sop: strip out 9999/10000 suspicious values, make null
        pl.when(pl.col("aims__design_sop").is_in(SOP_SENTINELS))
        .then(None)
        .otherwise(pl.col("aims__design_sop"))
        .alias("_design_sop"),
        # age, only valid inside 0-150 years, else null
        pl.when(age_years.is_between(AGE_MIN, AGE_MAX))
        .then(age_years)
        .otherwise(None)
        .alias("_age_years"),
        # bedrock lithology binned to top 20 + 'the_rest', nulls preserved as null
        pl.when(pl.col("bedrock_geo__lex_rcs").is_null())
        .then(None)
        .when(pl.col("bedrock_geo__lex_rcs").is_in(bedrock_top))
        .then(pl.col("bedrock_geo__lex_rcs"))
        .otherwise(pl.lit("the_rest"))
        .alias("_bedrock_binned"),
    )

    return df.select(
        # keys / target
        pl.col("aims__asset_id").alias("asset_id"),
        pl.col("condition_grade"),
        # nominal categoricals cat to enum
        *(
            pl.col(col).cast(to_enum(df.get_column(col)))
            for col in NOMINAL_CATEGORY_COLS
        ),
        pl.col("_bedrock_binned").cast(bedrock_enum).alias("bedrock_lex_rcs_binned"),
        # geosure ordinals 1-3
        *(pl.col(col).cast(pl.Int8) for col in GEOSURE_COLS),
        # binary indicator
        pl.col("aims__asset_maintainer")
        .str.contains("(?i)environment agency")
        .cast(pl.Int8)
        .alias("maintainer_is_ea"),
        # numerics
        pl.col("aims__asset_length").log1p().alias("asset_length_log1p"),
        pl.col("aims__actual_dcl").alias("actual_dcl"),
        pl.col("aims__actual_dcl").is_null().cast(pl.Int8).alias("actual_dcl_missing"),
        pl.col("_design_sop").alias("design_sop"),
        pl.col("_design_sop").is_null().cast(pl.Int8).alias("design_sop_missing"),
        pl.col("_age_years").alias("age_years"),
        pl.col("_age_years").is_null().cast(pl.Int8).alias("age_estimated"),
    )


# --- intended CALL-SITE asserts (belong in the materialisation script, not here) --
#
#   out = prepare_features(df)
#   assert set(LEAKY_COLS).isdisjoint(out.columns)        # no leak slipped in
#   assert out["asset_id"].n_unique() == out.height       # one row per asset
#   assert out["condition_grade"].is_between(1, 5).all()  # target domain
#   assert out["asset_length_log1p"].is_finite().all()    # no inf/NaN from log
#
#   # features the EDA proved 100%-covered should carry no nulls (fail loud if join drifts):
#   full_coverage = [*NOMINAL_COLS, *GEOSURE_COLS, "maintainer_is_ea", "asset_length_log1p"]
#   assert out.select(full_coverage).null_count().to_numpy().sum() == 0
