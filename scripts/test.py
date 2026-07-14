from src.features import prepare_features, LEAKY_COLS
from project_paths import paths
import polars as pl


def main():
    input_data = paths.processed_data / "unified_aims_eir_bgs.parquet"

    input_df = pl.read_parquet(input_data)

    features_df = prepare_features(df=input_df)

    print(features_df.head())
    print()

    assert set(LEAKY_COLS).isdisjoint(features_df.columns)
    assert features_df["asset_id"].n_unique() == features_df.height
    assert features_df["condition_grade"].is_between(1, 5).all()
    assert features_df["asset_length_log1p"].is_finite().all()

    print("passed assertions")

    print()
    print(features_df.schema)
    print()
    print(features_df.describe())


if __name__ == "__main__":
    main()
