import geopandas as gpd
import pandas as pd
from project_paths import project_root, paths


def load_aims() -> gpd.GeoDataFrame:
    return gpd.read_file(paths.aims_data / "aims.gpkg")


def load_ea() -> pd.DataFrame:
    return pd.read_csv(paths.ea_data / "ea_eir_data.csv")
