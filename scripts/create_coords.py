from project_paths import paths
import geopandas as gpd


def main():
    input = paths.aims_data / "aims.gpkg"
    gdf = gpd.read_file(input)
    gdf = gdf[["asset_id", "geometry"]]

    print(gdf)

    gdf["centroid_x"], gdf["centroid_y"] = (
        gdf.geometry.centroid.x,
        gdf.geometry.centroid.y,
    )

    gdf["centroid"] = gdf.geometry.centroid
    gdf["centroid_degrees"] = gdf.geometry.to_crs(epsg=4326).centroid

    gdf["centroid_lon"], gdf["centroid_lat"] = (
        gdf["centroid_degrees"].x,
        gdf["centroid_degrees"].y,
    )

    print(gdf[["asset_id", "centroid_x", "centroid_y"]])
    print()
    print(gdf[["asset_id", "centroid_lat", "centroid_lon"]])

    gdf = gdf[["asset_id", "centroid", "centroid_lat", "centroid_lon"]]
    gdf.to_parquet(paths.processed_data / "asset_coordinates.parquet")
    gdf.to_csv(paths.processed_data / "asset_coordinates.csv")


if __name__ == "__main__":
    main()
