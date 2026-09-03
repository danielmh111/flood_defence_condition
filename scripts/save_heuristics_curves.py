from project_paths import paths

from src.heuristic import curve_table

curve_table().write_parquet(paths.report_data / "halcrow_curves.parquet")
