import os

os.environ["GDAL_PAM_ENABLED"] = "NO"  # skip .aux.xml sidecar churn

import rasterio
from project_paths import paths

WINDOWS = paths.data / "lidar" / "dtm_windows"
files = sorted(WINDOWS.glob("*.tif"))
print(f"checking {len(files)} files", flush=True)

bad = []
for i, p in enumerate(files):
    if i % 500 == 0:
        print(f"  {i}/{len(files)} ok so far, {len(bad)} bad", flush=True)
    if p.stat().st_size < 1024:  # 0-byte / stub -> corrupt, no need to open
        bad.append(p.name)
        continue
    try:
        with rasterio.open(p) as ds:
            ds.read(1)  # FULL read - catches trailing-tile truncation
    except Exception as e:
        print(f"  BAD {p.name}: {type(e).__name__}", flush=True)
        bad.append(p.name)

print(f"\n{len(bad)} corrupt", flush=True)
for name in bad:
    (WINDOWS / name).unlink()
