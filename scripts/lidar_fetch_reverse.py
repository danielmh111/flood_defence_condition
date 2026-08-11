# second worker for src/lidar_fetch.py's fetch: same output dir, but walks the
# asset list back-to-front with a thread pool so it can run alongside the
# still-running serial front-to-back process without re-fetching the same
# assets. The two only risk colliding on a handful of assets near wherever they
# meet in the middle -- harmless (one overwrites the other's identical tile),
# not worth locking for a one-off run. Importing lidar_fetch is safe: it has no
# module-level side effects, run() only fires under __main__.

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.lidar_fetch import MAX_PX, OUT_DIR, padded_bbox, scoped_assets
from src.wcs_utils import WCSClient

N_WORKERS = 4


def fetch_one(asset_id: int, bbox: tuple[float, float, float, float], cov) -> tuple[int, str]:
    out = OUT_DIR / f"{asset_id}.tif"
    if out.exists():
        return asset_id, "cached"
    client = WCSClient()  # own session per call; full retry/backoff kept for real fetching
    try:
        out.write_bytes(client.get_coverage(cov, bbox))
        return asset_id, "ok"
    except Exception as e:
        return asset_id, str(e)[:100]


def run():
    assets = scoped_assets().iloc[::-1].reset_index(drop=True)
    cov = WCSClient().discover()

    tasks = []
    oversize: list[tuple[int, int, int]] = []
    for row in assets.itertuples(index=False):
        bbox = padded_bbox(row.geometry)
        width, height = round(bbox[2] - bbox[0]), round(bbox[3] - bbox[1])
        if max(width, height) > MAX_PX:
            oversize.append((int(row.asset_id), width, height))
            continue
        tasks.append((int(row.asset_id), bbox))

    print(f"reverse worker: {len(tasks)} candidates, {N_WORKERS} workers, {len(oversize)} oversize")

    fetched = cached = failed = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(fetch_one, aid, bbox, cov): aid for aid, bbox in tasks}
        for fut in as_completed(futs):
            asset_id, status = fut.result()
            if status == "ok":
                fetched += 1
            elif status == "cached":
                cached += 1
            else:
                failed += 1
                print(f"  asset {asset_id}: {status}")
            if (fetched + cached) % 100 == 0:
                print(f"{fetched} fetched, {cached} cached, {failed} failed")

    print(f"done: {fetched} fetched, {cached} cached, {failed} failed")


if __name__ == "__main__":
    run()
