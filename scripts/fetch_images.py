"""Re-collect the photographs used in the study from the URLs in data/bn_vn3s_mm_metadata.csv.

Images are saved as <out-root>/<Topic>/images/<first_image_file>, the layout the experiment code
expects. The photographs are copyrighted by VnExpress and the respective photographers/agencies;
you are responsible for complying with the VnExpress terms of use. Some URLs may no longer resolve.

Example
  python scripts/fetch_images.py --out-root data/VnExpress
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.request

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default=os.path.join(os.path.dirname(__file__), "..", "data", "bn_vn3s_mm_metadata.csv"))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between requests")
    a = ap.parse_args()

    meta = pd.read_csv(a.metadata)
    failed = []
    for i, r in enumerate(meta.itertuples(index=False)):
        path = os.path.join(a.out_root, r.category, "images", r.first_image_file)
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            req = urllib.request.Request(r.first_image_url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(path, "wb") as f:
                f.write(resp.read())
        except Exception as e:  # keep going; report at the end
            failed.append((r.global_id, str(e)))
        time.sleep(a.sleep)
        if i % 500 == 0:
            print(f"{i}/{len(meta)}", flush=True)
    print(f"done; {len(failed)} failed")
    if failed:
        pd.DataFrame(failed, columns=["global_id", "error"]).to_csv(os.path.join(a.out_root, "fetch_failed.csv"), index=False)


if __name__ == "__main__":
    main()
