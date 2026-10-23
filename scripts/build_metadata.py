"""Build data/bn_vn3s_mm_metadata.csv from the full corpus (article text and images stay local).

The released file lists, for every article, its id, topic, split, source URL and the URL of the
photograph used (the first image), plus an exact-duplicate-text flag. It is produced with the same
loading and split code as the experiments, so the split matches the paper.

Example
  python scripts/build_metadata.py --data-root data/VnExpress --out data/bn_vn3s_mm_metadata.csv
"""
from __future__ import annotations

import argparse
import hashlib
import os

import pandas as pd

from _common import V, load_corpus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="data/bn_vn3s_mm_metadata.csv")
    a = ap.parse_args()

    df = load_corpus(a.data_root)
    rows = []
    for cat in V.CATS:
        r = pd.read_parquet(os.path.join(a.data_root, cat, "data.parquet"))
        for _, x in r.iterrows():
            files, urls = list(x.image_files), list(x.image_urls)
            k = next((i for i, f in enumerate(files) if "_00." in f), 0)
            text = str(x.text).strip()
            rows.append(dict(global_id=f"{cat}_{int(x.source_index):04d}", article_id=x["id"], url=x.url,
                             first_image_file=files[k], first_image_url=urls[k] if k < len(urls) else None,
                             num_images=int(x.num_images), text_chars=len(text),
                             _h=hashlib.sha1(text.encode()).hexdigest()))
    meta = df[["global_id", "category", "label", "split"]].merge(pd.DataFrame(rows), on="global_id", validate="1:1")
    meta["exact_duplicate_text"] = (meta.groupby("_h").global_id.transform("count") > 1) & (meta.text_chars > 0)
    meta.drop(columns="_h").to_csv(a.out, index=False)
    V.log(f"{len(meta)} rows -> {a.out} | split {meta.split.value_counts().to_dict()} | "
          f"duplicate-text rows {int(meta.exact_duplicate_text.sum())}")


if __name__ == "__main__":
    main()
