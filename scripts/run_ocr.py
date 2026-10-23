"""Detect and read text printed inside the first photograph of every article (EasyOCR, vi + en).

Writes one JSON line per article to <out-root>/ocr/ocr.jsonl (resumable). Run this before
`run_study.py --steps ocr`.

Example
  python scripts/run_ocr.py --data-root data/VnExpress --out-root outputs
"""
from __future__ import annotations

import argparse
import os

from _common import ids_for, load_corpus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ids", choices=["all", "test"], default="all")
    ap.add_argument("--cpu", action="store_true", help="run EasyOCR on CPU")
    ap.add_argument("--text-col", default=None)
    a = ap.parse_args()

    from vlmstudy import ocr as O

    df = load_corpus(a.data_root, a.text_col)
    O.run_ocr(df, ids_for(df, a.ids), os.path.join(a.out_root, "ocr", "ocr.jsonl"), gpu=not a.cpu)


if __name__ == "__main__":
    main()
