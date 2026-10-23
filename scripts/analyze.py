"""Build the result tables (CSV + LaTeX) and diagnostic figures from stored model outputs (CPU only).

Reads <out-root>/<model>/ for every model that has been run and writes
  <out-root>/analysis/tables/T1..T8   (RQ1-RQ3, relevance, deletion check, topic words)
  <out-root>/analysis/tables/T9..T11  (text inside the image; only if the OCR conditions were run)
  <out-root>/analysis/figures/

Example
  python scripts/analyze.py --data-root data/VnExpress --out-root outputs
"""
from __future__ import annotations

import argparse
import os

from _common import V, load_corpus, saved_studies


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--ocr-path", default=None, help="OCR jsonl (default <out-root>/ocr/ocr.jsonl)")
    ap.add_argument("--text-col", default=None)
    a = ap.parse_args()

    df = load_corpus(a.data_root, a.text_col)
    studies = saved_studies(df, a.out_root)
    if not studies:
        raise SystemExit(f"no model outputs found under {a.out_root}")
    out_dir = os.path.join(a.out_root, "analysis")
    V.Analysis(studies, out_dir, B=a.bootstrap).run_all()

    ocr_path = a.ocr_path or os.path.join(a.out_root, "ocr", "ocr.jsonl")
    if os.path.exists(ocr_path):
        from vlmstudy import ocr as O

        O.ocr_analysis(studies, df, O.load_ocr(ocr_path), out_dir, B=a.bootstrap)
    V.log("tables and figures written to", out_dir)


if __name__ == "__main__":
    main()
