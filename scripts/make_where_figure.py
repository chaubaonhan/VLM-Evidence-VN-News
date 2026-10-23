"""Qualitative "where does the model look" figure (paper Fig. 3) for chosen test articles.

Needs the photographs on local disk (they are not distributed with this repository) and the stored
relevance maps of both models. Without --pick, renders candidate panels for each group
(image fixes / image breaks / text in image) so that examples can be chosen.

Example
  python scripts/make_where_figure.py --data-root data/VnExpress --out-root outputs \
      --pick Business_15442 Business_15141
"""
from __future__ import annotations

import argparse
import os

from _common import load_corpus, saved_studies


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--pick", nargs="*", default=[], help="global_ids of the articles to show")
    ap.add_argument("--titles", nargs="*", default=None)
    ap.add_argument("--per-group", type=int, default=3)
    a = ap.parse_args()

    from vlmstudy import figure as FG
    from vlmstudy import ocr as O

    df = load_corpus(a.data_root)
    studies = saved_studies(df, a.out_root)
    ocr_path = os.path.join(a.out_root, "ocr", "ocr.jsonl")
    ocr = O.load_ocr(ocr_path) if os.path.exists(ocr_path) else None
    fig_dir = os.path.join(a.out_root, "analysis", "figures", "where_it_looks")
    if a.pick:
        FG.figure(studies, df, a.pick, os.path.join(fig_dir, "fig_where_it_looks.pdf"), ocr, titles=a.titles)
    else:
        FG.render_candidates(studies, df, fig_dir, ocr=ocr, per_group=a.per_group)


if __name__ == "__main__":
    main()
