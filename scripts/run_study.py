"""Run the diagnosis protocol for one model (GPU).

Steps (all resumable; finished parts are skipped on re-run):
  check        sanity check on 30 test articles (constrained scoring vs greedy decoding)
  core         RQ1/RQ3: five input conditions on every article
  attribution  RQ2: gradient x input relevance on the test split (text+image, text only, image only)
  deletion     RQ2: top-k% relevant words / image cells vs the same number at random
  ocr          RQ2: text-in-image conditions (needs the OCR file from run_ocr.py)

Example
  python scripts/run_study.py --data-root data/VnExpress --out-root outputs --model qwen3vl8b
  python scripts/run_study.py --data-root data/VnExpress --out-root outputs --model internvl35_8b
"""
from __future__ import annotations

import argparse
import os

from _common import MODELS, V, ids_for, load_corpus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder with <Topic>/data.parquet and <Topic>/images/")
    ap.add_argument("--out-root", required=True, help="results are written to <out-root>/<model>/")
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--steps", nargs="+", default=["check", "core", "attribution", "deletion"],
                    choices=["check", "core", "attribution", "deletion", "ocr"])
    ap.add_argument("--core-ids", choices=["all", "test"], default="all", help="articles for RQ1/RQ3")
    ap.add_argument("--ocr-path", default=None, help="OCR jsonl (default <out-root>/ocr/ocr.jsonl)")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--ref-npz", default=None, help="optional npz with a reference test split (global_id)")
    ap.add_argument("--del-levels", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    ap.add_argument("--draws", type=int, default=2, help="random-deletion draws per article and level")
    # token budgets per batch; defaults are for a 96 GB GPU (40 GB A100: --budget 24576 --attr-budget 4096)
    ap.add_argument("--budget", type=int, default=65536)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--attr-budget", type=int, default=12288)
    ap.add_argument("--attr-cap", type=int, default=8)
    ap.add_argument("--revision", default=None, help="model revision / commit hash")
    a = ap.parse_args()

    df = load_corpus(a.data_root, a.text_col, a.ref_npz)
    core_ids, test_ids = ids_for(df, a.core_ids), ids_for(df, "test")
    V.log(f"{len(core_ids)} articles for RQ1/RQ3 | {len(test_ids)} test articles for RQ2")

    kind, ckpt = MODELS[a.model]
    vlm = V.VLM.load(kind, ckpt, revision=a.revision, name=a.model)
    st = V.Study(vlm, df, a.out_root, budget=a.budget, cap=a.cap, attr_budget=a.attr_budget, attr_cap=a.attr_cap)

    if "check" in a.steps:
        print(st.check(test_ids, n=30))
    if "core" in a.steps:
        st.run_core(core_ids)
    if "attribution" in a.steps:
        st.run_attribution(test_ids)
    if "deletion" in a.steps:
        st.run_deletion(test_ids, levels=tuple(a.del_levels), draws=a.draws)
    if "ocr" in a.steps:
        from vlmstudy import ocr as O

        path = a.ocr_path or os.path.join(a.out_root, "ocr", "ocr.jsonl")
        O.run_ocr_conditions(st, core_ids, O.load_ocr(path))


if __name__ == "__main__":
    main()
