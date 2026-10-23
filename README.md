# What Evidence Do Vision-Language Models Use to Classify Vietnamese News?

Code, prompt, data metadata and result tables for the paper

> **Near-Parity Hides Decision Churn: What Evidence Do Vision-Language Models Use to Classify Vietnamese News?**
> Bao Nhan Chau (Ton Duc Thang University) and Anh Hoang (University of Economics Ho Chi Minh City)

The study treats "which evidence does a VLM use?" as a measurement problem. The model checkpoint, the
prompt and the scoring are fixed, and only the input of the **same** article is changed. It is applied to
10,000 Vietnamese text–image news articles (BN_VN3S_MM, ten VnExpress topics) with
**Qwen3-VL-8B-Instruct** and **InternVL3.5-8B**.

## Protocol

| Component | What it does | Code |
|---|---|---|
| Constrained scoring | Softmax over the next-token logits of the ten label tokens at the answer position; reports the probability mass ("coverage") on those tokens | `core.VLM.score` |
| Five input conditions (RQ1, RQ3) | text+image, text only, image only, text + blank image (ImageNet-mean colour, same size), text + image of an article from another topic | `core.Study.build`, `run_core` |
| Per-article change measures | label flip, probability shift, Jensen–Shannon divergence, mismatched-image capture, gain/loss/oracle | `core.Analysis` (T1–T5) |
| Gradient×input relevance (RQ2) | One backward pass on log p(ŷ) w.r.t. all language-model inputs, including Qwen3-VL's DeepStack features; image share, word relevance, image heatmaps | `core.VLM.attribute`, `run_attribution` |
| Deletion check (RQ2) | Remove the top 10/20/40% relevant words or image cells vs the same number at random (two draws) | `core.Study.run_deletion` (T7) |
| Text inside the photograph (RQ2) | EasyOCR (vi+en); mask the printed text vs equal-area random boxes, with and without the article; OCR string only | `ocr.py` (T9–T11) |
| Statistics | Paired bootstrap over articles (10,000 resamples), Holm correction within families | `core.boot_*`, `core.holm` |

The verbatim prompt is in [`docs/prompt.md`](docs/prompt.md).

## Main results (Macro-F1, %, n = 10,000)

| Input | Qwen3-VL-8B | InternVL3.5-8B |
|---|---|---|
| text + image | 73.4 | 68.2 |
| text only | 71.3 | 68.7 |
| image only | 62.0 | 52.9 |
| text + blank image | 70.1 | 67.3 |
| text + other-topic image | 68.2 | 64.9 |
| labels flipped when the image is removed | 16.3% | 13.6% |

All tables with confidence intervals are in [`results/`](results/).

## Repository layout

```
src/vlmstudy/
  core.py      data, prompt, VLM wrapper (Qwen3-VL, InternVL3.5), conditions, relevance, deletion,
               statistics, analysis tables and figures
  ocr.py       text-in-image analysis (EasyOCR, masking conditions, tables T9-T11)
  figure.py    qualitative "where does the model look" figure (needs local photographs)
scripts/
  run_study.py           run one model (GPU): check, core, attribution, deletion, ocr
  run_ocr.py             EasyOCR over the first photograph of every article
  analyze.py             tables T1-T11 and diagnostic figures from stored outputs (CPU)
  make_paper_figures.py  paper Figs. 1, 2, 4 from results/tables
  make_where_figure.py   paper Fig. 3 (requires the photographs)
  build_metadata.py      data/bn_vn3s_mm_metadata.csv from the full corpus
  fetch_images.py        re-collect the photographs from their source URLs
data/                    BN_VN3S_MM metadata (ids, labels, split, URLs); no texts or images
results/                 result tables of the paper and redrawn figures
docs/prompt.md           the classification prompt
```

## Setup

```bash
git clone https://github.com/chaubaonhan/VLM-Evidence-VN-News.git
cd VLM-Evidence-VN-News
pip install -e .            # or: pip install -r requirements.txt
pip install -e ".[ocr]"     # adds EasyOCR for the text-in-image analysis
```

Tested with Python 3.13, PyTorch 2.11 (CUDA 13.0) and Transformers 5.18.0 (also 4.57.6). Models are
loaded in bfloat16 with SDPA attention from `Qwen/Qwen3-VL-8B-Instruct` and `OpenGVLab/InternVL3_5-8B-HF`.

## Data

`data/bn_vn3s_mm_metadata.csv` lists every article with its topic, split, article URL and the URL of the
photograph used. Article texts and photographs are copyrighted by VnExpress and are **not** included
(see [`data/README.md`](data/README.md)). To run the experiments you need a local copy laid out as

```
data/VnExpress/<Topic>/data.parquet          # url, label, id, source_index, text, image_files, image_urls, ...
data/VnExpress/<Topic>/images/<id>_00.jpg    # first photograph of each article
```

The photographs can be re-collected with `python scripts/fetch_images.py --out-root data/VnExpress`.
The article texts correspond to the text corpus of Chau et al. (IEEE Access, 2026,
doi:10.1109/ACCESS.2026.3711994), which BN_VN3S_MM extends.

## Reproducing the paper

```bash
# 1. text inside the photographs (once)
python scripts/run_ocr.py   --data-root data/VnExpress --out-root outputs

# 2. both models (GPU; every step is resumable)
python scripts/run_study.py --data-root data/VnExpress --out-root outputs --model qwen3vl8b \
       --steps check core attribution deletion ocr
python scripts/run_study.py --data-root data/VnExpress --out-root outputs --model internvl35_8b \
       --steps check core attribution deletion ocr

# 3. tables and figures (CPU)
python scripts/analyze.py   --data-root data/VnExpress --out-root outputs
python scripts/make_paper_figures.py --tables outputs/analysis/tables --out outputs/analysis/figures
```

The defaults are for a 96 GB GPU. On a 40 GB A100 use `--budget 24576 --attr-budget 4096`. Batches that
run out of memory are split automatically. For reference, on one RTX PRO 6000 (96 GB) the attribution
pass for InternVL3.5 took about 7 minutes for the 1,000 test articles in all three contexts, and the
deletion check about 30 minutes.

Details that matter for exact reproduction:

- Article text is truncated to 1,024 tokens, and the same truncated text is used in every condition.
- Qwen3-VL images are resized to about 448² pixels (≈196 visual tokens, native aspect ratio).
  InternVL3.5 uses one 448×448 tile (256 visual tokens) with dynamic tiling disabled, so masks align with
  the visual-token grid.
- The split is 80/10/10, stratified by topic, with seed 42. The other seeds are 13 for other-topic image
  donors, 7 for random deletions and 2027 for bootstrap.
- All 10,000 articles are used. That includes the seven redundant exact-duplicate Education texts, which
  matter only for supervised splits.

## Outputs

`run_study.py` writes `outputs/<model>/core/<condition>/part-*.parquet` (per-article label probabilities
and coverage), `attribution/<context>.pkl` (relevance per token and image cell) and
`deletion/part-*.parquet`. `analyze.py` writes `outputs/analysis/tables/T*.csv|tex` and
`outputs/analysis/figures/`.

## Copyright and license

The source code is released under the [MIT License](LICENSE). The license does not cover the VnExpress
articles or photographs referenced in `data/`. These remain the property of VnExpress and the
respective photographers and agencies, and they were used solely for non-commercial scientific research.

## Citation

The paper is under review at [ACIIDS 2027](https://aciids.pwr.edu.pl/2027/). Until it is published, please cite it as

```bibtex
@unpublished{chau2027nearparity,
  title  = {Near-Parity Hides Decision Churn: What Evidence Do Vision-Language Models Use to Classify Vietnamese News?},
  author = {Chau, Bao Nhan and Hoang, Anh},
  note   = {Under review at the Asian Conference on Intelligent Information and Database Systems (ACIIDS 2027)},
  year   = {2027},
  url    = {https://github.com/chaubaonhan/VLM-Evidence-VN-News}
}
```

## Contact

Bao Nhan Chau — 522h0093@student.tdtu.edu.vn · Anh Hoang (corresponding author) — anhh@ueh.edu.vn
