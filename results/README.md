# Results

Aggregate tables produced by `scripts/analyze.py` from the runs reported in the paper
(Qwen3-VL-8B-Instruct and InternVL3.5-8B-HF, bfloat16, one NVIDIA RTX PRO 6000 Blackwell 96 GB,
PyTorch 2.11, Transformers 5.18.0). Confidence intervals are 95% percentile intervals from 10,000 paired
bootstrap resamples of articles; `p_holm` is Holm-adjusted within each family.

| file | content | paper |
|---|---|---|
| `tables/T1_performance.csv` | Macro-F1 and accuracy per input condition, paired difference to text+image | Table 1 |
| `tables/T2_change_vs_full.csv` | label flips, probability shift, JS divergence, mismatched-image capture | Table 1, Sect. 5.1 |
| `tables/T3_complementarity.csv` | gain / loss / net effect of the image, oracle, agreement cells | Table 2 |
| `tables/T4_by_text_confidence.csv` | effect of the image by quintile of text-only confidence | Fig. 1 |
| `tables/T5_by_category.csv` | per-topic accuracy and effect of the image | Fig. 2 |
| `tables/T6_where_it_looks.csv` | gradient x input relevance shares, Spearman with the effect of removing the image | Table 3 |
| `tables/T7_deletion_check.csv` | deletion of most relevant vs random words / image cells | Fig. 4 |
| `tables/T8_top_words.csv` | topic-specific words by relevance (whitespace syllables) | Table 4 |
| `tables/T9_text_in_image_effect.csv` | masking printed text vs equal-area random masks | Table 5 |
| `tables/T10_text_in_image_accuracy.csv` | accuracy with / without printed text, OCR-only | Table 5 |
| `tables/T11_text_in_image_by_topic.csv` | text-in-image results per topic | Sect. 5.4 |

`figures/` holds Figs. 1, 2 and 4 of the paper, redrawn from these tables with
`python scripts/make_paper_figures.py`. Fig. 3 shows VnExpress photographs and is not included.

Model values are fractions (multiply by 100 for percentages) unless a column is already formatted.
