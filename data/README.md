# BN_VN3S_MM metadata

`bn_vn3s_mm_metadata.csv` describes the 10,000 VnExpress articles used in the study (ten topics,
1,000 articles each). It contains identifiers, labels, the data split, and the source URLs of each article
and of the photograph used. **It does not contain article texts or images.**

| column | meaning |
|---|---|
| `global_id` | `<Topic>_<source_index>`, the article id used in all result files |
| `category`, `label` | topic name and label index (order of `vlmstudy.core.CATS`) |
| `split` | `train` / `val` / `test` (80/10/10, stratified by topic, seed 42); RQ2 uses the 1,000 `test` articles |
| `article_id` | per-topic record id; image files are named `<article_id>_<k>.jpg` |
| `url` | URL of the article on vnexpress.net |
| `first_image_file`, `first_image_url` | file name and source URL of the first photograph (`_00`), the only image used |
| `num_images` | number of photographs published with the article |
| `text_chars` | length of the article text in characters (one Politics article has no body text) |
| `exact_duplicate_text` | `True` for the 11 rows in four groups of identical Education texts; all are kept, as in the paper |

## Why texts and images are not included

The article texts and photographs are copyrighted by VnExpress and, for agency images, by the respective
photographers and news agencies; we hold no license to redistribute them. The image set has not been
released separately. `scripts/fetch_images.py` re-collects the photographs from the URLs above; whoever
re-collects them is responsible for complying with the VnExpress terms of use. Some URLs may have changed
since collection, and the original acquisition window was not preserved in the data snapshot.

## Layout expected by the code

```
<data-root>/
  Automotive/
    data.parquet      # columns: url, label, id, source_index, text, image_files, image_urls, ...
    images/0001_00.jpg ...
  Business/
  ...
  World/
```

`scripts/build_metadata.py` regenerates this CSV from such a folder.
