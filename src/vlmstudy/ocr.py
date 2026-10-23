"""vlmstudy.ocr — two add-ons to vlmstudy (reuses its models, Study objects and stored results).

(A) TEXT INSIDE THE IMAGE. EasyOCR finds text regions in every photograph. On articles whose image
    contains text we score four more conditions and one text condition:
      full_ocrmask        article text + image with the text regions filled
      full_ocrrand        article text + image with boxes of the SAME sizes filled at random places
      image_only_ocrmask  image with the text regions filled (no article text)
      image_only_ocrrand  image with same-size random boxes filled
      ocr_only            only the string read from the image (no article text, no image)
    -> is the model reading the words in the photo, or the scene?

(B) QUALITATIVE EXAMPLES. Exports a few test articles (small thumbnail, image heatmaps of both models,
    word relevance, predictions under every condition) to one small JSON for the paper figure.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import core as V

OCR_CONDS = ["full_ocrmask", "full_ocrrand", "image_only_ocrmask", "image_only_ocrrand", "ocr_only"]
OCR_HEADER = "(Chữ nhận dạng được trong ảnh, không phải nội dung bài báo) "


# ============================================================================ OCR
def run_ocr(df, ids, path, langs=("vi", "en"), gpu=True, reader=None, max_side=1280):
    """Detect + read text in the first image of each article (resumable jsonl)."""
    done = set(load_ocr(path))
    todo = [g for g in ids if g not in done]
    if not todo:
        V.log("OCR: all done")
        return
    if reader is None:
        import easyocr

        reader = easyocr.Reader(list(langs), gpu=gpu)
    dfi = df.set_index("global_id")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for i, g in enumerate(todo):
            img = Image.open(dfi.at[g, "image_path"]).convert("RGB")
            img.thumbnail((max_side, max_side))
            W, H = img.size
            res = reader.readtext(np.asarray(img))
            boxes = [{"poly": [[float(x) / W, float(y) / H] for x, y in pts], "text": str(t), "conf": float(c)}
                     for pts, t, c in res]
            f.write(json.dumps({"global_id": g, "boxes": boxes}, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                f.flush()
                V.log(f"OCR {i}/{len(todo)}")
    V.log(f"OCR finished: {len(todo)} images")


def load_ocr(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["global_id"]] = r
    return out


def text_boxes(rec, min_conf=0.3, min_chars=2, pad=0.01):
    """Axis-aligned normalised boxes [x1,y1,x2,y2] of confident text regions (slightly padded)."""
    out, texts = [], []
    for b in (rec or {}).get("boxes", []):
        if b["conf"] < min_conf or len(b["text"].strip()) < min_chars:
            continue
        xs, ys = [p[0] for p in b["poly"]], [p[1] for p in b["poly"]]
        out.append([max(0.0, min(xs) - pad), max(0.0, min(ys) - pad), min(1.0, max(xs) + pad), min(1.0, max(ys) + pad)])
        texts.append(b["text"].strip())
    return out, texts


def _boxes_mask(boxes, size):
    W, H = size
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for x1, y1, x2, y2 in boxes:
        d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], fill=255)
    return np.asarray(m) > 0


def _random_boxes(g, boxes, size, tries=80):
    """Same-size boxes moved to random places, avoiding the real text regions as far as possible."""
    rng = np.random.default_rng(int(hashlib.sha1(f"ocrrand|{g}".encode()).hexdigest()[:12], 16))
    avoid = _boxes_mask(boxes, size)
    out = []
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        best, best_ov = None, 2.0
        for _ in range(tries):
            nx, ny = rng.uniform(0, max(1e-6, 1 - w)), rng.uniform(0, max(1e-6, 1 - h))
            cand = [nx, ny, nx + w, ny + h]
            m = _boxes_mask([cand], size)
            ov = (m & avoid).sum() / max(1, m.sum())
            if ov < best_ov:
                best, best_ov = cand, ov
            if ov == 0:
                break
        out.append(best)
    return out


def _fill(img, mask):
    a = np.asarray(img).copy()
    a[mask] = V.FILL_RGB
    return Image.fromarray(a)


def build_ocr(st, cond, g, ocr, min_conf=0.3):
    boxes, texts = text_boxes(ocr.get(g), min_conf)
    meta = {"global_id": g, "condition": cond}
    if not boxes:
        return None, meta
    if cond == "ocr_only":
        return {"text": OCR_HEADER + " ".join(texts), "image": None}, meta
    img = st._img(g)
    if cond.endswith("ocrrand"):
        boxes = _random_boxes(g, boxes, img.size)
    m = _boxes_mask(boxes, img.size)
    meta["mask_area"] = float(m.mean())
    text = st.info(g).text if cond.startswith("full") else ""
    return {"text": text, "image": _fill(img, m)}, meta


def run_ocr_conditions(st, ids, ocr, conds=OCR_CONDS, min_conf=0.3):
    st._infos(ids)
    with_text = [g for g in ids if text_boxes(ocr.get(g), min_conf)[0]]
    V.log(f"[{st.vlm.name}] {len(with_text)}/{len(ids)} images contain text (conf>={min_conf})")
    for cond in conds:
        w = V.PartWriter(os.path.join(st.out, "core", cond), ["global_id"])
        done = {k[0] for k in w.done()}
        todo = [g for g in with_text if g not in done]
        if not todo:
            V.log(f"[{st.vlm.name}] {cond}: done already")
            continue

        def est(g, cond=cond):
            if cond == "ocr_only":
                return st.vlm.est_tokens(len(" ".join(text_boxes(ocr.get(g), min_conf)[1])) // 2 + 20, False)
            return st.vlm.est_tokens(st.info(g).ntok if cond.startswith("full") else 0, True)

        specs = [(g, est(g), (lambda g=g, cond=cond: build_ocr(st, cond, g, ocr, min_conf))) for g in todo]

        def on(m, r, w=w):
            p, c = r
            w.add({**m, "pred": int(p.argmax()), "coverage": float(c), **{f"p{k}": float(p[k]) for k in range(10)}})

        V.run_specs(st.vlm, specs, on, st.budget, st.cap, label=f"[{st.vlm.name}] {cond}")
        w.flush()


# ============================================================================ OCR analysis
def ocr_analysis(studies, df, ocr, out_dir, B=10000, seed=2027, min_conf=0.3):
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    man = df.set_index("global_id")
    rows_eff, rows_acc, rows_cat = [], [], []
    for st in studies:
        m = st.vlm.name
        C = {c: d for c in V.CORE_CONDITIONS + OCR_CONDS if (d := st.core(c)) is not None}
        if "full_ocrmask" not in C:
            continue
        all_ids = C["full"].index
        has = pd.Series([bool(text_boxes(ocr.get(g), min_conf)[0]) for g in all_ids], index=all_ids)
        ids = [g for g in C["full_ocrmask"].index if g in C["full_ocrrand"].index and g in C["image_only_ocrmask"].index
               and g in C["image_only_ocrrand"].index]
        y = man.loc[ids, "label"].values
        P = lambda c, I: C[c].loc[I, [f"p{k}" for k in range(10)]].values.astype(float)
        # --- effect of removing the text in the image vs removing same-size random regions
        for ctx in ("full", "image_only"):
            ref = P(ctx, ids)
            yh = ref.argmax(1)
            r = np.arange(len(ids))
            out = {}
            for kind in ("ocrmask", "ocrrand"):
                Q = P(f"{ctx}_{kind}", ids)
                out[kind] = dict(dp=ref[r, yh] - Q[r, yh], flip=(Q.argmax(1) != yh).astype(float),
                                 acc=(Q.argmax(1) == y).astype(float),
                                 area=C[f"{ctx}_{kind}"].loc[ids, "mask_area"].values)
            e = out["ocrmask"]["dp"] - out["ocrrand"]["dp"]
            fe = out["ocrmask"]["flip"] - out["ocrrand"]["flip"]
            be, bf = V.boot_mean(e, B, seed), V.boot_mean(fe, B, seed)
            ae = out["ocrmask"]["acc"] - out["ocrrand"]["acc"]
            ba = V.boot_mean(ae, B, seed)
            rows_eff.append(dict(model=m, context=ctx, n=len(ids), acc_ref=float((yh == y).mean()),
                                 acc_textmask=out["ocrmask"]["acc"].mean(), acc_randmask=out["ocrrand"]["acc"].mean(),
                                 acc_diff=ae.mean(), acc_lo=V.ci(ba)[0], acc_hi=V.ci(ba)[1],
                                 flip_textmask=out["ocrmask"]["flip"].mean(), flip_randmask=out["ocrrand"]["flip"].mean(),
                                 flip_diff=fe.mean(), flip_lo=V.ci(bf)[0], flip_hi=V.ci(bf)[1],
                                 dp_textmask=out["ocrmask"]["dp"].mean(), dp_randmask=out["ocrrand"]["dp"].mean(),
                                 effect=e.mean(), lo=V.ci(be)[0], hi=V.ci(be)[1], p=V.pval(be),
                                 area_textmask=out["ocrmask"]["area"].mean(), area_randmask=out["ocrrand"]["area"].mean()))
        # --- accuracies on the articles whose image contains text
        row = dict(model=m, n_all=len(all_ids), n_with_text=int(has.sum()), share_with_text=float(has.mean()))
        for c in ("full", "text_only", "image_only", "full_ocrmask", "image_only_ocrmask", "image_only_ocrrand", "ocr_only"):
            if c in C:
                I = [g for g in ids if g in C[c].index]
                row[f"acc_{c}"] = float((C[c].loc[I, "pred"].values == man.loc[I, "label"].values).mean())
        for flag, name in ((True, "with_text"), (False, "no_text")):
            I = has.index[has.values == flag]
            yy = man.loc[I, "label"].values
            row[f"acc_image_only_{name}"] = float((C["image_only"].loc[I, "pred"].values == yy).mean())
            row[f"acc_full_{name}"] = float((C["full"].loc[I, "pred"].values == yy).mean())
            row[f"acc_text_only_{name}"] = float((C["text_only"].loc[I, "pred"].values == yy).mean())
        rows_acc.append(row)
        # --- per topic
        for cat in V.CATS:
            I = [g for g in ids if man.at[g, "category"] == cat]
            n_all = int((man.loc[all_ids, "category"] == cat).sum())
            if not I:
                rows_cat.append(dict(model=m, category=cat, share_with_text=0.0, n=0))
                continue
            yy = man.loc[I, "label"].values
            rows_cat.append(dict(model=m, category=cat, n=len(I), share_with_text=len(I) / max(n_all, 1),
                                 acc_image_only=float((C["image_only"].loc[I, "pred"].values == yy).mean()),
                                 acc_image_only_textmask=float((C["image_only_ocrmask"].loc[I, "pred"].values == yy).mean()),
                                 acc_image_only_randmask=float((C["image_only_ocrrand"].loc[I, "pred"].values == yy).mean()),
                                 acc_ocr_only=float((C["ocr_only"].loc[I, "pred"].values == yy).mean()) if "ocr_only" in C else np.nan))
    T = {"T9_text_in_image_effect": pd.DataFrame(rows_eff), "T10_text_in_image_accuracy": pd.DataFrame(rows_acc),
         "T11_text_in_image_by_topic": pd.DataFrame(rows_cat)}
    for k, t in T.items():
        t.to_csv(os.path.join(out_dir, "tables", f"{k}.csv"), index=False)
    return T


# ============================================================================ qualitative examples
def _thumb_b64(img, side=256, q=72):
    im = img.convert("RGB").copy()
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode(), list(im.size)


def export_examples(studies, df, out_path, ocr=None, per_group=2, n_words=70, seed=0):
    """Pick illustrative TEST articles and save everything the paper figure needs in one small JSON.

    Groups (decided on the first study = main model):
      image_fixes    text-only wrong, text+image right
      image_breaks   text-only right, text+image wrong
      captured       an image from another topic pulls the prediction to that topic
      text_in_image  image contains text and masking it changes the image-only prediction
    """
    main = studies[0]
    man = df.set_index("global_id")
    Cm = {c: main.core(c) for c in V.CORE_CONDITIONS + OCR_CONDS if main.core(c) is not None}
    Am = main.attribution("full")
    test = [g for g in Am if all(g in s.attribution("full") for s in studies)]
    y = man.loc[test, "label"].values
    pred = lambda c, I: Cm[c].loc[I, "pred"].values
    pf, pt = pred("full", test), pred("text_only", test)
    pfull = Cm["full"].loc[test, [f"p{k}" for k in range(10)]].values
    conf = pfull[np.arange(len(test)), y]
    groups = {
        "image_fixes": [test[i] for i in np.argsort(-conf) if pf[i] == y[i] and pt[i] != y[i]],
        "image_breaks": [test[i] for i in np.argsort(-pfull.max(1)) if pf[i] != y[i] and pt[i] == y[i]],
    }
    if "swap_diff" in Cm:
        sw = Cm["swap_diff"].reindex(test)
        cap = [g for g, r in zip(test, sw.itertuples()) if r.pred == r.pred and
               int(r.pred) == int(man.at[r.donor_id, "label"]) and int(r.pred) != int(man.at[g, "label"])]
        groups["captured"] = cap
    if "image_only_ocrmask" in Cm:
        io_ = Cm["image_only"]
        om = Cm["image_only_ocrmask"]
        cand = [g for g in test if g in om.index and io_.at[g, "pred"] == man.at[g, "label"] and om.at[g, "pred"] != man.at[g, "label"]]
        groups["text_in_image"] = cand
    picked = []
    for k, lst in groups.items():
        for g in lst[:per_group]:
            picked.append((k, g))
    V.log("example groups:", {k: len(v) for k, v in groups.items()})
    out = []
    for group, g in picked:
        rec = {"group": group, "global_id": g, "category": man.at[g, "category"], "models": {}}
        img = Image.open(man.at[g, "image_path"])
        rec["image_b64"], rec["image_size"] = _thumb_b64(img)
        if group == "captured":
            d = Cm["swap_diff"].at[g, "donor_id"]
            rec["donor_id"], rec["donor_category"] = d, man.at[d, "category"]
            rec["donor_b64"], _ = _thumb_b64(Image.open(man.at[d, "image_path"]))
        if ocr is not None:
            bx, tx = text_boxes(ocr.get(g))
            rec["ocr_boxes"], rec["ocr_text"] = bx, tx
        for s in studies:
            A = s.attribution("full").get(g)
            mrec = {"preds": {}, "probs": {}}
            for c in V.CORE_CONDITIONS + OCR_CONDS:
                d = s.core(c)
                if d is not None and g in d.index:
                    pv = d.loc[g, [f"p{k}" for k in range(10)]].values.astype(float)
                    mrec["preds"][c] = V.CATS[int(pv.argmax())]
                    mrec["probs"][c] = [round(float(v), 4) for v in pv]
            if A:
                mrec["img_map"] = np.round(A["img_map"] / (A["img_map"].max() + 1e-12), 3).tolist() if "img_map" in A else None
                mrec["image_share"] = A["R_img"] / (A["R_img"] + A["R_text"] + 1e-12)
                mrec["top_words"] = A.get("top_words", [])[:12]
                if "word_rel" in A:
                    wr = np.asarray(A["word_rel"])
                    mrec["word_rel_first"] = np.round(wr[:n_words] / (wr.max() + 1e-12), 3).tolist()
            rec["models"][s.vlm.name] = mrec
        text = man.at[g, "text"]
        rec["words_first"] = text.split()[:n_words]
        out.append(rec)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    V.log(f"saved {len(out)} examples -> {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")
    return out