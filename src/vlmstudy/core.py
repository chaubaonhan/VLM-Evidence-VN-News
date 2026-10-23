"""vlmstudy.core — What evidence does a VLM actually use to classify Vietnamese news? (BN_VN3S_MM)

RQ1  Does the prediction change?   same article, fixed model + prompt, inputs varied:
     full (text+image) | text_only | image_only | text_gray (text + blank image) | swap_diff
     (text + image of an article from another topic).
RQ2  Where does the model look?    one backward pass per article gives gradient×input relevance
     of every LLM input token (article words, image tokens incl. DeepStack features, prompt);
     -> text-vs-image share, top words, image heatmap. Verified by deletion: removing the top-k%
     words / image cells vs removing the same amount at random.
RQ3  How do text and image complement each other?  gain/loss of fusion over text-only,
     oracle headroom, dependence on text confidence, topic, swap capture.

All inference is deterministic (constrained next-token scoring over the 10 label tokens).
"""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import math
import os
import pickle
import queue
import re
import shutil
import threading
import time
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image

CATS = ["Automotive", "Business", "Education", "Entertainment", "Health",
        "Law", "Lifestyle", "Politics", "Sports", "World"]
GLOSS = {
    "Automotive": "Xe – ô tô, xe máy, thị trường xe, giao thông",
    "Business": "Kinh doanh – doanh nghiệp, tài chính, chứng khoán, thị trường",
    "Education": "Giáo dục – trường học, thi cử, tuyển sinh, du học",
    "Entertainment": "Giải trí – nghệ sĩ, phim ảnh, âm nhạc, hoa hậu, thời trang",
    "Health": "Sức khỏe – bệnh tật, y tế, dinh dưỡng, chăm sóc sức khỏe",
    "Law": "Pháp luật – vụ án, điều tra, xét xử, tư vấn pháp lý",
    "Lifestyle": "Đời sống – gia đình, quan hệ, nhịp sống, câu chuyện đời thường",
    "Politics": "Thời sự – chính trị, chính sách, hoạt động nhà nước, sự kiện trong nước",
    "Sports": "Thể thao – bóng đá, giải đấu, vận động viên",
    "World": "Thế giới – tin quốc tế, quan hệ giữa các nước",
}
# exact-duplicate texts removed in the ICCIES/UGAF audit (9,993 articles remain)
DROP_IDS = ["Education_0570", "Education_0587", "Education_0761", "Education_0765",
            "Education_0861", "Education_0899", "Education_0976"]
FILL_RGB = (124, 116, 104)  # ImageNet mean: used for blank images and deleted regions
CORE_CONDITIONS = ["full", "text_only", "image_only", "text_gray", "swap_diff"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================ data
_ROLES = {
    "text": ["text", "content", "body", "body_text", "article", "article_text", "noi_dung", "full_text"],
    "title": ["title", "headline", "tieu_de"],
    "index": ["source_index", "index", "idx", "id", "article_id", "stt"],
    "images": ["image_filenames", "image_files", "image_names", "images", "image_paths_resolved",
               "image_paths", "image_path", "image_file", "image"],
}


def _as_list(v) -> list:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return [x for x in list(v) if x is not None]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                return list(ast.literal_eval(s))
            except Exception:
                pass
        return [p for p in re.split(r"[;|,]\s*", s) if p]
    return [v]


def detect_columns(df, text_col=None):
    low = {c.lower(): c for c in df.columns}
    out = {}
    for role, cands in _ROLES.items():
        for c in cands:
            if c in low:
                out[role] = low[c]
                break
    if "text" not in out:
        lens = {c: df[c].astype(str).str.len().mean() for c in df.columns if df[c].dtype == object}
        out["text"] = max(lens, key=lens.get)
    if text_col:
        out["text"] = text_col
    return out


def inspect_schema(data_root, cat="Health"):
    df = pd.read_parquet(os.path.join(data_root, cat, "data.parquet"))
    print(f"{cat}: {len(df)} rows | columns: {list(df.columns)}")
    print("detected:", detect_columns(df))
    for c in df.columns:
        print(f"  {c:>22}: {repr(df[c].iloc[0])[:110]}")


def stage_data(drive_root, local_root, suffix="_00"):
    """Copy parquet + first images Drive -> local disk via ONE zip (Drive FUSE is slow on many files)."""
    if os.path.exists(os.path.join(local_root, ".staged")):
        log("already staged:", local_root)
        return local_root
    zpath = os.path.join(os.path.dirname(drive_root.rstrip("/")), "BN_VN3S_MM_first_images.zip")
    if not os.path.exists(zpath):
        log("building", zpath, "(one-off, ~10-20 min)")
        tmp = "/tmp/_first_images.zip"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for cat in CATS:
                z.write(os.path.join(drive_root, cat, "data.parquet"), f"{cat}/data.parquet")
                for fn in sorted(os.listdir(os.path.join(drive_root, cat, "images"))):
                    if suffix + "." in fn:
                        z.write(os.path.join(drive_root, cat, "images", fn), f"{cat}/images/{fn}")
        shutil.copy(tmp, zpath)
    log("copy + extract ->", local_root)
    shutil.copy(zpath, "/tmp/_stage.zip")
    os.makedirs(local_root, exist_ok=True)
    with zipfile.ZipFile("/tmp/_stage.zip") as z:
        z.extractall(local_root)
    open(os.path.join(local_root, ".staged"), "w").write("ok")
    return local_root


def _first_image(img_dir, idx, names, suffix):
    for n in names:
        b = os.path.basename(str(n))
        if suffix in b and os.path.exists(os.path.join(img_dir, b)):
            return os.path.join(img_dir, b)
    hits = sorted(glob.glob(os.path.join(img_dir, f"{idx:04d}{suffix}.*")))
    if hits:
        return hits[0]
    if names and os.path.exists(os.path.join(img_dir, os.path.basename(str(names[0])))):
        return os.path.join(img_dir, os.path.basename(str(names[0])))
    return None


def load_corpus(data_root, text_col=None, suffix="_00"):
    rows = []
    for li, cat in enumerate(CATS):
        df = pd.read_parquet(os.path.join(data_root, cat, "data.parquet"))
        cols = detect_columns(df, text_col)
        img_dir = os.path.join(data_root, cat, "images")
        for ri, (_, r) in enumerate(df.iterrows()):
            names = _as_list(r[cols["images"]]) if "images" in cols else []
            idx = ri
            if "index" in cols:
                try:
                    idx = int(r[cols["index"]])
                except Exception:
                    pass
            else:
                for n in names:
                    m = re.match(r"(\d+)_\d+", os.path.basename(str(n)))
                    if m:
                        idx = int(m.group(1))
                        break
            t = r[cols["text"]]
            t = "" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t).strip()
            rows.append(dict(global_id=f"{cat}_{idx:04d}", category=cat, label=li, text=t,
                             image_path=_first_image(img_dir, idx, names, suffix)))
    df = pd.DataFrame(rows)
    df = df[~df.global_id.isin(DROP_IDS)].reset_index(drop=True)
    assert not df.global_id.duplicated().any(), "duplicate ids"
    miss = df.image_path.isna().sum()
    log(f"corpus: {len(df)} articles | missing image: {miss} | empty text: {(df.text == '').sum()}")
    return df


def assign_split(df, seed=42, ref_npz=None):
    """ICCIES split (stratified 80/10/10, seed 42); test ids synced with the UGAF predictions if found."""
    from sklearn.model_selection import train_test_split

    df = df.copy()
    tr, tmp = train_test_split(df.index.values, test_size=0.2, stratify=df.label.values, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=df.label.values[tmp], random_state=seed)
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"
    if ref_npz and os.path.exists(ref_npz):
        ref = set(map(str, np.load(ref_npz, allow_pickle=True)["global_id"].tolist()))
        ours = set(df.global_id[df.split == "test"])
        log(f"test split vs ICCIES reference: overlap {len(ref & ours)}/{len(ref)}")
        if ref != ours and ref <= set(df.global_id):
            df.loc[df.split == "test", "split"] = "train"
            df.loc[df.global_id.isin(ref), "split"] = "test"
            log("-> adopted the ICCIES test ids")
    log("split sizes:", df.split.value_counts().to_dict())
    return df


def make_donors(df, seed=13):
    """swap_diff donor: image of an article from a DIFFERENT topic (deterministic)."""
    rng = np.random.default_rng(seed)
    cats = df.category.values
    ids = df.global_id.values
    donor = [ids[rng.choice(np.where(cats != c)[0])] for c in cats]
    return dict(zip(ids, donor))


# ============================================================================ prompt
NO_TEXT = "(Không có văn bản)"


def classify_prompt(text):
    body = text if (text and text.strip()) else NO_TEXT
    labels = "\n".join(f"- {c}: {GLOSS[c]}" for c in CATS)
    return ("Bạn là hệ thống phân loại chủ đề tin tức tiếng Việt. Hãy xác định chủ đề của bài báo "
            "dựa trên văn bản bài báo và ảnh minh họa đi kèm (nếu có).\n\n"
            f"Danh sách nhãn:\n{labels}\n\n"
            f"Văn bản bài báo:\n\"\"\"\n{body}\n\"\"\"\n\n"
            "Chỉ trả lời bằng đúng một nhãn tiếng Anh trong danh sách, không giải thích.")


# ============================================================================ model
class TextInfo:
    """Truncated article text + its standalone tokenisation + token->word map."""
    __slots__ = ("text", "ids", "tok2word", "spans", "ntok")

    def __init__(self, text, ids, tok2word, spans):
        self.text, self.ids, self.tok2word, self.spans = text, ids, tok2word, spans
        self.ntok = len(ids)

    @property
    def words(self):
        return [self.text[a:b] for a, b in self.spans]


class VLM:
    """Constrained-scoring wrapper for Qwen3-VL ('qwen3vl') and InternVL3.5-HF ('internvl')."""

    def __init__(self, kind, model, processor, image_area=448 * 448, max_text_tokens=1024, name=None):
        import torch

        self.torch, self.kind, self.model, self.processor = torch, kind, model, processor
        self.name = name or kind
        self.tok = processor.tokenizer
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.max_text_tokens = max_text_tokens
        self.image_area = image_area
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.image_token_id = int(processor.image_token_id)
        self.lm = model.model.language_model
        if kind == "qwen3vl":
            ip = processor.image_processor
            self.patch, self.merge = int(ip.patch_size), int(ip.merge_size)
            lo, hi = (self.patch * self.merge) ** 2 * 4, 4096 * 4096
            ip.size = {"shortest_edge": lo, "longest_edge": hi}
            ip.min_pixels, ip.max_pixels = lo, hi  # prepared images are never resized again
            self.proc_kw = {}
        elif kind == "internvl":
            vc = model.config.vision_config
            isz = vc.image_size[0] if isinstance(vc.image_size, (list, tuple)) else vc.image_size
            psz = vc.patch_size[0] if isinstance(vc.patch_size, (list, tuple)) else vc.patch_size
            self.side = int(isz)
            self.cells = int(isz // psz * model.config.downsample_ratio)
            self.proc_kw = {"crop_to_patches": False}
        else:
            raise ValueError(kind)
        ids = [self.tok.encode(c, add_special_tokens=False)[0] for c in CATS]
        if len(set(ids)) != len(ids):
            raise ValueError(f"label first tokens collide: {ids}")
        self.label_ids = ids
        self._overhead = len(self.tok.encode(self.render("", False), add_special_tokens=False))
        self._img_tokens = (image_area // (self.patch * self.merge) ** 2) if kind == "qwen3vl" else self.cells ** 2

    @classmethod
    def load(cls, kind, checkpoint, revision=None, dtype="bfloat16", **kw):
        import torch
        import transformers

        torch.backends.cuda.matmul.allow_tf32 = True
        mcls = {"qwen3vl": "Qwen3VLForConditionalGeneration", "internvl": "InternVLForConditionalGeneration"}[kind]
        model = getattr(transformers, mcls).from_pretrained(
            checkpoint, revision=revision, dtype=getattr(torch, dtype), attn_implementation="sdpa",
            device_map="cuda" if torch.cuda.is_available() else "cpu").eval()
        model.requires_grad_(False)
        proc = transformers.AutoProcessor.from_pretrained(checkpoint, revision=revision)
        vlm = cls(kind, model, proc, name=kw.pop("name", None), **kw)
        vlm.commit = getattr(model.config, "_commit_hash", None)
        log(f"loaded {checkpoint} (commit {vlm.commit}) on {vlm.device}")
        return vlm

    # ---------------------------------------------------------------- geometry
    def prepare_image(self, img):
        img = img.convert("RGB")
        if self.kind == "internvl":
            return img.resize((self.side, self.side), Image.BICUBIC)
        f = self.patch * self.merge
        w, h = img.size
        s = math.sqrt(self.image_area / (w * h))
        nw, nh = max(f, round(w * s / f) * f), max(f, round(h * s / f) * f)
        while nw * nh > self.image_area * 1.15:
            if nw >= nh:
                nw -= f
            else:
                nh -= f
        return img.resize((nw, nh), Image.BICUBIC)

    def grid(self, img):
        if self.kind == "internvl":
            return self.cells, self.cells
        f = self.patch * self.merge
        return img.size[1] // f, img.size[0] // f

    # ---------------------------------------------------------------- text
    def text_info(self, text):
        text = text or ""
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) > self.max_text_tokens:
            text = self.tok.decode(ids[:self.max_text_tokens], skip_special_tokens=True)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        starts = np.array([a for a, _ in spans]) if spans else np.zeros(0, int)
        t2w = np.full(len(enc["input_ids"]), -1)
        for k, (a, b) in enumerate(enc["offset_mapping"]):
            if b <= a or not spans:
                continue
            j = int(np.searchsorted(starts, b - 1, side="right") - 1)
            if j >= 0 and spans[j][0] <= b - 1 < spans[j][1]:
                t2w[k] = j
        return TextInfo(text, np.array(enc["input_ids"]), t2w, spans)

    def render(self, text, has_image):
        content = ([{"type": "image"}] if has_image else []) + [{"type": "text", "text": classify_prompt(text)}]
        return self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                  add_generation_prompt=True)

    def est_tokens(self, ntok, has_image):
        return self._overhead + ntok + (self._img_tokens if has_image else 0)

    # ---------------------------------------------------------------- encode / score
    def encode(self, items):
        imgs = [it["image"] for it in items if it["image"] is not None]
        assert not imgs or len(imgs) == len(items), "batch must be all-image or all-text"
        kw = dict(text=[self.render(it["text"], it["image"] is not None) for it in items],
                  padding=True, return_tensors="pt")
        if imgs:
            kw["images"] = imgs
            kw.update(self.proc_kw)
        enc = self.processor(**kw)
        if imgs and self.kind == "qwen3vl":
            for (t, gh, gw), im in zip(enc["image_grid_thw"].tolist(), imgs):
                assert (gh * self.patch, gw * self.patch) == (im.size[1], im.size[0]), "processor resized image"
        if imgs and self.kind == "internvl":
            assert enc["pixel_values"].shape[0] == len(imgs), "unexpected tiling"
        return enc

    def _dev(self, enc):
        enc.pop("token_type_ids", None)
        enc = enc.to(self.device)
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
        return enc

    def score(self, enc):
        torch = self.torch
        with torch.inference_mode():
            out = self.model(**self._dev(enc), logits_to_keep=1, use_cache=False)
            lg = out.logits[:, -1, :].float()
            P = torch.softmax(lg[:, self.label_ids], -1)
            cov = torch.softmax(lg, -1)[:, self.label_ids].sum(-1)
        return P.cpu().numpy(), cov.cpu().numpy()

    def generate(self, items, max_new_tokens=8):
        torch = self.torch
        with torch.inference_mode():
            enc = self._dev(self.encode(items))
            g = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # ---------------------------------------------------------------- attribution
    def attribute(self, enc, items, targets):
        """Gradient×input relevance of every LLM input vector for log p(target).

        Leaves: the merged input embeddings (text-token embeddings + projected image tokens) and,
        for Qwen3-VL, the DeepStack visual features injected into the first decoder layers.
        Relevance of a position = |sum over its input vectors of <grad, input>|.
        Samples in a batch are independent, so one backward of sum_i log p_i gives per-sample grads.
        """
        torch = self.torch
        enc = self._dev(enc)
        st = {}

        def hook(mod, args, kwargs):
            e = kwargs["inputs_embeds"].detach().requires_grad_(True)
            kwargs["inputs_embeds"] = e
            st["e"] = e
            ds = kwargs.get("deepstack_visual_embeds")
            if ds is not None:
                ds = [d.detach().requires_grad_(True) for d in ds]
                kwargs["deepstack_visual_embeds"] = ds
                st["ds"], st["vm"] = ds, kwargs["visual_pos_masks"]
            return args, kwargs

        h = self.lm.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.enable_grad():
                out = self.model(**enc, logits_to_keep=1, use_cache=False)
                lp = torch.log_softmax(out.logits[:, -1, :][:, self.label_ids].float(), -1)
                tg = torch.as_tensor([int(t) for t in targets], device=lp.device)
                lp.gather(1, tg[:, None]).sum().backward()
        finally:
            h.remove()
        e = st["e"]
        r = (e.grad.float() * e.detach().float()).sum(-1)
        if "ds" in st:
            vm = st["vm"]
            for d in st["ds"]:
                r[vm] = r[vm] + (d.grad.float() * d.detach().float()).sum(-1)
        R = r.abs().cpu().numpy()
        ids = enc["input_ids"].cpu().numpy()
        am = enc["attention_mask"].cpu().numpy().astype(bool)
        del out, e, st
        res = []
        for i, it in enumerate(items):
            valid = am[i].copy()
            rec = {"target": int(targets[i])}
            ipos = np.where(ids[i] == self.image_token_id)[0]
            if it["image"] is not None:
                gh, gw = self.grid(it["image"])
                assert len(ipos) == gh * gw, (len(ipos), gh, gw)
                rec["img_map"] = R[i, ipos].reshape(gh, gw).astype(np.float32)
            rec["R_img"] = float(R[i, ipos].sum())
            rec["n_img_tok"] = int(len(ipos))
            valid[ipos] = False
            info = it.get("info")
            rec["R_text"], rec["n_text_tok"], rec["text_found"] = 0.0, 0, False
            if info is not None and info.ntok > 0:
                pos = find_subseq(ids[i], info.ids)
                if pos is not None:
                    tr = R[i, pos:pos + info.ntok]
                    wr = np.zeros(len(info.spans), np.float32)
                    m = info.tok2word >= 0
                    np.add.at(wr, info.tok2word[m], tr[m])
                    rec.update(word_rel=wr, R_text=float(tr.sum()), n_text_tok=info.ntok, text_found=True)
                    agg = {}
                    for j, (a0, b0) in enumerate(info.spans):  # aggregate occurrences of the same word
                        wd = re.sub(r"^\W+|\W+$", "", info.text[a0:b0].lower())
                        if wd:
                            agg[wd] = agg.get(wd, 0.0) + float(wr[j])
                    rec["top_words"] = sorted(agg.items(), key=lambda kv: -kv[1])[:25]
                    valid[pos:pos + info.ntok] = False
            rec["R_prompt"] = float(R[i, valid].sum())
            res.append(rec)
        return res


def find_subseq(row, sub):
    """Start index of `sub` inside `row` (exact), tolerant to a merged first/last token."""
    n = len(sub)
    if n == 0:
        return None
    for trim in (0, 1, 2):
        s = sub[trim:n - trim] if trim else sub
        if len(s) == 0:
            break
        cand = np.where(row == s[0])[0]
        for c in cand:
            if c + len(s) <= len(row) and np.array_equal(row[c:c + len(s)], s):
                st = c - trim
                if st >= 0 and st + n <= len(row):
                    return int(st)
    return None


# ============================================================================ throughput
class PartWriter:
    def __init__(self, d, keys, every=512):
        self.d, self.keys, self.every, self.buf = d, keys, every, []
        os.makedirs(d, exist_ok=True)
        self.n = len(glob.glob(os.path.join(d, "part-*.parquet")))

    def done(self):
        df = read_parts(self.d)
        return set() if df is None else set(map(tuple, df[self.keys].astype(str).values.tolist()))

    def add(self, row):
        self.buf.append(row)
        if len(self.buf) >= self.every:
            self.flush()

    def flush(self):
        if self.buf:
            p = os.path.join(self.d, f"part-{self.n:05d}.parquet")
            pd.DataFrame(self.buf).to_parquet(p + ".tmp", index=False)
            os.replace(p + ".tmp", p)
            self.n += 1
            self.buf = []


def read_parts(d):
    fs = sorted(glob.glob(os.path.join(d, "part-*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None


def make_batches(specs, budget, cap):
    specs = sorted(specs, key=lambda s: -s[1])
    out, cur, mx = [], [], 0
    for s in specs:
        m = max(mx, s[1])
        if cur and (m * (len(cur) + 1) > budget or len(cur) >= cap):
            out.append(cur)
            cur, m = [], s[1]
        cur.append(s)
        mx = m
    if cur:
        out.append(cur)
    return out


def run_specs(vlm, specs, on_result, budget, cap, mode="score", prefetch=3, every=2000, label=""):
    """specs: (key, est_tokens, build) with build() -> (item, meta). CPU producer thread overlaps the GPU."""
    batches = make_batches(specs, budget, cap)
    q, err = queue.Queue(maxsize=prefetch), []

    def producer():
        try:
            for b in batches:
                groups = {}
                for _, _, build in b:
                    it, meta = build()
                    if it is not None:
                        groups.setdefault(it["image"] is None, []).append((it, meta))
                for g in groups.values():
                    its = [x[0] for x in g]
                    q.put((its, [x[1] for x in g], vlm.encode(its)))
        except Exception as e:
            err.append(e)
        finally:
            q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    n, nxt, t0 = 0, every, time.time()
    while True:
        got = q.get()
        if got is None:
            break
        its, metas, enc = got
        if mode == "score":
            P, cov = vlm.score(enc)
            for m, p, c in zip(metas, P, cov):
                on_result(m, (p, c))
        else:
            for m, r in zip(metas, vlm.attribute(enc, its, [m["target"] for m in metas])):
                on_result(m, r)
        n += len(its)
        if n >= nxt:
            el = time.time() - t0
            log(f"{label}: {n}/{len(specs)}  {n / el:.1f}/s  ETA {(len(specs) - n) / max(n / el, 1e-9) / 60:.0f} min")
            nxt += every
    th.join()
    if err:
        raise err[0]
    return n


# ============================================================================ experiment
class Study:
    def __init__(self, vlm, df, out_root, budget=65536, cap=64, attr_budget=12288, attr_cap=8, donor_seed=13):
        self.vlm, self.df, self.out = vlm, df.set_index("global_id", drop=False), os.path.join(out_root, vlm.name)
        self.budget, self.cap, self.attr_budget, self.attr_cap = budget, cap, attr_budget, attr_cap
        self.donor = make_donors(df, donor_seed)
        self._img = lru_cache(maxsize=2048)(lambda g: vlm.prepare_image(Image.open(self.df.at[g, "image_path"])))
        self._info = {}
        os.makedirs(self.out, exist_ok=True)

    def info(self, g):
        if g not in self._info:
            self._info[g] = self.vlm.text_info(self.df.at[g, "text"])
        return self._info[g]

    def _infos(self, ids):
        t = time.time()
        for g in ids:
            self.info(g)
        if time.time() - t > 5:
            log(f"tokenised {len(ids)} articles in {time.time() - t:.0f}s")

    # ---------------------------------------------------------------- RQ1 core conditions
    def build(self, cond, g):
        info = self.info(g)
        meta = {"global_id": g, "condition": cond}
        if cond == "full":
            return {"text": info.text, "image": self._img(g)}, meta
        if cond == "text_only":
            return {"text": info.text, "image": None}, meta
        if cond == "image_only":
            return {"text": "", "image": self._img(g)}, meta
        if cond == "text_gray":
            return {"text": info.text, "image": Image.new("RGB", self._img(g).size, FILL_RGB)}, meta
        if cond == "swap_diff":
            d = self.donor[g]
            meta["donor_id"] = d
            return {"text": info.text, "image": self._img(d)}, meta
        raise ValueError(cond)

    def run_core(self, ids, conditions=CORE_CONDITIONS):
        self._infos(ids)
        for cond in conditions:
            w = PartWriter(os.path.join(self.out, "core", cond), ["global_id"])
            done = {k[0] for k in w.done()}
            todo = [g for g in ids if g not in done]
            if not todo:
                log(f"[{self.vlm.name}] {cond}: done already")
                continue
            has_t, has_i = cond != "image_only", cond != "text_only"
            specs = [(g, self.vlm.est_tokens(self.info(g).ntok if has_t else 0, has_i),
                      (lambda g=g, cond=cond: self.build(cond, g))) for g in todo]

            def on(m, r, w=w):
                p, c = r
                w.add({**m, "pred": int(p.argmax()), "coverage": float(c), **{f"p{k}": float(p[k]) for k in range(10)}})

            run_specs(self.vlm, specs, on, self.budget, self.cap, label=f"[{self.vlm.name}] {cond}")
            w.flush()
            log(f"[{self.vlm.name}] {cond}: finished {len(todo)}")

    def core(self, cond):
        df = read_parts(os.path.join(self.out, "core", cond))
        return None if df is None else df.drop_duplicates("global_id", keep="last").set_index("global_id")

    # ---------------------------------------------------------------- RQ2 attribution
    def attr_path(self, ctx):
        return os.path.join(self.out, "attribution", f"{ctx}.pkl")

    def attribution(self, ctx):
        p = self.attr_path(ctx)
        return pickle.load(open(p, "rb")) if os.path.exists(p) else {}

    def run_attribution(self, ids, contexts=("full", "text_only", "image_only")):
        self._infos(ids)
        for ctx in contexts:
            base = self.core(ctx)
            assert base is not None, f"run_core({ctx}) first"
            A = self.attribution(ctx)
            todo = [g for g in ids if g not in A and g in base.index]
            if not todo:
                continue
            p = self.attr_path(ctx)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            has_t, has_i = ctx != "image_only", ctx != "text_only"

            def build(g, ctx=ctx):
                it, _ = self.build(ctx, g)
                it["info"] = self.info(g) if ctx != "image_only" else None
                return it, {"global_id": g, "target": int(base.at[g, "pred"])}

            specs = [(g, self.vlm.est_tokens(self.info(g).ntok if has_t else 0, has_i), (lambda g=g: build(g)))
                     for g in todo]
            cnt = {"n": 0}

            def on(m, r, A=A):
                A[m["global_id"]] = r
                cnt["n"] += 1
                if cnt["n"] % 500 == 0:
                    pickle.dump(A, open(p + ".tmp", "wb"))
                    os.replace(p + ".tmp", p)

            run_specs(self.vlm, specs, on, self.attr_budget, self.attr_cap, mode="attr", every=500,
                      label=f"[{self.vlm.name}] attribution {ctx}")
            pickle.dump(A, open(p + ".tmp", "wb"))
            os.replace(p + ".tmp", p)

    # ---------------------------------------------------------------- RQ2 deletion check
    def run_deletion(self, ids, levels=(0.1, 0.2, 0.4), draws=2, seed=7,
                     plan=(("full", "text"), ("full", "image"), ("text_only", "text"), ("image_only", "image"))):
        """Delete the top-k% most relevant words / image cells vs the same number chosen at random."""
        self._infos(ids)
        w = PartWriter(os.path.join(self.out, "deletion"), ["global_id", "context", "modality", "level", "kind", "draw"])
        done = w.done()
        for ctx in sorted({c for c, _ in plan}):
            A = self.attribution(ctx)
            specs = []
            for g in ids:
                a = A.get(g)
                if a is None:
                    continue
                for c2, mod in plan:
                    if c2 != ctx:
                        continue
                    if mod == "text":
                        if not a.get("text_found") or len(a["word_rel"]) < 5:
                            continue
                        score_vec = a["word_rel"]
                    else:
                        if "img_map" not in a:
                            continue
                        score_vec = a["img_map"].ravel()
                    n = len(score_vec)
                    rng = np.random.default_rng(int(hashlib.sha1(f"{g}|{ctx}|{mod}|{seed}".encode()).hexdigest()[:12], 16))
                    order = np.argsort(-score_vec, kind="stable")
                    for lv in levels:
                        k = max(1, int(round(lv * n)))
                        sets = [("evidence", 0, order[:k])] + [("random", d, rng.choice(n, k, replace=False))
                                                               for d in range(draws)]
                        for kind, d, sel in sets:
                            key = tuple(map(str, (g, ctx, mod, lv, kind, d)))
                            if key in done:
                                continue
                            meta = dict(global_id=g, context=ctx, modality=mod, level=lv, kind=kind, draw=d, k=k, n=n,
                                        target=a["target"])
                            ntok = self.info(g).ntok if ctx != "image_only" else 0
                            specs.append((key, self.vlm.est_tokens(ntok, ctx != "text_only"),
                                          (lambda g=g, ctx=ctx, mod=mod, sel=sel, meta=meta: self._deleted(g, ctx, mod, sel, meta))))
            log(f"[{self.vlm.name}] deletion {ctx}: {len(specs)} inputs to score")
            if not specs:
                continue

            def on(m, r):
                p, c = r
                w.add({**m, "pred": int(p.argmax()), "p_target": float(p[m["target"]]), "coverage": float(c)})

            run_specs(self.vlm, specs, on, self.budget, self.cap, label=f"[{self.vlm.name}] deletion {ctx}")
            w.flush()

    def _deleted(self, g, ctx, mod, sel, meta):
        it, _ = self.build(ctx, g)
        if mod == "text":
            info = self.info(g)
            drop = set(int(j) for j in sel)
            parts, last = [], 0
            for j, (a, b) in enumerate(info.spans):
                if j in drop:
                    parts.append(info.text[last:a])
                    last = b
            parts.append(info.text[last:])
            it["text"] = "".join(parts)
        else:
            img = it["image"]
            gh, gw = self.vlm.grid(img)
            cells = np.zeros(gh * gw, bool)
            cells[np.asarray(sel)] = True
            W, H = img.size
            ys = (np.arange(H) * gh // H).clip(0, gh - 1)
            xs = (np.arange(W) * gw // W).clip(0, gw - 1)
            pm = cells.reshape(gh, gw)[ys[:, None], xs[None, :]]
            a = np.asarray(img).copy()
            a[pm] = FILL_RGB
            it["image"] = Image.fromarray(a)
        return it, meta

    def deletion(self):
        df = read_parts(os.path.join(self.out, "deletion"))
        return None if df is None else df.drop_duplicates(
            ["global_id", "context", "modality", "level", "kind", "draw"], keep="last")

    # ---------------------------------------------------------------- pre-flight
    def check(self, ids, n=30):
        ids = list(ids)[:n]
        self._infos(ids)
        rep = {}
        for cond in ("full", "text_only", "image_only"):
            items = [self.build(cond, g)[0] for g in ids]
            t = time.time()
            P, C = [], []
            for s in range(0, len(items), 8):
                p, c = self.vlm.score(self.vlm.encode(items[s:s + 8]))
                P.append(p)
                C.append(c)
            P, C = np.concatenate(P), np.concatenate(C)
            y = self.df.loc[ids, "label"].values
            rep[cond] = dict(acc=float((P.argmax(1) == y).mean()), coverage=float(C.mean()),
                             sec_per_item=(time.time() - t) / len(items))
            log(f"{cond:>10}: acc={rep[cond]['acc']:.2f}  coverage={C.mean():.3f}  {rep[cond]['sec_per_item']:.3f}s/item (batch 8)")
        its = [self.build("full", g)[0] for g in ids[:6]]
        gens = self.vlm.generate(its)
        P, _ = self.vlm.score(self.vlm.encode(its))
        agree = np.mean([g.strip().lower().startswith(CATS[int(p.argmax())].lower()[:4]) for g, p in zip(gens, P)])
        log("greedy outputs:", gens, "| agree with scored argmax:", agree)
        it = dict(its[0], info=self.info(ids[0]))
        t = time.time()
        a = self.vlm.attribute(self.vlm.encode([it]), [it], [int(P[0].argmax())])[0]
        log(f"attribution ok: img grid {a['img_map'].shape}, text found={a['text_found']}, "
            f"image share={a['R_img'] / (a['R_img'] + a['R_text'] + 1e-12):.2f}, {time.time() - t:.2f}s")
        if rep["full"]["coverage"] < 0.5:
            log("WARNING: low label coverage — inspect greedy outputs")
        rep["greedy_agree"] = float(agree)
        json.dump(rep, open(os.path.join(self.out, "check.json"), "w"), indent=2)
        return rep


# ============================================================================ statistics
def _w(n, B, seed, chunk=500):
    rng = np.random.default_rng(seed)
    k = 0
    while k < B:
        b = min(chunk, B - k)
        yield rng.multinomial(n, np.full(n, 1 / n), size=b).astype(np.float64)
        k += b


def boot_mean(x, B=10000, seed=0):
    x = np.asarray(x, float)
    return np.concatenate([w @ x / len(x) for w in _w(len(x), B, seed)])


def f1_from_cm(cm):
    tp = np.diag(cm)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def macro_f1(y, p):
    return float(f1_from_cm(np.bincount(np.asarray(y) * 10 + np.asarray(p), minlength=100).reshape(10, 10)).mean())


def boot_f1(y, preds, B=10000, seed=0):
    codes = [np.asarray(y) * 10 + np.asarray(p) for p in preds]
    out = [[] for _ in preds]
    for W in _w(len(y), B, seed):
        for w in W:
            for k, c in enumerate(codes):
                out[k].append(f1_from_cm(np.bincount(c, weights=w, minlength=100).reshape(10, 10)).mean())
    return [np.array(o) for o in out]


def ci(d, lvl=0.95):
    return float(np.quantile(d, (1 - lvl) / 2)), float(np.quantile(d, 1 - (1 - lvl) / 2))


def pval(d):
    d = np.asarray(d)
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def jsd(P, Q, eps=1e-12):
    P, Q = np.clip(P, eps, 1), np.clip(Q, eps, 1)
    M = (P + Q) / 2
    return 0.5 * (P * np.log(P / M)).sum(1) + 0.5 * (Q * np.log(Q / M)).sum(1)


def fmt(v, lo, hi, s=100, nd=1):
    return f"{v * s:.{nd}f} [{lo * s:.{nd}f}, {hi * s:.{nd}f}]"


# ============================================================================ analysis
class Analysis:
    """Builds every table/figure for one or more models from the stored results."""

    def __init__(self, studies, out_dir, B=10000, seed=2027):
        self.S = {s.vlm.name if hasattr(s, "vlm") else s.name: s for s in studies}
        self.out, self.B, self.seed = out_dir, B, seed
        os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
        self.tables = {}

    # helpers
    @staticmethod
    def P(df, ids):
        return df.loc[ids, [f"p{k}" for k in range(10)]].values.astype(float)

    def _core(self, s):
        return {c: d for c in CORE_CONDITIONS if (d := s.core(c)) is not None}

    def save(self, name, t, cols=None, caption=""):
        t.to_csv(os.path.join(self.out, "tables", f"{name}.csv"), index=False)
        d = t[cols] if cols else t
        with open(os.path.join(self.out, "tables", f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write(f"% {caption}\n" + d.to_latex(index=False, escape=True, na_rep="--",
                                                   float_format=lambda v: f"{v:.3f}"))
        self.tables[name] = t
        return t

    # ---------------------------------------------------------------- T1
    def t1_performance(self):
        rows = []
        for m, s in self.S.items():
            C = self._core(s)
            if "full" not in C:
                continue
            for c, d in C.items():
                ids = d.index.intersection(C["full"].index)
                y = s.df.loc[ids, "label"].values
                pc, pf = d.loc[ids, "pred"].values, C["full"].loc[ids, "pred"].values
                bc, bf = boot_f1(y, [pc, pf], self.B, self.seed)
                dd = bc - bf
                rows.append(dict(model=m, condition=c, n=len(ids), macro_f1=macro_f1(y, pc), f1_lo=ci(bc)[0],
                                 f1_hi=ci(bc)[1], acc=float((pc == y).mean()), d_vs_full=macro_f1(y, pc) - macro_f1(y, pf),
                                 d_lo=ci(dd)[0], d_hi=ci(dd)[1], p=pval(dd) if c != "full" else np.nan,
                                 coverage=float(d.loc[ids, "coverage"].mean())))
        t = pd.DataFrame(rows)
        for m in t.model.unique():
            k = (t.model == m) & t.p.notna()
            t.loc[k, "p_holm"] = holm(t.loc[k, "p"].values)
        t["Macro-F1 [95% CI]"] = [fmt(r.macro_f1, r.f1_lo, r.f1_hi) for r in t.itertuples()]
        t["ΔF1 vs full [95% CI]"] = [fmt(r.d_vs_full, r.d_lo, r.d_hi) if r.condition != "full" else "--"
                                     for r in t.itertuples()]
        return self.save("T1_performance", t, ["model", "condition", "n", "Macro-F1 [95% CI]", "ΔF1 vs full [95% CI]",
                                               "p_holm"], "RQ1: Macro-F1 per input condition")

    # ---------------------------------------------------------------- T2
    def t2_change(self):
        rows = []
        for m, s in self.S.items():
            C = self._core(s)
            if "full" not in C:
                continue
            for c, d in C.items():
                if c == "full":
                    continue
                ids = d.index.intersection(C["full"].index)
                Pf, Pc = self.P(C["full"], ids), self.P(d, ids)
                yh = Pf.argmax(1)
                flip = (Pc.argmax(1) != yh).astype(float)
                dp = Pf[np.arange(len(ids)), yh] - Pc[np.arange(len(ids)), yh]
                js = jsd(Pf, Pc)
                row = dict(model=m, condition=c, n=len(ids))
                for k, v in (("flip", flip), ("dp", dp), ("js", js)):
                    b = boot_mean(v, self.B, self.seed)
                    row[k], (row[f"{k}_lo"], row[f"{k}_hi"]) = float(v.mean()), ci(b)
                if c == "swap_diff":
                    dc = s.df.loc[d.loc[ids, "donor_id"].values, "label"].values
                    cap = (d.loc[ids, "pred"].values == dc).astype(float)
                    row["capture"], (row["cap_lo"], row["cap_hi"]) = float(cap.mean()), ci(boot_mean(cap, self.B, self.seed))
                rows.append(row)
        t = pd.DataFrame(rows)
        t["label flip % [CI]"] = [fmt(r.flip, r.flip_lo, r.flip_hi) for r in t.itertuples()]
        t["Δp(ŷ) ×100 [CI]"] = [fmt(r.dp, r.dp_lo, r.dp_hi) for r in t.itertuples()]
        t["JS divergence"] = [f"{r.js:.3f}" for r in t.itertuples()]
        if "capture" in t:
            t["swap capture % [CI]"] = [fmt(r.capture, r.cap_lo, r.cap_hi) if r.condition == "swap_diff" else "--"
                                        for r in t.itertuples()]
        cols = ["model", "condition", "n", "label flip % [CI]", "Δp(ŷ) ×100 [CI]", "JS divergence"] + \
               (["swap capture % [CI]"] if "capture" in t else [])
        return self.save("T2_change_vs_full", t, cols, "RQ1: how much the prediction changes w.r.t. text+image")

    # ---------------------------------------------------------------- T3
    def t3_complementarity(self):
        rows = []
        for m, s in self.S.items():
            C = self._core(s)
            if not all(k in C for k in ("full", "text_only", "image_only")):
                continue
            ids = C["full"].index.intersection(C["text_only"].index).intersection(C["image_only"].index)
            y = s.df.loc[ids, "label"].values
            f, t_, i_ = [(C[k].loc[ids, "pred"].values == y) for k in ("full", "text_only", "image_only")]
            ind = {
                "acc_text_only": t_, "acc_image_only": i_, "acc_full": f, "oracle_text_or_image": t_ | i_,
                "gain_vs_text (full✓ text✗)": f & ~t_, "loss_vs_text (full✗ text✓)": ~f & t_,
                "net_vs_text": f.astype(float) - t_.astype(float),
                "gain_vs_image (full✓ image✗)": f & ~i_, "loss_vs_image (full✗ image✓)": ~f & i_,
            }
            for k, v in ind.items():
                v = v.astype(float)
                lo, hi = ci(boot_mean(v, self.B, self.seed))
                rows.append(dict(model=m, metric=k, n=len(ids), value=float(v.mean()), lo=lo, hi=hi,
                                 **{"% [CI]": fmt(v.mean(), lo, hi)}))
            for tt in (True, False):
                for ii in (True, False):
                    cell = (t_ == tt) & (i_ == ii)
                    rows.append(dict(model=m, metric=f"cell text{'✓' if tt else '✗'} image{'✓' if ii else '✗'}",
                                     n=int(cell.sum()), value=float(cell.mean()), lo=np.nan, hi=np.nan,
                                     **{"% [CI]": f"{100 * cell.mean():.1f}% of articles; full correct in "
                                                  f"{100 * f[cell].mean() if cell.any() else float('nan'):.1f}%"}))
        t = pd.DataFrame(rows)
        return self.save("T3_complementarity", t, ["model", "metric", "n", "% [CI]"],
                         "RQ3: complementarity of text and image")

    # ---------------------------------------------------------------- T4
    def t4_by_text_confidence(self, nbins=5):
        rows = []
        for m, s in self.S.items():
            C = self._core(s)
            if not all(k in C for k in ("full", "text_only", "image_only")):
                continue
            ids = C["full"].index.intersection(C["text_only"].index).intersection(C["image_only"].index)
            y = s.df.loc[ids, "label"].values
            Pt = self.P(C["text_only"], ids)
            conf = Pt.max(1)
            q = pd.qcut(conf, nbins, labels=False, duplicates="drop")
            f, t_, i_ = [(C[k].loc[ids, "pred"].values == y) for k in ("full", "text_only", "image_only")]
            flip = C["full"].loc[ids, "pred"].values != C["text_only"].loc[ids, "pred"].values
            cap = None
            if "swap_diff" in C:
                sw = C["swap_diff"].reindex(ids)
                ok = sw.pred.notna().values
                cap = np.full(len(ids), np.nan)
                cap[ok] = (sw.pred.values[ok] == s.df.loc[sw.donor_id.values[ok], "label"].values).astype(float)
            for b in sorted(np.unique(q)):
                k = q == b
                gain = (f & ~t_)[k].astype(float)
                loss = (~f & t_)[k].astype(float)
                row = dict(model=m, bin=int(b) + 1, n=int(k.sum()), text_conf_range=f"{conf[k].min():.2f}–{conf[k].max():.2f}",
                           acc_text=float(t_[k].mean()), acc_image=float(i_[k].mean()), acc_full=float(f[k].mean()),
                           gain=float(gain.mean()), loss=float(loss.mean()), flip_full_vs_text=float(flip[k].mean()))
                row["net_lo"], row["net_hi"] = ci(boot_mean(gain - loss, self.B, self.seed))
                if cap is not None:
                    row["swap_capture"] = float(np.nanmean(cap[k]))
                rows.append(row)
        t = pd.DataFrame(rows)
        return self.save("T4_by_text_confidence", t, None, "RQ3: when does the image help? (bins of text-only confidence)")

    # ---------------------------------------------------------------- T5
    def t5_by_category(self):
        rows = []
        for m, s in self.S.items():
            C = self._core(s)
            if not all(k in C for k in ("full", "text_only", "image_only")):
                continue
            A = s.attribution("full")
            ids_all = C["full"].index.intersection(C["text_only"].index).intersection(C["image_only"].index)
            for cat in CATS:
                ids = [g for g in ids_all if s.df.at[g, "category"] == cat]
                if not ids:
                    continue
                y = s.df.loc[ids, "label"].values
                f, t_, i_ = [(C[k].loc[ids, "pred"].values == y) for k in ("full", "text_only", "image_only")]
                row = dict(model=m, category=cat, n=len(ids), acc_text=t_.mean(), acc_image=i_.mean(), acc_full=f.mean(),
                           gain=(f & ~t_).mean(), loss=(~f & t_).mean())
                if "swap_diff" in C:
                    sw = C["swap_diff"].reindex(ids).dropna(subset=["pred"])
                    row["swap_capture"] = float((sw.pred.values == s.df.loc[sw.donor_id.values, "label"].values).mean())
                sh = [A[g]["R_img"] / (A[g]["R_img"] + A[g]["R_text"]) for g in ids if g in A and A[g]["R_text"] > 0]
                row["image_share"] = float(np.mean(sh)) if sh else np.nan
                rows.append(row)
        t = pd.DataFrame(rows)
        return self.save("T5_by_category", t, None, "RQ1-3 per topic")

    # ---------------------------------------------------------------- T6 where it looks
    def t6_where_it_looks(self):
        rows = []
        for m, s in self.S.items():
            C = self._core(s)
            A = s.attribution("full")
            if not A or "text_only" not in C:
                continue
            ids = [g for g in A if A[g]["R_text"] > 0 and g in C["text_only"].index]
            sh = np.array([A[g]["R_img"] / (A[g]["R_img"] + A[g]["R_text"]) for g in ids])
            ptok_i = np.array([A[g]["R_img"] / max(A[g]["n_img_tok"], 1) for g in ids])
            ptok_t = np.array([A[g]["R_text"] / max(A[g]["n_text_tok"], 1) for g in ids])
            prm = np.array([A[g]["R_prompt"] / (A[g]["R_img"] + A[g]["R_text"] + A[g]["R_prompt"]) for g in ids])
            y = s.df.loc[ids, "label"].values
            corr = C["full"].loc[ids, "pred"].values == y
            Pf, Pt = self.P(C["full"], ids), self.P(C["text_only"], ids)
            yh = Pf.argmax(1)
            dp_img = Pf[np.arange(len(ids)), yh] - Pt[np.arange(len(ids)), yh]
            rho = pd.Series(sh).corr(pd.Series(dp_img), method="spearman")
            rng = np.random.default_rng(self.seed)
            rb = [pd.Series(sh[ix]).corr(pd.Series(dp_img[ix]), method="spearman")
                  for ix in (rng.integers(0, len(ids), len(ids)) for _ in range(min(self.B, 2000)))]
            rows += [
                dict(model=m, metric="image share of relevance (mean)", n=len(ids), value=sh.mean(),
                     ci=fmt(sh.mean(), *ci(boot_mean(sh, self.B, self.seed)))),
                dict(model=m, metric="image share (median)", n=len(ids), value=float(np.median(sh)), ci=""),
                dict(model=m, metric="image share | full correct", n=int(corr.sum()), value=sh[corr].mean(), ci=""),
                dict(model=m, metric="image share | full wrong", n=int((~corr).sum()),
                     value=sh[~corr].mean() if (~corr).any() else np.nan, ci=""),
                dict(model=m, metric="relevance per image token / per text token", n=len(ids),
                     value=float(np.median(ptok_i / np.maximum(ptok_t, 1e-12))), ci="median ratio"),
                dict(model=m, metric="prompt/instruction share of all relevance", n=len(ids), value=prm.mean(), ci=""),
                dict(model=m, metric="Spearman(image share, Δp when image removed)", n=len(ids), value=float(rho),
                     ci=f"[{np.nanquantile(rb, .025):.2f}, {np.nanquantile(rb, .975):.2f}]"),
            ]
        t = pd.DataFrame(rows)
        return self.save("T6_where_it_looks", t, None, "RQ2: text vs image share of gradient×input relevance")

    # ---------------------------------------------------------------- T7 deletion
    def t7_deletion(self):
        rows = []
        for m, s in self.S.items():
            D = s.deletion()
            if D is None:
                continue
            for (ctx, mod, lv), g in D.groupby(["context", "modality", "level"]):
                base = s.core(ctx)
                ev = g[g.kind == "evidence"].set_index("global_id")
                rdf = g[g.kind == "random"].copy()
                rdf["flip"] = (rdf.pred != rdf.target).astype(float)
                rd = rdf.groupby("global_id")[["p_target", "flip"]].mean()
                ids = ev.index.intersection(rd.index)
                p0 = self.P(base, ids)[np.arange(len(ids)), ev.loc[ids, "target"].values]
                d_ev = p0 - ev.loc[ids, "p_target"].values
                d_rd = p0 - rd.loc[ids, "p_target"].values
                eff = d_ev - d_rd
                f_ev = (ev.loc[ids, "pred"].values != ev.loc[ids, "target"].values).astype(float)
                f_rd = rd.loc[ids, "flip"].values
                b = boot_mean(eff, self.B, self.seed)
                fb = boot_mean(f_ev - f_rd, self.B, self.seed)
                rows.append(dict(model=m, context=ctx, deleted=mod, level=lv, n=len(ids), dp_evidence=d_ev.mean(),
                                 dp_random=d_rd.mean(), effect=eff.mean(), lo=ci(b)[0], hi=ci(b)[1], p=pval(b),
                                 flip_evidence=f_ev.mean(), flip_random=f_rd.mean(), flip_lo=ci(fb)[0], flip_hi=ci(fb)[1],
                                 win_rate=float((eff > 0).mean())))
        t = pd.DataFrame(rows)
        if len(t):
            t["p_holm"] = holm(t.p.values)
            t["Δp evidence − random ×100 [CI]"] = [fmt(r.effect, r.lo, r.hi) for r in t.itertuples()]
            t["flip evid. / rand. %"] = [f"{100 * r.flip_evidence:.1f} / {100 * r.flip_random:.1f}" for r in t.itertuples()]
        return self.save("T7_deletion_check", t, ["model", "context", "deleted", "level", "n",
                                                  "Δp evidence − random ×100 [CI]", "flip evid. / rand. %", "p_holm"]
                         if len(t) else None, "RQ2: does deleting what the model 'looks at' matter more than random?")

    # ---------------------------------------------------------------- T8 top words
    def t8_top_words(self, k=12, min_df=5):
        """Words with the largest relevance that is SPECIFIC to a topic: mean normalised relevance in the
        topic minus the mean in the other topics (removes generic words such as 'của', 'tin', 'hôm nay')."""
        rows = []
        for m, s in self.S.items():
            A = s.attribution("full") or s.attribution("text_only")
            if not A:
                continue
            per = {}
            for g, a in A.items():
                tw = a.get("top_words")
                if not tw:
                    continue
                tot = sum(r for _, r in tw) or 1.0
                per[g] = {w: r / tot for w, r in tw if len(w) >= 2}
            cat_of = {g: s.df.at[g, "category"] for g in per}
            for cat in CATS:
                inn = [g for g in per if cat_of[g] == cat]
                out = [g for g in per if cat_of[g] != cat]
                if not inn:
                    continue
                mi, mo, dfc = {}, {}, {}
                for g in inn:
                    for w, r in per[g].items():
                        mi[w] = mi.get(w, 0) + r / len(inn)
                        dfc[w] = dfc.get(w, 0) + 1
                for g in out:
                    for w, r in per[g].items():
                        if w in mi:
                            mo[w] = mo.get(w, 0) + r / max(len(out), 1)
                thr = min(min_df, max(1, len(inn) // 10))
                score = {w: mi[w] - mo.get(w, 0) for w in mi if dfc[w] >= thr}
                top = sorted(score, key=lambda w: -score[w])[:k]
                rows.append(dict(model=m, category=cat, n=len(inn), top_words=", ".join(top)))
        t = pd.DataFrame(rows)
        return self.save("T8_top_words", t, None, "RQ2: topic-specific words with the largest relevance (full input)")

    # ---------------------------------------------------------------- figures
    def _plt(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
        return plt

    def _savefig(self, fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(self.out, "figures", f"{name}.{ext}"), dpi=200, bbox_inches="tight")
        return fig

    def f1_conditions(self):
        t = self.tables.get("T1_performance")
        if t is None:
            return
        plt = self._plt()
        ms = list(t.model.unique())
        conds = [c for c in CORE_CONDITIONS if c in set(t.condition)]
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        wdt = 0.8 / len(ms)
        for j, m in enumerate(ms):
            tt = t[t.model == m].set_index("condition").reindex(conds)
            x = np.arange(len(conds)) + (j - (len(ms) - 1) / 2) * wdt
            ax.bar(x, tt.macro_f1 * 100, wdt, yerr=[(tt.macro_f1 - tt.f1_lo) * 100, (tt.f1_hi - tt.macro_f1) * 100],
                   capsize=2, label=m, color=["#1f4e79", "#d98c3a", "#6aa2d8"][j % 3])
        ax.set_xticks(np.arange(len(conds)), conds)
        ax.set_ylabel("Macro-F1 (%)")
        ax.legend(frameon=False)
        return self._savefig(fig, "F1_conditions")

    def f2_confidence(self):
        t = self.tables.get("T4_by_text_confidence")
        if t is None or not len(t):
            return
        plt = self._plt()
        ms = list(t.model.unique())
        fig, axes = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 2.9), squeeze=False)
        for ax, m in zip(axes[0], ms):
            tt = t[t.model == m]
            x = tt.bin.values
            ax.bar(x - 0.2, tt.gain * 100, 0.4, color="#2e7d4f", label="image fixes (full✓ text✗)")
            ax.bar(x + 0.2, -tt.loss * 100, 0.4, color="#b5561c", label="image breaks (full✗ text✓)")
            if "swap_capture" in tt:
                ax.plot(x, tt.swap_capture * 100, "k.--", label="swap capture")
            ax.axhline(0, color="#888", lw=.6)
            ax.set_xticks(x, [f"Q{b}" for b in x])
            ax.set_xlabel("text-only confidence quintile (low → high)")
            ax.set_ylabel("% of articles")
            ax.set_title(m)
        axes[0][0].legend(frameon=False, fontsize=7)
        return self._savefig(fig, "F2_when_image_helps")

    def f3_deletion(self):
        t = self.tables.get("T7_deletion_check")
        if t is None or not len(t):
            return
        plt = self._plt()
        combos = [("full", "text"), ("full", "image"), ("text_only", "text"), ("image_only", "image")]
        ms = list(t.model.unique())
        fig, axes = plt.subplots(len(ms), 4, figsize=(11, 2.5 * len(ms)), squeeze=False)
        for r, m in enumerate(ms):
            for c, (ctx, mod) in enumerate(combos):
                ax = axes[r][c]
                tt = t[(t.model == m) & (t.context == ctx) & (t.deleted == mod)].sort_values("level")
                if not len(tt):
                    ax.axis("off")
                    continue
                ax.plot(tt.level * 100, tt.dp_evidence * 100, "o-", color="#1f4e79", label="most relevant removed")
                ax.plot(tt.level * 100, tt.dp_random * 100, "s--", color="#999", label="random, same amount")
                ax.set_title(f"{m}: delete {mod} | input={ctx}", fontsize=8)
                ax.set_xlabel(f"% of {'words' if mod == 'text' else 'image cells'} removed")
                ax.set_ylabel("drop in p(ŷ) ×100")
        axes[0][0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        return self._savefig(fig, "F3_deletion_check")

    def f4_share(self):
        plt = self._plt()
        data = []
        for m, s in self.S.items():
            A = s.attribution("full")
            for g, a in A.items():
                if a["R_text"] > 0:
                    data.append((m, s.df.at[g, "category"], a["R_img"] / (a["R_img"] + a["R_text"])))
        if not data:
            return
        d = pd.DataFrame(data, columns=["model", "category", "share"])
        ms = list(d.model.unique())
        fig, axes = plt.subplots(1, len(ms), figsize=(5.5 * len(ms), 2.9), squeeze=False)
        for ax, m in zip(axes[0], ms):
            dd = d[d.model == m]
            ax.boxplot([dd[dd.category == c].share.values * 100 for c in CATS], showfliers=False)
            ax.set_xticks(range(1, 11), CATS, rotation=45, ha="right")
            ax.set_ylabel("image share of relevance (%)")
            ax.set_title(m)
        return self._savefig(fig, "F4_image_share_by_topic")

    def f5_examples(self, model=None, n=6):
        """Image heatmap + top words for illustrative test articles (fusion fixed / broke / high image share)."""
        plt = self._plt()
        m = model or list(self.S)[0]
        s = self.S[m]
        C = self._core(s)
        A = s.attribution("full")
        if not A or "text_only" not in C:
            return
        ids = [g for g in A if g in C["text_only"].index and "img_map" in A[g] and A[g].get("top_words")]
        y = s.df.loc[ids, "label"].values
        pf, pt = C["full"].loc[ids, "pred"].values, C["text_only"].loc[ids, "pred"].values
        sh = np.array([A[g]["R_img"] / (A[g]["R_img"] + A[g]["R_text"]) for g in ids])
        groups = [("image fixed text", np.where((pf == y) & (pt != y))[0]),
                  ("image broke text", np.where((pf != y) & (pt == y))[0]),
                  ("highest image share", np.argsort(-sh))]
        pick = []
        for name, idx in groups:
            for j in list(idx)[: n // 3]:
                pick.append((name, ids[j]))
        if not pick:
            return
        fig, axes = plt.subplots(2, len(pick), figsize=(2.6 * len(pick), 5.2), squeeze=False,
                                 gridspec_kw={"height_ratios": [1.3, 1]})
        for c, (name, g) in enumerate(pick):
            a = A[g]
            img = np.asarray(s._img(g)).astype(float) / 255
            hm = a["img_map"] / (a["img_map"].max() + 1e-12)
            H, W = img.shape[:2]
            up = np.kron(hm, np.ones((math.ceil(H / hm.shape[0]), math.ceil(W / hm.shape[1]))))[:H, :W]
            ax = axes[0][c]
            ax.imshow(img)
            ax.imshow(up, cmap="inferno", alpha=0.5)
            ax.axis("off")
            ax.set_title(f"{name}\n{g}\ntrue={s.df.at[g, 'category']} | text→{CATS[pt[ids.index(g)]]} | "
                         f"full→{CATS[pf[ids.index(g)]]}\nimage share {sh[ids.index(g)]:.0%}", fontsize=6.5)
            ax = axes[1][c]
            tw = a["top_words"][:8][::-1]
            ax.barh(range(len(tw)), [r for _, r in tw], color="#1f4e79")
            ax.set_yticks(range(len(tw)), [w for w, _ in tw], fontsize=7)
            ax.set_xticks([])
        fig.tight_layout()
        return self._savefig(fig, f"F5_examples_{m}")

    def run_all(self):
        steps = [self.t1_performance, self.t2_change, self.t3_complementarity, self.t4_by_text_confidence,
                 self.t5_by_category, self.t6_where_it_looks, self.t7_deletion, self.t8_top_words,
                 self.f1_conditions, self.f2_confidence, self.f3_deletion, self.f4_share, self.f5_examples]
        for f in steps:
            t = time.time()
            f()
            log(f"{f.__name__} done ({time.time() - t:.0f}s)")
        return self.tables

# ============================================================================
# OOM-Safe runner wrappers
# ============================================================================

def run_specs_oom_safe(vlm, specs, on_result, budget, cap, mode="score", prefetch=3, every=2000, label=""):
    """An OOM-resilient version of run_specs that catches CUDA OOM, splits batches dynamically, and continues."""
    import gc
    import torch

    batches = make_batches(specs, budget, cap)
    # We process batches sequentially in the main thread here to easily catch and recover from OOM
    n, nxt, t0 = 0, every, time.time()

    # Helper to process a group of items with dynamic batch splitting on OOM
    def process_subbatch(sub_its, sub_metas):
        import torch
        try:
            enc = vlm.encode(sub_its)
            if mode == "score":
                P, cov = vlm.score(enc)
                for m, p, c in zip(sub_metas, P, cov):
                    on_result(m, (p, c))
            else:
                res = vlm.attribute(enc, sub_its, [m["target"] for m in sub_metas])
                for m, r in zip(sub_metas, res):
                    on_result(m, r)
            return True
        except torch.cuda.OutOfMemoryError as oom:
            log(f"[OOM Warning] CUDA Out of Memory detected during {label}. Attempting recovery...")
            torch.cuda.empty_cache()
            gc.collect()

            if len(sub_its) <= 1:
                log(f"[OOM Error] Single item batch failed for {sub_metas[0].get('global_id')}. Skipping this item.")
                return False

            # Split batch into two halves and retry
            mid = len(sub_its) // 2
            log(f"[OOM Recovery] Splitting batch of size {len(sub_its)} into halves: {mid} and {len(sub_its) - mid}")
            ok1 = process_subbatch(sub_its[:mid], sub_metas[:mid])
            ok2 = process_subbatch(sub_its[mid:], sub_metas[mid:])
            return ok1 or ok2
        except Exception as e:
            log(f"[Unexpected Error] {str(e)}. Skipping batch.")
            return False

    for b in batches:
        groups = {}
        for _, _, build in b:
            try:
                it, meta = build()
                if it is not None:
                    groups.setdefault(it["image"] is None, []).append((it, meta))
            except Exception as build_err:
                log(f"[Build Error] Failed to prepare input: {build_err}")
                continue

        for is_text_only, group_items in groups.items():
            its = [x[0] for x in group_items]
            metas = [x[1] for x in group_items]
            process_subbatch(its, metas)
            n += len(its)

        if n >= nxt:
            el = time.time() - t0
            log(f"{label} (OOM-Safe): {n}/{len(specs)}  {n / el:.1f}/s  ETA {(len(specs) - n) / max(n / el, 1e-9) / 60:.0f} min")
            nxt += every

    return n

# Override the default run_specs with our OOM-safe runner in vlmstudy namespace
run_specs = run_specs_oom_safe
