"""vlmstudy.figure — "where does the VLM look?" figure: image heatmaps + highlighted words for both models.

The maps are gradient×input relevance (the same quantity validated by the deletion test in the paper),
not raw attention weights. For each article the figure shows
  photo (with OCR text boxes) | Qwen3-VL heatmap | InternVL3.5 heatmap | predictions per input condition
  highlighted article excerpt for each model
"""
from __future__ import annotations

import os
import re

import numpy as np
from PIL import Image, ImageFilter

from . import core as V

NAMES = {"qwen3vl8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B"}
COND_LABEL = [("text_only", "text only"), ("image_only", "image only"), ("full", "text+image")]
GROUP_TITLE = {"image_fixes": "The image fixes a text-only error",
               "image_breaks": "The image overrides a correct text-only decision",
               "text_in_image": "The model reads text printed in the photo",
               "agree": "Text and image agree"}
OK, BAD = "#1b7f4c", "#c2410c"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    return plt


# ----------------------------------------------------------------------------- data access
def _probs(st, cond, g):
    d = st.core(cond)
    if d is None or g not in d.index:
        return None
    return d.loc[g, [f"p{k}" for k in range(10)]].values.astype(float)


def _words_and_rel(st, df, g):
    a = st.attribution("full").get(g)
    if not a or "word_rel" not in a:
        return None, None
    words = df.at[g, "text"].split()
    wr = np.asarray(a["word_rel"], float)
    n = min(len(words), len(wr))
    return words[:n], wr[:n]


def candidates(studies, df, ocr=None, min_conf=0.3):
    """Ranked article ids per group (decided on the first = main model)."""
    from .ocr import text_boxes

    dfi = df.set_index("global_id", drop=False)
    main = studies[0]
    ids = [g for g in main.attribution("full") if all(g in s.attribution("full") for s in studies)]
    out = {k: [] for k in GROUP_TITLE}
    for g in ids:
        y = int(dfi.at[g, "label"])
        pf, pt, pi = (_probs(main, c, g) for c in ("full", "text_only", "image_only"))
        if pf is None or pt is None or pi is None:
            continue
        a = main.attribution("full")[g]
        share = a["R_img"] / (a["R_img"] + a["R_text"] + 1e-12)
        f_ok, t_ok, i_ok = pf.argmax() == y, pt.argmax() == y, pi.argmax() == y
        if f_ok and not t_ok:
            out["image_fixes"].append((pf[y] - pt[y] + share, g))
        if t_ok and not f_ok:
            out["image_breaks"].append((pt[y] - pf[y], g))
        if f_ok and t_ok and i_ok:
            out["agree"].append((share, g))
        if ocr is not None and text_boxes(ocr.get(g), min_conf)[0]:
            pm = _probs(main, "image_only_ocrmask", g)
            if pm is not None and i_ok and pm.argmax() != y:
                out["text_in_image"].append((pi[y] - pm[y], g))
    return {k: [g for _, g in sorted(v, reverse=True)] for k, v in out.items()}


# ----------------------------------------------------------------------------- drawing helpers
def _heat(img, grid, alpha=0.8, dim=0.5, cmap="inferno"):
    from matplotlib import cm

    g = np.asarray(grid, float)
    g = np.clip(g / (np.percentile(g, 99) + 1e-12), 0, 1)
    W, H = img.size
    m = Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    m = m.filter(ImageFilter.GaussianBlur(radius=max(W, H) / max(g.shape) * 0.35))
    m = np.asarray(m, float) / 255.0
    col = getattr(cm, cmap)(m)[..., :3] * 255
    base = np.asarray(img.convert("RGB"), float) * dim
    a = (alpha * m)[..., None]
    return Image.fromarray((base * (1 - a) + col * a).clip(0, 255).astype(np.uint8))


def _window(rel, n):
    if len(rel) <= n:
        return 0, len(rel)
    c = np.convolve(rel, np.ones(n), "valid")
    s = int(c.argmax())
    return s, s + n


def _draw_words(ax, words, scores, fontsize=6.6, line_h=0.155, cmap="Blues", x0=0.0, y0=0.92):
    """Word-wrapped text with a background whose intensity encodes relevance."""
    from matplotlib import cm

    fig = ax.figure
    r = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    x, y = x0, y0
    space = None
    cmap = cm.get_cmap(cmap) if hasattr(cm, "get_cmap") else __import__("matplotlib").colormaps[cmap]
    for w, s in zip(words, scores):
        t = ax.text(x, y, w, fontsize=fontsize, va="top", ha="left", transform=ax.transAxes, zorder=3,
                    bbox=dict(boxstyle="square,pad=0.12", fc=cmap(0.1 + 0.8 * s) if s > 0.15 else "none",
                              ec="none"), color="white" if s > 0.6 else "#111111")
        bb = t.get_window_extent(renderer=r)
        (x1, _), (x2, _) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        if space is None:
            sp = ax.text(0, 0, "a a", fontsize=fontsize, transform=ax.transAxes)
            b1 = sp.get_window_extent(renderer=r)
            sp2 = ax.text(0, 0, "aa", fontsize=fontsize, transform=ax.transAxes)
            b2 = sp2.get_window_extent(renderer=r)
            space = inv.transform((b1.x1 - b2.x1 + b2.x0, 0))[0] - inv.transform((b2.x0, 0))[0]
            sp.remove()
            sp2.remove()
        if x2 > 1.0 and x > x0:
            x, y = x0, y - line_h
            t.set_position((x, y))
            bb = t.get_window_extent(renderer=r)
            x2 = inv.transform((bb.x1, bb.y1))[0]
        x = x2 + space * 1.15
    return y


def _pred_panel(ax, studies, dfi, g):
    ax.axis("off")
    y_true = int(dfi.at[g, "label"])
    ax.text(0.0, 1.0, f"True topic: {V.CATS[y_true]}", fontsize=7, fontweight="bold", va="top", transform=ax.transAxes)
    cols = [0.0, 0.285, 0.655]  # wide enough that "✓ Entertainment" in both model columns does not overlap
    ax.text(cols[1], 0.84, NAMES.get(studies[0].vlm.name, studies[0].vlm.name).split("-")[0], fontsize=6.3,
            va="top", transform=ax.transAxes, color="#444")
    if len(studies) > 1:
        ax.text(cols[2], 0.84, NAMES.get(studies[1].vlm.name, studies[1].vlm.name).split("-")[0], fontsize=6.3,
                va="top", transform=ax.transAxes, color="#444")
    y = 0.68
    for c, lab in COND_LABEL:
        ax.text(cols[0], y, lab, fontsize=6.3, va="top", transform=ax.transAxes)
        for j, st in enumerate(studies[:2]):
            p = _probs(st, c, g)
            if p is None:
                continue
            k = int(p.argmax())
            ok = k == y_true
            ax.text(cols[1 + j], y, f"{'✓' if ok else '✗'} {V.CATS[k]}\n   p={p[k]:.2f}", fontsize=5.5, va="top",
                    transform=ax.transAxes, color=OK if ok else BAD)
        y -= 0.22


# ----------------------------------------------------------------------------- one example
def draw_example(fig, outer, studies, df, g, ocr=None, title=None, n_words=46):
    from matplotlib import gridspec
    from matplotlib.patches import Rectangle

    from .ocr import text_boxes

    dfi = df.set_index("global_id", drop=False)
    img = Image.open(dfi.at[g, "image_path"]).convert("RGB")
    img.thumbnail((640, 640))
    gs = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer, height_ratios=[1.0, 0.55],
                                          width_ratios=[1, 1, 1, 1.35], hspace=0.12, wspace=0.06)
    # photo
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    ax.set_title("photo" + ("  (- - text read by OCR)" if ocr is not None and text_boxes(ocr.get(g))[0] else ""),
                 fontsize=6.3, pad=2)
    if ocr is not None:
        W, H = img.size
        for x1, y1, x2, y2 in text_boxes(ocr.get(g))[0]:
            ax.add_patch(Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False, ec="#22d3ee",
                                   lw=0.9, ls="--"))
    ax.axis("off")
    if title:
        ax.text(0, 1.22, title, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    # heatmaps
    for j, st in enumerate(studies[:2]):
        a = st.attribution("full").get(g)
        ax = fig.add_subplot(gs[0, 1 + j])
        if a and "img_map" in a:
            ax.imshow(_heat(img, a["img_map"]))
            share = a["R_img"] / (a["R_img"] + a["R_text"] + 1e-12)
            ax.set_title(f"{NAMES.get(st.vlm.name, st.vlm.name)}: {share:.0%} on image", fontsize=6.3, pad=2)
        ax.axis("off")
    _pred_panel(fig.add_subplot(gs[0, 3]), studies, dfi, g)
    # text
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    w0, r0 = _words_and_rel(studies[0], dfi, g)
    if w0 is None:
        return
    s, e = _window(r0, n_words)
    y = 1.0
    for st in studies[:2]:
        w, r = _words_and_rel(st, dfi, g)
        if w is None:
            continue
        seg = r[s:e]
        seg = np.clip(seg / (np.percentile(r, 99.5) + 1e-12), 0, 1)
        words = (["…"] if s > 0 else []) + w[s:e] + (["…"] if e < len(w) else [])
        scores = ([0] if s > 0 else []) + list(seg) + ([0] if e < len(w) else [])
        ax.text(0, y, NAMES.get(st.vlm.name, st.vlm.name), fontsize=6.3, fontweight="bold", va="top",
                transform=ax.transAxes)
        y = _draw_words(ax, words, scores, x0=0.16, y0=y) - 0.2


def figure(studies, df, gids, path, ocr=None, titles=None, width=8.4):
    plt = _plt()
    from matplotlib import gridspec
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(width, 2.85 * len(gids) + 0.35))
    outer = gridspec.GridSpec(len(gids), 1, figure=fig, hspace=0.42)
    for i, g in enumerate(gids):
        t = (titles or [None] * len(gids))[i]
        draw_example(fig, outer[i], studies, df, g, ocr, title=t)
    cax = fig.add_axes([0.30, 0.02 if len(gids) > 1 else 0.03, 0.18, 0.012])
    cb = fig.colorbar(ScalarMappable(Normalize(0, 1), "inferno"), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1], labels=["low", "high"]); cb.ax.tick_params(labelsize=6)
    cax.set_title("image relevance", fontsize=6, pad=2)
    cax2 = fig.add_axes([0.58, 0.02 if len(gids) > 1 else 0.03, 0.18, 0.012])
    cb2 = fig.colorbar(ScalarMappable(Normalize(0, 1), "Blues"), cax=cax2, orientation="horizontal")
    cb2.set_ticks([0, 1], labels=["low", "high"]); cb2.ax.tick_params(labelsize=6)
    cax2.set_title("word relevance", fontsize=6, pad=2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.splitext(path)[0] + "." + ext, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def render_candidates(studies, df, out_dir, ocr=None, per_group=3):
    """One figure per candidate (to choose from) + a two-example figure for the paper."""
    C = candidates(studies, df, ocr)
    V.log("candidates:", {k: len(v) for k, v in C.items()})
    made = []
    for grp, lst in C.items():
        for k, g in enumerate(lst[:per_group]):
            p = os.path.join(out_dir, f"{grp}_{k + 1}.pdf")
            figure(studies, df, [g], p, ocr, titles=[GROUP_TITLE[grp]])
            made.append((grp, k + 1, g, p))
    pick = [(grp, C[grp][0]) for grp in ("image_fixes", "text_in_image", "image_breaks") if C.get(grp)][:2]
    if pick:
        figure(studies, df, [g for _, g in pick], os.path.join(out_dir, "fig_where_it_looks.pdf"), ocr,
               titles=[f"({'ab'[i]}) {GROUP_TITLE[grp]}" for i, (grp, _) in enumerate(pick)])
    return C, made