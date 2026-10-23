"""Paper figures from results/tables: fig_confidence (Fig. 1), fig_topics (Fig. 2), fig_deletion (Fig. 4)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse, os
ap = argparse.ArgumentParser(description="Paper figures (Figs. 1, 2, 4) from the result tables.")
ap.add_argument("--tables", default=os.path.join(os.path.dirname(__file__), "..", "results", "tables"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "figures"))
a = ap.parse_args()
R = a.tables.rstrip("/") + "/"; F = a.out.rstrip("/") + "/"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42})
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b"
NAMES = {"qwen3vl8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B"}
W = 4.8

# ---------------- Fig: when does the image help (by text-only confidence)
t = pd.read_csv(R + "T4_by_text_confidence.csv")
fig, axes = plt.subplots(1, 2, figsize=(W, 1.85), sharey=True)
for ax, m in zip(axes, NAMES):
    d = t[t.model == m].reset_index(drop=True)
    x = np.arange(len(d))
    ax.bar(x - 0.19, d.gain * 100, 0.36, color=BLUE, label="image fixes text-only error")
    ax.bar(x + 0.19, -d.loss * 100, 0.36, color=ORANGE, label="image breaks correct text-only")
    ax.plot(x, d.swap_capture * 100, "o--", color=INK, ms=3, lw=0.9, label="captured by mismatched image")
    ax.axhline(0, color="#52514e", lw=0.6)
    labs = ["Q1", "Q2", "Q3–Q5\n(tied at p≈1)"] if len(d) == 3 else [f"Q{i+1}" for i in range(len(d))]
    ax.set_xticks(x, labs, fontsize=6.5)
    ax.set_title(NAMES[m], fontsize=8)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
axes[0].set_ylabel("% of articles in bin")
fig.text(0.5, 0.0, "quintiles of text-only confidence max$_c\\,p(c)$ (low → high)", ha="center", fontsize=7)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, 1.07), handlelength=1.4)
fig.tight_layout(w_pad=0.6, rect=(0, 0.04, 1, 0.95))
fig.savefig(F + "fig_confidence.pdf", bbox_inches="tight"); fig.savefig(F + "fig_confidence.png", dpi=220, bbox_inches="tight")

# ---------------- Fig: deletion check
t = pd.read_csv(R + "T7_deletion_check.csv")
combos = [("full", "text", "words removed\ninput: text+image"), ("full", "image", "image cells removed\ninput: text+image"),
          ("text_only", "text", "words removed\ninput: text only"), ("image_only", "image", "image cells removed\ninput: image only")]
fig, axes = plt.subplots(2, 4, figsize=(W, 2.45), sharex=True)
for r, m in enumerate(NAMES):
    for c, (ctx, mod, title) in enumerate(combos):
        ax = axes[r, c]
        d = t[(t.model == m) & (t.context == ctx) & (t.deleted == mod)].sort_values("level")
        x = d.level * 100
        sig = d.p_holm < 0.05
        ax.plot(x, d.dp_random * 100, "s--", color=GRAY, ms=3, lw=0.9, label="random, same amount")
        ax.plot(x, d.dp_evidence * 100, "-", color=BLUE, lw=1.2, label="most relevant (grad×input)")
        ax.scatter(x[sig], d.dp_evidence[sig] * 100, s=14, color=BLUE, zorder=3)
        ax.scatter(x[~sig], d.dp_evidence[~sig] * 100, s=14, facecolor="white", edgecolor=BLUE, zorder=3, lw=0.9)
        ax.set_xticks([10, 20, 40])
        ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.tick_params(labelsize=6)
        if r == 0:
            ax.set_title(title, fontsize=6.3)
        if c == 0:
            ax.set_ylabel(NAMES[m], fontsize=6.5)
        if r == 1:
            ax.set_xlabel("% removed", fontsize=6.5)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(h_pad=0.4, w_pad=0.3, rect=(0.03, 0, 1, 0.96))
fig.text(0.005, 0.5, "drop in $p(\\hat y)$ ×100", rotation=90, ha="left", va="center", fontsize=6.5)
fig.savefig(F + "fig_deletion.pdf", bbox_inches="tight"); fig.savefig(F + "fig_deletion.png", dpi=220, bbox_inches="tight")

# ---------------- Fig: per-topic effect of adding the image
t = pd.read_csv(R + "T5_by_category.csv")
t["d"] = (t.acc_full - t.acc_text) * 100
cats = list(t[t.model == "qwen3vl8b"].sort_values("d").category)
fig, ax = plt.subplots(figsize=(W * 0.62, 2.2))
y = np.arange(len(cats))
for k, (m, col) in enumerate(zip(NAMES, [BLUE, ORANGE])):
    d = t[t.model == m].set_index("category").loc[cats]
    ax.barh(y + (0.2 if k == 0 else -0.2), d.d, 0.38, color=col, label=NAMES[m])
ax.axvline(0, color="#52514e", lw=0.6)
ax.set_yticks(y, cats, fontsize=6.5)
ax.set_xlabel("accuracy(text+image) − accuracy(text only), points")
ax.grid(axis="x"); ax.set_axisbelow(True)
ax.legend(fontsize=6.5, loc="lower right")
fig.tight_layout()
fig.savefig(F + "fig_topics.pdf", bbox_inches="tight"); fig.savefig(F + "fig_topics.png", dpi=220, bbox_inches="tight")
print("ok")
