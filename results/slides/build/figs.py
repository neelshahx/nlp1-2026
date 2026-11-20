import glob
import matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

OUT = "results/slides/figures"
A, B, GREY = "#2b6cb0", "#dd6b20", "#a0aec0"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})

# 1. training curve
fig, ax = plt.subplots(figsize=(6.4, 4))
for name, col in [("lstm", A), ("max", B)]:
    (d,) = glob.glob(f"results/snli-glove-bowman-{name}-freeze_embeddings=False-*")
    h = pd.read_csv(d + "/history.csv")
    g = h.groupby("epoch").agg(tr=("train_accuracy", "mean"), va=("val_accuracy", "mean"), n=("seed", "size"))
    g = g[g.n >= 5]
    label = "LSTM" if name == "lstm" else "max pool"
    ax.plot(g.index, g.tr, color=col, lw=2, label=f"{label} train")
    ax.plot(g.index, g.va, color=col, lw=2, ls="--", label=f"{label} val")
    best = h.loc[h.groupby(["seed", "fold"]).val_accuracy.idxmax(), "epoch"].mean()  # mean kept epoch
    ax.axvline(best, ymax=.08, color=col, lw=1.5)
    ax.text(best, ax.get_ylim()[0], f" {best:.1f} " , color=col, fontsize=9,
            ha="right" if name == "lstm" else "left", va="bottom")
ax.set(xlabel="epoch", ylabel="accuracy"); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.legend(frameon=False, fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/training_curve.png"); plt.close(fig)

# 2. ROPE stacked bars (seed-averaged, ±0.5 pt), sorted by P(ROPE)
rows = [("BERT: full vs hypothesis-only", 1.00, .00, .00), ("max vs mean", .99, .01, .00),
        ("LSTM vs sum", .99, .00, .00), ("max: trainable vs frozen", .97, .03, .00),
        ("sum: trainable vs frozen", .95, .03, .01), ("BERT: matching vs concat", .86, .11, .03),
        ("BERT vs GloVe (frozen, mean)", .81, .13, .06), ("LSTM: trainable vs frozen", .36, .55, .09),
        ("LSTM vs max pool", .17, .72, .11)]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for i, (lab, l, r, rt) in enumerate(rows):
    y = len(rows) - 1 - i
    for left, w, col in [(0, l, A), (l, r, GREY), (l + r, rt, B)]:
        ax.barh(y, w, left=left, color=col, height=.6)
        if w >= .08:
            ax.text(left + w / 2, y, f"{w:.0%}", ha="center", va="center", color="white", fontsize=11)
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
ax.set_xlim(0, 1); ax.set_xlabel("posterior probability, ROPE ±0.5 pt")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
from matplotlib.patches import Patch
ax.legend([Patch(color=A), Patch(color=GREY), Patch(color=B)], ["left better", "equivalent", "right better"],
          frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.4, 1.0), fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/rope_decisions.png"); plt.close(fig)

# 3. BDA forest plot
w = pd.read_csv("results/figures/bda-v1.1/weight_differences_all.csv")
feats = ["cue_misleading", "cue_gold", "negation", "overlap", "hypothesis_len"]
fig, ax = plt.subplots(figsize=(6.4, 4))
for comp, col, off in [("lstm - bow-loglin", A, .15), ("lstm - sum", B, -.15)]:
    d = w[w.comparison == comp].set_index("feature").loc[feats]
    ys = [len(feats) - 1 - i + off for i in range(len(feats))]
    ax.errorbar(d["mean"], ys, xerr=[d["mean"] - d.low_95, d.high_95 - d["mean"]], fmt="o", color=col,
                capsize=3, label=comp.replace("lstm", "LSTM").replace("bow-loglin", "BoW log-lin"))
ax.axvline(0, color=GREY, lw=1)
ax.set_yticks(range(len(feats)), [f.replace("_", " ") for f in feats][::-1])
ax.set_xlabel("weight difference (logit of correct)")
ax.legend(frameon=False, fontsize=11, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/bda_weights.png"); plt.close(fig)

print("ok")
