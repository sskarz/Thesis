"""Regenerate thesis/figures/ladder_n200.pdf (Fig. 4) from counts in the thesis scoreboard.

Usage: python3 thesis/figures/src/make_ladder.py   (needs matplotlib; writes thesis/figures/)

Same counts and colors as paper/figures/src/make_refocus_figs.py, with thesis names,
larger type, and all four paired runs of the compact-list test shown as separate bars.
Nothing here is new data: every count is a row of the scoreboard table.
"""
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parents[1]
plt.rcParams.update({
    "font.size": 11, "xtick.labelsize": 10.5, "ytick.labelsize": 11, "legend.fontsize": 10.5,
    "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


C = {"base": "#8c8c8c", "ctrl": "#2b8a3e", "upper": "#74b816", "syntax": "#c92a2a", "train": "#e67700"}
FIRST = "Full stack, first run"

# Bottom to top. Each bar is its own run on the same 200 tasks, not a stack of the bars below.
rows = [  # label, correct of 200, kind  (scoreboard run id in the comment)
    ("Constrained decoding", 21, "syntax"),                 # 8
    ("Tool-name automaton", 27, "syntax"),                  # 46
    ("Best fine-tune (SFT)", 30, "train"),                  # 15
    ("Preference-tuned (DPO)", 53, "train"),                # 51
    ("Native API", 54, "base"),                             # 4
    ("JSON text", 61, "base"),                              # 5
    ("JSON text + stop on repeats", 65, "ctrl"),            # 30
    ("JSON text + state summary", 68, "ctrl"),              # 10
    ("Vote of five (about 7x time)", 73, "upper"),          # 11
    ("Minimal recipe", 71, "ctrl"),                         # 42
    (FIRST, 77, "ctrl"),                                    # 78
    ("Full stack, original list, pair 1", 71, "ctrl"),      # 82
    ("Full stack, original list, pair 2", 72, "ctrl"),      # 85
    ("Full stack, compact list, pair 1", 84, "ctrl"),       # 83
    ("Full stack, compact list, pair 2", 88, "ctrl"),       # 84
]

fig, ax = plt.subplots(figsize=(6.5, 5.9))
for y, (lab, k, kind) in enumerate(rows):
    acc = 100 * k / 200
    lo, hi = wilson(k, 200)
    hl = lab == FIRST
    ax.barh(y, acc, color=C[kind], height=0.65, edgecolor="black" if hl else "none", linewidth=1.8 if hl else 0)
    ax.errorbar(acc, y, xerr=[[acc - lo], [hi - acc]], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.text(hi + 0.8, y, f"{k}/200 = {acc:.1f}%", va="center", fontsize=10.5,
            fontweight="bold" if hl else "normal",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=5, clip_on=False)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows])
for t in ax.get_yticklabels():
    if t.get_text() == FIRST:
        t.set_fontweight("bold")
ax.axhline(10.5, color="black", lw=0.5, ls="-", alpha=0.35)
ax.axvline(100 * 61 / 200, color="black", ls=":", lw=0.9, zorder=1)
ax.set_xlim(0, 72)
ax.set_ylim(-0.55, len(rows) - 0.35)
ax.set_xlabel("Tasks fully correct, % of 200 (error bars: Wilson 95% interval)", labelpad=6)
ax.legend(
    handles=[Patch(color=C[k], label=l) for k, l in [
        ("base", "Output format only"), ("ctrl", "Harness control"), ("upper", "Majority vote"),
        ("syntax", "Forced-valid output"), ("train", "Training")]]
    + [plt.Line2D([], [], color="black", ls=":", lw=0.9, label="JSON-text baseline")],
    loc="upper center", bbox_to_anchor=(0.40, -0.11), ncol=3, frameon=False,
    borderaxespad=0.2, columnspacing=1.0,
)
fig.tight_layout()
fig.savefig(OUT / "ladder_n200.pdf", bbox_inches="tight", pad_inches=0.10)
plt.close(fig)
print("wrote", OUT / "ladder_n200.pdf")
