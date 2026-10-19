"""Regenerate the refocused-paper figures from counts quoted in paper/main.tex.
Usage: python3 paper/figures/src/make_refocus_figs.py   (writes into paper/figures/)
All counts are measured rows from the paper scoreboard; nothing here is new data."""
from math import sqrt
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

C = {"base": "#8c8c8c", "ctrl": "#2b8a3e", "upper": "#74b816", "syntax": "#c92a2a", "train": "#e67700"}

# ---- Figure: ladder (n=200) ----
rows = [  # label, correct, kind
    ("Native API (NFC)", 54, "base"),
    ("JSON text (IFFC)", 61, "base"),
    ("JSON text + stop on repeats", 65, "ctrl"),
    ("JSON text + state summary", 68, "ctrl"),
    ("Minimal recipe\n(JSON text + stop + card)", 71, "ctrl"),
    ("Vote of five on state summary\n(~7x generation time)", 73, "upper"),
    ("Full stack (headline)\n(summary + stop + card)", 77, "ctrl"),
    ("Constrained decoding", 21, "syntax"),
    ("Tool-name automaton", 27, "syntax"),
    ("Best fine-tune (SFT)", 30, "train"),
    ("Preference-tuned (DPO)", 53, "train"),
]
# Bottom to top. Vote (73) sits below the minimal recipe (71) so the
# last bars are not a rising stack of added components. Each bar is its own run.
order = [7, 8, 9, 10, 0, 1, 2, 3, 5, 4, 6]
fig, ax = plt.subplots(figsize=(6.5, 3.55))
for y, i in enumerate(order):
    lab, k, kind = rows[i]
    acc = 100 * k / 200; lo, hi = wilson(k, 200)
    hl = lab.startswith("Full stack")
    ax.barh(y, acc, color=C[kind], height=0.65, edgecolor="black" if hl else "none", linewidth=1.8 if hl else 0)
    ax.errorbar(acc, y, xerr=[[acc - lo], [hi - acc]], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.text(hi + 0.8, y, f"{k}/200 = {acc:.1f}%", va="center", fontsize=9, fontweight="bold" if hl else "normal",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([rows[i][0] for i in order], fontsize=9)
for t in ax.get_yticklabels():
    if t.get_text().startswith("Full stack"): t.set_fontweight("bold")
ax.axvline(100 * 61 / 200, color="black", ls=":", lw=0.9, zorder=1)
ax.set_xlim(0, 60); ax.set_ylim(-0.6, len(order) - 0.1); ax.set_xlabel("Tasks fully correct, % of 200 (error bars: Wilson 95% interval)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C[k], label=l) for k, l in [("base", "Output format only"), ("ctrl", "Harness control"),
          ("upper", "Majority vote (costly)"), ("syntax", "Forced valid syntax"), ("train", "Training")]] + [plt.Line2D([], [], color="black", ls=":", lw=0.9, label="JSON-text baseline")],
          loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=3, fontsize=9, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "ladder_n200.pdf"); plt.close(fig)

# ---- Figure: 20-task gate ----
g = [("Untuned\n(bar)", 17, "ctrl"), ("Rebuilt\nmix", 2, "train"),
     ("Rebuilt,\nno card", 1, "train"), ("Rebuilt,\nlayout", 2, "train"),
     ("Null\ntest", 15, "base"), ("Self-\ndistill", 8, "train"),
     ("Self-distill,\nstops cap", 14, "train"), ("Self-distill,\nlook-ups", 12, "train")]
fig, ax = plt.subplots(figsize=(6.5, 2.4))
for x, (lab, k, kind) in enumerate(g):
    lo, hi = wilson(k, 20)
    ax.bar(x, k, color=C[kind], width=0.65)
    ax.errorbar(x, k, yerr=[[k - lo / 5], [hi / 5 - k]], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.text(x, hi / 5 + 0.35, f"{k}/20", ha="center", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=5)
ax.axhline(17, color="black", ls=":", lw=0.9); ax.text(2.0, 17.4, "pass bar = 17/20", fontsize=9)
ax.set_xticks(range(len(g))); ax.set_xticklabels([x[0] for x in g], fontsize=8.5)
ax.set_ylim(0, 22); ax.set_ylabel("Tasks passed (of 20)")
fig.tight_layout(); fig.savefig(OUT / "gate20.pdf"); plt.close(fig)

# ---- Figure: failure anatomy (primary label per failed task) ----
f = [  # label, empty, under, state
    ("Native\nAPI", 59, 19, 68), ("JSON\ntext", 41, 38, 60), ("State\nsummary", 31, 37, 64),
    ("Minimal\nrecipe", 17, 41, 71), ("Full\nstack*", 20, 35, 68), ("Vote\nof five", 36, 38, 53),
    ("Constr.\ndecoding", 14, 70, 95), ("Name\nautomaton", 17, 54, 102), ("Best\nfine-tune", 2, 59, 109)]
fig, ax = plt.subplots(figsize=(7.4, 2.55))
xs = range(len(f)); b = [0] * len(f)
for j, (name, col) in enumerate([("Empty turn (no call when one was needed)", "#adb5bd"),
                                 ("Under-call (required tool never ran)", "#f08c00"),
                                 ("State mismatch (final state wrong)", "#1971c2")]):
    v = [r[j + 1] for r in f]
    ax.bar(xs, v, bottom=b, color=col, width=0.65, label=name)
    b = [p + q for p, q in zip(b, v)]
for x, t in zip(xs, b): ax.text(x, t + 2, str(t), ha="center", fontsize=9)
ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in f], fontsize=8)
for t in ax.get_xticklabels():
    if t.get_text().startswith("Full stack"): t.set_fontweight("bold")
ax.set_ylabel("Failed tasks (of 200)"); ax.set_ylim(0, 200)
ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, handlelength=1.0, columnspacing=0.8)
fig.tight_layout(); fig.savefig(OUT / "failure_anatomy.pdf", bbox_inches="tight"); plt.close(fig)
print("ok")
