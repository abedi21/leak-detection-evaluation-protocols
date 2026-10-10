import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np, pandas as pd, os

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("figs", exist_ok=True)
d = pd.read_csv("protocol_v2.csv")
M = ["hydro", "accel", "pressure"]
NAME = {"hydro": "Hydrophone", "accel": "Accelerometer", "pressure": "Dynamic pressure"}
COL = {"hydro": "#2E6F9E", "accel": "#C8742B", "pressure": "#4A8C5A"}
PSHORT = {"P1 random-window": "P1\nrandom\nwindow", "P2 grouped-record": "P2\ngrouped\nrecord",
          "P2b grouped-experiment": "P2b\ngrouped\nexperiment", "P3 leave-condition-out": "P3\nleave-cond.\nout",
          "P4a LO->BR": "P4a\nLO→BR", "P4b BR->LO": "P4b\nBR→LO"}
P = list(PSHORT)

# ---------- Fig 1: protocol schematic ----------
fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.5))
titles = ["P1 random-window", "P2 grouped-record", "P3 leave-condition-out", "P4 cross-topology"]
rng = np.random.default_rng(3)
for ax, t in zip(axes, titles):
    ax.set_xlim(-.4, 6.4); ax.set_ylim(-1.5, 4.2); ax.axis("off")
    ax.set_title(t, fontsize=9, pad=6)
    # 4 recordings x 6 windows
    for r in range(4):
        for w in range(6):
            if t == "P1 random-window":
                test = rng.random() < 0.25
            elif t == "P2 grouped-record":
                test = r == 3
            elif t == "P3 leave-condition-out":
                test = w >= 4
            else:
                test = r >= 2
            ax.add_patch(mp.Rectangle((w, 3 - r * 0.8), 0.86, 0.62,
                         fc="#D94F4F" if test else "#BFD3E6",
                         ec="white", lw=0.8))
    if t == "P2 grouped-record":
        ax.text(-0.3, 0.55, "rec.", rotation=90, fontsize=7, va="center", ha="center", color="#555")
    if t == "P3 leave-condition-out":
        ax.annotate("", xy=(4, -0.25), xytext=(6, -0.25),
                    arrowprops=dict(arrowstyle="-", lw=1.2, color="#D94F4F"))
        ax.text(5, -0.75, "unseen condition", fontsize=7, ha="center", color="#D94F4F")
    if t == "P4 cross-topology":
        ax.text(3, 3.95, "looped (train)", fontsize=7, ha="center", color="#555")
        ax.text(3, -0.75, "branched (test)", fontsize=7, ha="center", color="#D94F4F")
handles = [mp.Patch(fc="#BFD3E6", label="training window"), mp.Patch(fc="#D94F4F", label="test window")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.04), fontsize=8)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("figs/fig1_protocols.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 2 & 3: results ----------
def results_fig(task, chance, fname, ylab):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(P)); w = 0.26
    for i, m in enumerate(M):
        mu = [d[(d.task == task) & (d.modality == m) & (d.protocol == p)]["balanced_acc"].mean() for p in P]
        sd = [d[(d.task == task) & (d.modality == m) & (d.protocol == p)]["balanced_acc"].std() for p in P]
        ax.bar(x + (i - 1) * w, mu, w, yerr=sd, capsize=2.5, label=NAME[m],
               color=COL[m], edgecolor="white", linewidth=0.6,
               error_kw=dict(lw=0.8, ecolor="#444"))
    ax.axhline(chance, ls="--", lw=1, color="#777")
    ax.text(-0.45, chance + 0.02, "chance", fontsize=7.5, color="#777", ha="left")
    ax.set_xticks(x); ax.set_xticklabels([PSHORT[p] for p in P], fontsize=7.5)
    ax.set_ylabel(ylab); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    ax.grid(axis="y", lw=0.4, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(fname, dpi=300, bbox_inches="tight"); plt.close(fig)

results_fig("binary", 0.5, "figs/fig2_binary.png", "Balanced accuracy")
results_fig("multiclass", 0.2, "figs/fig3_multiclass.png", "Balanced accuracy")

# ---------- Fig 4: P1 -> P4 degradation slope ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
for ax, task, chance in zip(axes, ["binary", "multiclass"], [0.5, 0.2]):
    steps = ["P1 random-window", "P2 grouped-record", "P3 leave-condition-out"]
    for m in M:
        ys = [d[(d.task == task) & (d.modality == m) & (d.protocol == p)]["balanced_acc"].mean() for p in steps]
        p4 = d[(d.task == task) & (d.modality == m) & (d.protocol.str.startswith("P4"))]["balanced_acc"].mean()
        ys.append(p4)
        ax.plot(range(4), ys, "o-", color=COL[m], label=NAME[m], lw=1.6, ms=5)
    ax.axhline(chance, ls="--", lw=1, color="#777")
    ax.set_xticks(range(4)); ax.set_xticklabels(["P1", "P2", "P3", "P4"], fontsize=8)
    ax.set_title("Binary" if task == "binary" else "Five-class", fontsize=9)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", lw=0.4, alpha=0.3); ax.set_axisbelow(True)
axes[0].set_ylabel("Balanced accuracy")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figs/fig4_degradation.png", dpi=300, bbox_inches="tight"); plt.close(fig)

print("figures written:", sorted(os.listdir("figs")))
