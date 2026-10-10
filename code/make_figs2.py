import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
NAME = {"hydro": "Hydrophone", "accel": "Accelerometer", "pressure": "Dynamic pressure"}
COL = {"hydro": "#2E6F9E", "accel": "#C8742B", "pressure": "#4A8C5A"}
M = ["hydro", "accel", "pressure"]
P = ["P1 random-window", "P2 grouped-record", "P2b grouped-experiment",
     "P3 leave-condition-out", "P4a LO->BR", "P4b BR->LO"]
SH = ["P1", "P2", "P2b", "P3", "P4a", "P4b"]

# ---- Fig 5: representation comparison (hydrophone) ----
sp = pd.read_csv("spec_results.csv")
rf = pd.read_csv("protocol_v2.csv"); rf = rf[rf.modality == "hydro"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, task, chance in zip(axes, ["binary", "multiclass"], [0.5, 0.2]):
    x = np.arange(len(P)); w = 0.38
    a = [sp[(sp.task == task) & (sp.protocol == p)]["balanced_acc"] for p in P]
    b = [rf[(rf.task == task) & (rf.protocol == p)]["balanced_acc"] for p in P]
    ax.bar(x - w/2, [v.mean() for v in b], w, yerr=[v.std() for v in b], capsize=2,
           label="Random Forest, 45 features", color="#9AB8CE", edgecolor="white",
           error_kw=dict(lw=0.7, ecolor="#444"))
    ax.bar(x + w/2, [v.mean() for v in a], w, yerr=[v.std() for v in a], capsize=2,
           label="MLP, log-mel spectrogram", color="#17457A", edgecolor="white",
           error_kw=dict(lw=0.7, ecolor="#444"))
    ax.axhline(chance, ls="--", lw=1, color="#777")
    ax.text(-0.45, chance + 0.02, "chance", fontsize=7, color="#777")
    ax.set_xticks(x); ax.set_xticklabels(SH, fontsize=8)
    ax.set_title("Binary" if task == "binary" else "Five-class", fontsize=9)
    ax.set_ylim(0, 1.08); ax.grid(axis="y", lw=0.4, alpha=0.3); ax.set_axisbelow(True)
axes[0].set_ylabel("Balanced accuracy")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("figs/fig5_representation.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig 6: (a) remedies, (b) per-class recall under transfer ----
r = pd.read_csv("remedy_results.csv")
c = pd.read_csv("perclass_results.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

MODES = ["R0", "R1", "R2", "R3"]
MLAB = ["baseline", "per-recording z", "per-domain z", "within-rec. ranks"]
ax = axes[0]; x = np.arange(len(MODES)); w = 0.26
for i, m in enumerate(M):
    mu = [r[(r.task == "binary") & (r["mode"] == md) & (r.modality == m)]["bal_acc"].mean() for md in MODES]
    sd = [r[(r.task == "binary") & (r["mode"] == md) & (r.modality == m)]["bal_acc"].std() for md in MODES]
    ax.bar(x + (i - 1) * w, mu, w, yerr=sd, capsize=2, color=COL[m], label=NAME[m],
           edgecolor="white", error_kw=dict(lw=0.7, ecolor="#444"))
ax.axhline(0.5, ls="--", lw=1, color="#777")
ax.set_xticks(x); ax.set_xticklabels(MLAB, fontsize=7, rotation=15)
ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0.4, 0.68)
ax.set_title("(a) Normalisation remedies, cross-topology", fontsize=8.5)
ax.legend(frameon=False, fontsize=7); ax.grid(axis="y", lw=0.4, alpha=0.3); ax.set_axisbelow(True)

ax = axes[1]
labs = ["NL", "OL", "LC", "CC", "GL"]
piv = c.groupby(["modality", "cls"])["recall"].mean().unstack()[labs]
x = np.arange(len(labs))
for i, m in enumerate(M):
    ax.bar(x + (i - 1) * w, piv.loc[m].values, w, color=COL[m], label=NAME[m], edgecolor="white")
ax.axhline(0.2, ls="--", lw=1, color="#777")
ax.text(len(labs) - 0.6, 0.215, "chance", fontsize=7, color="#777")
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
ax.set_ylabel("Recall"); ax.set_ylim(0, 0.72)
ax.set_title("(b) Per-class recall, cross-topology", fontsize=8.5)
ax.grid(axis="y", lw=0.4, alpha=0.3); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figs/fig6_remedies_perclass.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("ok")
