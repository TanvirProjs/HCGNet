"""Fig. 2: MAPE versus circuit width (seed-0 split; ID test n<=20, size-OOD n=21..28)."""
import pickle, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import train

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"], "font.size": 6.5,
                     "axes.linewidth": 0.5, "mathtext.fontset": "stix", "xtick.major.width": 0.5, "ytick.major.width": 0.5})
S = train.load("data/sherbrooke/processed.pkl")
tr, va, te = train.split_id(S, 0)
ood = [r for r in S if r["ood"]]
P = pickle.load(open("results/preds_sherbrooke_s0.pkl", "rb"))
assert P["te_uid"] == [r["uid"] for r in te]
fm = train.fit_tab("gbdt", tr + va, 0)
recs = te + ood
g = np.concatenate([P["te_pred"], P["ood_pred"]])
b = train.predict_tab(fm, recs)
yt = np.stack([r["y"] for r in recs]); n = np.array([r["n"] for r in recs])
rt = train.to_raw(yt)
bins = [(4, 8), (9, 12), (13, 16), (17, 20), (21, 24), (25, 28)]
labels = ["4–8", "9–12", "13–16", "17–20", "21–24", "25–28"]
nf = json.load(open("results/dataset_stats.json"))["sherbrooke"]["noise_floor"]
fig, axs = plt.subplots(1, 3, figsize=(3.38, 1.40), dpi=600, sharex=True)
titles = ["Depth", "$N_{2q}$", "$-\\ln$ ESP"]
cols = {"GBDT": "#c26a1f", "HCG-Net": "#2b5c8a", "HCG-Ens": "#6b3d7a"}
mk = {"GBDT": "s", "HCG-Net": "o", "HCG-Ens": "^"}
for k, ax in enumerate(axs):
    for name, p in [("GBDT", b), ("HCG-Net", g), ("HCG-Ens", (g + b) / 2)]:
        rp = train.to_raw(p)
        den = np.maximum(rt[:, k], 1) if k == 1 else rt[:, k]
        ape = np.abs(rp[:, k] - rt[:, k]) / den * 100
        ys = [ape[(n >= lo) & (n <= hi)].mean() for lo, hi in bins]
        ax.plot(range(6), ys, marker=mk[name], ms=2.2, lw=0.8, color=cols[name], label=name)
    ax.axhline(nf[["depth", "twoq", "esp"][k]]["loo_mape"], ls="--", lw=0.6, color="gray", label="Seed-noise floor")
    ax.axvspan(3.5, 5.5, color="#eeeeee", lw=0)
    ax.set_title(f"({'abc'[k]}) " + titles[k], fontsize=6.2, pad=2)
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=60, fontsize=5)
    ax.tick_params(labelsize=5, length=2, pad=1)
    ax.set_ylim(bottom=0)
axs[0].set_ylabel("MAPE (%)", fontsize=6)
axs[1].set_xlabel("Logical qubits (shaded: unseen widths)", fontsize=6, labelpad=1)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, fontsize=5, frameon=False, bbox_to_anchor=(0.5, 1.11))
plt.tight_layout(pad=0.2, w_pad=0.4)
plt.savefig("results/fig_size.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("ok")
