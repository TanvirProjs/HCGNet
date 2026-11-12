"""Fig. 3: parity plots of HCG-Ens (seed-0 split) and per-optimisation-level / per-family errors."""
import pickle, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import train

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"], "font.size": 6,
                     "axes.linewidth": 0.5, "mathtext.fontset": "stix"})
S = train.load("data/sherbrooke/processed.pkl")
tr, va, te = train.split_id(S, 0)
ood = [r for r in S if r["ood"]]
P = pickle.load(open("results/preds_sherbrooke_s0.pkl", "rb"))
fm = train.fit_tab("gbdt", tr + va, 0)
ens_te = (P["te_pred"] + train.predict_tab(fm, te)) / 2
ens_ood = (P["ood_pred"] + train.predict_tab(fm, ood)) / 2

out = {"by_ol": {}, "by_family": {}}
for ol in (1, 2, 3):
    idx = [i for i, r in enumerate(te) if r["opt_level"] == ol]
    m = train.metrics([te[i] for i in idx], ens_te[idx])
    out["by_ol"][ol] = {k: round(m[k]["mape"], 2) for k in train.TASKS}
for f in sorted(set(r["family"] for r in te)):
    idx = [i for i, r in enumerate(te) if r["family"] == f]
    m = train.metrics([te[i] for i in idx], ens_te[idx])
    out["by_family"][f] = {k: round(m[k]["mape"], 2) for k in train.TASKS}
json.dump(out, open("results/breakdown_s0.json", "w"), indent=1)
print(out)

fig, axs = plt.subplots(1, 3, figsize=(3.38, 1.42), dpi=600)
names = ["Depth", "$N_{2q}$", "$-\\ln$ ESP"]
for k, ax in enumerate(axs):
    allv = []
    for T, p, c, lab in [(ood, ens_ood, "#c9772b", "Size-OOD (21\u201328 q)"), (te, ens_te, "#2b4c7e", "ID test (4\u201320 q)")]:
        yt = train.to_raw(np.stack([r["y"] for r in T]))[:, k]; yp = train.to_raw(p)[:, k]
        if k == 1:
            yt, yp = yt + 1, yp + 1
        ax.scatter(yt, yp, s=0.6, c=c, alpha=0.5, lw=0, label=lab, rasterized=True)
        allv += [yt.min(), yp.min(), yt.max(), yp.max()]
    lo, hi = min(allv) * 0.7, max(allv) * 1.4
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], lw=0.5, color="k")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"({'abc'[k]}) " + names[k] + (" + 1" if k == 1 else ""), fontsize=6, pad=2)
    mi = train.metrics(te, ens_te)[train.TASKS[k]]; mo = train.metrics(ood, ens_ood)[train.TASKS[k]]
    ax.text(0.04, 0.96, f"ID  $R^2$={mi['r2']:.3f}\nOOD $R^2$={mo['r2']:.3f}", transform=ax.transAxes,
            fontsize=4.6, va="top", ha="left", linespacing=1.15)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5, length=2, pad=1, which="both")
axs[0].set_ylabel("Predicted", fontsize=6); axs[1].set_xlabel("Transpiled (label)", fontsize=6, labelpad=1)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, fontsize=5, frameon=False, markerscale=6, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout(pad=0.2, w_pad=0.3)
plt.savefig("results/fig_parity.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
