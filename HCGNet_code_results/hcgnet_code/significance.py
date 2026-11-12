"""Paired per-circuit significance tests on the seed-0 split (ID test and size-OOD pool).

For each target we compare absolute percentage errors of two models on the same circuits with the
Wilcoxon signed-rank test and report a 95% paired-bootstrap CI of the MAPE difference (B = 2000).
"""
import json, pickle
import numpy as np
from scipy.stats import wilcoxon
import train

S = train.load("data/sherbrooke/processed.pkl")
tr, va, te = train.split_id(S, 0)
ood = [r for r in S if r["ood"]]
P = pickle.load(open("results/preds_sherbrooke_s0.pkl", "rb"))
fm = train.fit_tab("gbdt", tr + va, 0)
rng = np.random.RandomState(0)


def ape(recs, pred):
    rt = train.to_raw(np.stack([r["y"] for r in recs])); rp = train.to_raw(pred)
    den = rt.copy(); den[:, 1] = np.maximum(den[:, 1], 1)
    return 100 * np.abs(rp - rt) / den


out = {}
for name, T, g in [("id", te, P["te_pred"]), ("ood", ood, P["ood_pred"])]:
    b = train.predict_tab(fm, T)
    models = {"hcgnet": g, "gbdt": b, "hcg_ens": (g + b) / 2}
    A = {m: ape(T, p) for m, p in models.items()}
    res = {}
    for a, c in [("hcg_ens", "gbdt"), ("hcg_ens", "hcgnet"), ("hcgnet", "gbdt")]:
        for k, t in enumerate(train.TASKS):
            d = A[a][:, k] - A[c][:, k]
            boot = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)]
            p = wilcoxon(A[a][:, k], A[c][:, k]).pvalue
            res[f"{a}_vs_{c}_{t}"] = dict(diff=round(float(d.mean()), 3), ci=[round(float(np.percentile(boot, 2.5)), 3),
                                          round(float(np.percentile(boot, 97.5)), 3)], p=float(p), n=len(d))
    out[name] = res
json.dump(out, open("results/significance_s0.json", "w"), indent=1)
for s in out:
    for k, v in out[s].items():
        print(s, k, v)
