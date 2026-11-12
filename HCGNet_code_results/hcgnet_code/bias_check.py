"""Mean log-error (bias) of the seed-0 HCG-Net and GBDT on ID test and size-OOD pools."""
import train, pickle, json, numpy as np
S = train.load("data/sherbrooke/processed.pkl"); tr, va, te = train.split_id(S, 0); ood = [r for r in S if r["ood"]]
P = pickle.load(open("results/preds_sherbrooke_s0.pkl", "rb")); fm = train.fit_tab("gbdt", tr + va, 0)
out = {}
for nm, T, g in [("id", te, P["te_pred"]), ("ood", ood, P["ood_pred"])]:
    y = np.stack([r["y"] for r in T]); b = train.predict_tab(fm, T)
    out[nm] = dict(hcgnet=(g - y).mean(0).round(4).tolist(), gbdt=(b - y).mean(0).round(4).tolist())
json.dump(out, open("results/bias_s0.json", "w"), indent=1); print(out)
