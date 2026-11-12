"""Dataset statistics and the seed-induced label-noise floor."""
import pickle, json, math
import numpy as np

out = {}
for dev in ("sherbrooke", "torino"):
    R = pickle.load(open(f"data/{dev}/processed.pkl", "rb"))
    fam = {}
    for r in R:
        fam.setdefault(r["family"], []).append(r)
    stats = {}
    for f, rs in sorted(fam.items()):
        stats[f] = dict(count=len(rs), nmin=min(r["n"] for r in rs), nmax=max(r["n"] for r in rs),
                        logical_cx_med=float(np.median([r["graph"]["logical"]["cx"] for r in rs])),
                        depth_med=float(np.median([r["raw"]["depth"] for r in rs])),
                        twoq_med=float(np.median([r["raw"]["twoq"] for r in rs])),
                        esp_med=float(np.median([math.exp(r["raw"]["log_esp"]) for r in rs])),
                        t_med=float(np.median([r["t_transpile"] for r in rs])))
    allr = R
    stats["_all"] = dict(count=len(allr), n_id=sum(r["uid"] < 10**6 for r in allr), n_ood=sum(r["uid"] >= 10**6 for r in allr),
                         depth_range=[int(min(r["raw"]["depth"] for r in allr)), int(max(r["raw"]["depth"] for r in allr))],
                         twoq_range=[int(min(r["raw"]["twoq"] for r in allr)), int(max(r["raw"]["twoq"] for r in allr))],
                         esp_range=[float(min(math.exp(r["raw"]["log_esp"]) for r in allr)), float(max(math.exp(r["raw"]["log_esp"]) for r in allr))],
                         blocks_mean=float(np.mean([r["graph"]["G"] for r in allr])),
                         cx_mean=float(np.mean([r["graph"]["logical"]["cx"] for r in allr])),
                         t_total_h=float(sum(r["t_transpile"] for r in allr) / 3600))
    # noise floor: circuits transpiled with 5 seeds
    multi = [r for r in R if len(r["seeds"]) >= 5]
    nf = {}
    for k, name in enumerate(["depth", "twoq", "esp"]):
        apes, cvs = [], []
        for r in multi:
            v = np.array([s[k] for s in r["seeds"]], float)
            if name == "esp":
                v = -v                    # -ln ESP
            if name == "twoq":
                v = np.maximum(v, 1)
            for i in range(len(v)):
                rest = np.delete(v, i)
                apes.append(abs(rest.mean() - v[i]) / v[i])
            cvs.append(v.std() / v.mean())
        nf[name] = dict(loo_mape=100 * float(np.mean(apes)), mean_cv=100 * float(np.mean(cvs)))
    nf["n_circuits"] = len(multi)
    out[dev] = dict(stats=stats, noise_floor=nf)
json.dump(out, open("results/dataset_stats.json", "w"), indent=1)
print(json.dumps({d: dict(all=out[d]["stats"]["_all"], nf=out[d]["noise_floor"]) for d in out}, indent=1))
for f, s in out["sherbrooke"]["stats"].items():
    print(f, s)
