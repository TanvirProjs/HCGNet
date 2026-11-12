"""Aggregate result JSONs into nums.json (all numbers quoted in the paper come from here)."""
import json, os, math
import numpy as np

R = "results"
L = lambda f: json.load(open(os.path.join(R, f)))
N = {}
T = ["depth", "twoq", "esp"]


def ms(vals, p=1):
    v = np.array(vals, float)
    return f"{v.mean():.{p}f}±{v.std(ddof=0):.{p}f}" if len(v) > 1 else f"{v.mean():.{p}f}"


def mean(vals):
    return float(np.mean(vals))


iid = L("iid_sherbrooke.json")
order = [("scaling", "Scaling law"), ("ridge", "Ridge"), ("rf", "Random forest"), ("mlp", "MLP"),
         ("gbdt", "GBDT"), ("hcgnet", "HCG-Net"), ("hcg_ens", "HCG-Ens")]
N["n_seeds"] = len(iid["hcgnet"])
# Table II (ID) and III (OOD)
t2 = [["Method", "Depth", "N{sub:2q}", "−ln ESP", "ρ{sub:ESP}", "Top 10%"]]
t3 = [["Method", "Depth", "N{sub:2q}", "−ln ESP", "ESP MAE"]]
best_id = {k: min(mean([s["id"][k]["mape"] for s in iid[m]]) for m, _ in order) for k in T}
best_ood = {k: min(mean([s["ood"][k]["mape"] for s in iid[m]]) for m, _ in order) for k in T}
for m, name in order:
    rs = iid[m]
    row = [name]
    for k in T:
        cell = ms([s["id"][k]["mape"] for s in rs], 2)
        row.append("{b:" + cell + "}" if abs(mean([s["id"][k]["mape"] for s in rs]) - best_id[k]) < 1e-9 else cell)
    row += [f"{mean([s['id']['esp']['rho'] for s in rs]):.4f}", f"{mean([s['id']['esp']['top10_recall'] for s in rs]):.1f}"]
    t2.append(row)
    row = [name]
    for k in T:
        cell = ms([s["ood"][k]["mape"] for s in rs], 2)
        row.append("{b:" + cell + "}" if abs(mean([s["ood"][k]["mape"] for s in rs]) - best_ood[k]) < 1e-9 else cell)
    row.append(f"{mean([s['ood']['esp']['esp_mae'] for s in rs]):.4f}")
    t3.append(row)
    for k in T:
        N[f"{m}_id_{k}"] = f"{mean([s['id'][k]['mape'] for s in rs]):.2f}"
        N[f"{m}_ood_{k}"] = f"{mean([s['ood'][k]['mape'] for s in rs]):.2f}"
        N[f"{m}_id_{k}_r2"] = f"{mean([s['id'][k]['r2'] for s in rs]):.4f}"
        N[f"{m}_id_{k}_w10"] = f"{mean([s['id'][k]['within10'] for s in rs]):.1f}"
        N[f"{m}_id_{k}_med"] = f"{mean([s['id'][k]['medape'] for s in rs]):.2f}"
    N[f"{m}_top10"] = f"{mean([s['id']['esp']['top10_recall'] for s in rs]):.1f}"
    N[f"{m}_id_rho"] = f"{mean([s['id']['esp']['rho'] for s in rs]):.4f}"
    N[f"{m}_id_espmae"] = f"{mean([s['id']['esp']['esp_mae'] for s in rs]):.4f}"
    N[f"{m}_ood_espmae"] = f"{mean([s['ood']['esp']['esp_mae'] for s in rs]):.4f}"
    if "train_s" in rs[0]:
        N[f"{m}_train_s"] = f"{mean([s['train_s'] for s in rs]):.0f}"
N["table2"], N["table3"] = t2, t3
for k in T:
    N[f"hcgnet_s0_id_{k}"] = f"{iid['hcgnet'][0]['id'][k]['mape']:.2f}"

# noise floor & dataset
ds = L("dataset_stats.json")
for dev in ("sherbrooke", "torino"):
    nf = ds[dev]["noise_floor"]
    for k in T:
        N[f"nf_{dev}_{k}"] = f"{nf[k]['loo_mape']:.2f}"
        N[f"cv_{dev}_{k}"] = f"{nf[k]['mean_cv']:.2f}"
    N[f"nf_{dev}_n"] = nf["n_circuits"]
    a = ds[dev]["stats"]["_all"]
    N[f"{dev}_count"] = a["count"]; N[f"{dev}_nid"] = a["n_id"]; N[f"{dev}_nood"] = a["n_ood"]
    N[f"{dev}_depth_max"] = a["depth_range"][1]; N[f"{dev}_twoq_max"] = a["twoq_range"][1]
    N[f"{dev}_cx_mean"] = f"{a['cx_mean']:.1f}"; N[f"{dev}_blocks_mean"] = f"{a['blocks_mean']:.1f}"
    N[f"{dev}_esp_min"] = f"{a['esp_range'][0]:.1e}"
fam_names = dict(adder="Adders", bv="Bernstein–Vazirani", ghz="GHZ", qaoa="QAOA (MaxCut)", qft="QFT",
                 qv="Quantum volume", random="Random", su2="EfficientSU2", trotter="Heisenberg Trotter")
N["fam_names"] = fam_names
t1 = [["Family", "#", "CX{sub:L}", "Depth", "N{sub:2q}", "ESP", "t (ms)"]]
for f, s in ds["sherbrooke"]["stats"].items():
    if f.startswith("_"):
        continue
    t1.append([fam_names[f], str(s["count"]), f"{s['logical_cx_med']:.0f}", f"{s['depth_med']:.0f}",
               f"{s['twoq_med']:.0f}", f"{s['esp_med']:.3f}", f"{1000 * s['t_med']:.0f}"])
N["table1"] = t1
# ratios to the noise floor
for m in ("hcgnet", "hcg_ens", "gbdt"):
    for k in T:
        N[f"ratio_{m}_{k}"] = f"{float(N[f'{m}_id_{k}']) / float(N[f'nf_sherbrooke_{k}']):.2f}"

# timing
if os.path.exists(os.path.join(R, "timing.json")):
    t = L("timing.json")
    N["t_transpile_ms"] = f"{1000 * t['transpile_mean']:.0f}"
    N["t_transpile_med_ms"] = f"{1000 * t['transpile_median']:.0f}"
    for ol, v in t["transpile_by_ol"].items():
        N[f"t_ol{ol}_ms"] = f"{1000 * v:.0f}"
    N["t_feat_ms"] = f"{1000 * t['feat_mean']:.2f}"
    N["t_gnn1_ms"] = f"{1000 * t['gnn_single_mean']:.2f}"
    N["t_gnnb_ms"] = f"{1000 * t['gnn_batch_per_circ']:.2f}"
    N["speedup1"] = f"{t['speedup_single']:.0f}"
    N["speedupb"] = f"{t['speedup_batch']:.0f}"

# ablation
if os.path.exists(os.path.join(R, "ablation.json")):
    ab = L("ablation.json")
    nseed = len(ab.get("no_seq", []))
    full = iid["hcgnet"][:nseed]
    rows = [["Variant", "ID test", "", "", "Size-OOD", "", ""], ["", "Depth", "N{sub:2q}", "−ln ESP", "Depth", "N{sub:2q}", "−ln ESP"]]
    var = [("full", "HCG-Net (full)"), ("no_seq", "− block→block edges"), ("no_qubit", "− qubit nodes"),
           ("no_glob", "− global descriptors"), ("sum_readout", "+ sum readout"), ("no_prior", "− structural prior"),
           ("gbdt", "GBDT (with prior)"), ("gbdt_no_prior", "GBDT − structural prior"), ("single_task", "Single-task (3 nets)†")]
    for v, name in var:
        if v == "full":
            rs = full
        elif v == "gbdt":
            rs = iid["gbdt"][:nseed]
        else:
            rs = ab.get(v, [])
        if not rs:
            continue
        row = [name] + [f"{mean([s['id'][k]['mape'] for s in rs]):.2f}" for k in T]
        row += [f"{mean([s['ood'][k]['mape'] for s in rs]):.2f}" if rs[0].get("ood") else "–" for k in T]
        rows.append(row)
        for k in T:
            N[f"abl_{v}_id_{k}"] = f"{mean([s['id'][k]['mape'] for s in rs]):.2f}"
            if rs[0].get("ood"):
                N[f"abl_{v}_ood_{k}"] = f"{mean([s['ood'][k]['mape'] for s in rs]):.2f}"
    N["table4"] = rows; N["abl_nseed"] = nseed

# LOFO
if os.path.exists(os.path.join(R, "lofo.json")):
    lo = L("lofo.json")
    rows = [["Held-out family", "Depth", "", "N{sub:2q}", "", "−ln ESP", ""],
            ["", "GBDT", "HCG", "GBDT", "HCG", "GBDT", "HCG"]]
    for f, v in lo.items():
        row = [fam_names[f]]
        for k in T:
            a, b = v["gbdt"][k]["mape"], v["hcgnet"][k]["mape"]
            row += ["{b:" + f"{a:.1f}" + "}" if a < b else f"{a:.1f}", "{b:" + f"{b:.1f}" + "}" if b < a else f"{b:.1f}"]
        rows.append(row)
        for m in ("gbdt", "hcgnet", "hcg_ens"):
            for k in T:
                N[f"lofo_{f}_{m}_{k}"] = f"{v[m][k]['mape']:.1f}"
    for m in ("gbdt", "hcgnet", "hcg_ens"):
        for k in T:
            N[f"lofo_mean_{m}_{k}"] = f"{mean([v[m][k]['mape'] for v in lo.values()]):.1f}"
            N[f"lofo_median_{m}_{k}"] = f"{float(np.median([v[m][k]['mape'] for v in lo.values()])):.1f}"
    rows.append(["Mean"] + sum([[N[f"lofo_mean_gbdt_{k}"], N[f"lofo_mean_hcgnet_{k}"]] for k in T], []))
    rows.append(["Median"] + sum([[N[f"lofo_median_gbdt_{k}"], N[f"lofo_median_hcgnet_{k}"]] for k in T], []))
    N["lofo_wins"] = sum(1 for v in lo.values() for k in T if v["hcgnet"][k]["mape"] < v["gbdt"][k]["mape"])
    N["lofo_cells"] = 3 * len(lo)
    N["table5"] = rows

# Torino
if os.path.exists(os.path.join(R, "transfer_torino.json")):
    it = L("iid_torino.json"); tf = L("transfer_torino.json")
    rows = [["Setting", "Depth", "N{sub:2q}", "−ln ESP"]]
    for m, name in [("ridge", "Ridge"), ("gbdt", "GBDT"), ("hcgnet", "HCG-Net"), ("hcg_ens", "HCG-Ens")]:
        rows.append([name + " (full data)"] + [ms([s["id"][k]["mape"] for s in it[m]], 2) for k in T])
        for k in T:
            N[f"tor_{m}_id_{k}"] = f"{mean([s['id'][k]['mape'] for s in it[m]]):.2f}"
            N[f"tor_{m}_ood_{k}"] = f"{mean([s['ood'][k]['mape'] for s in it[m]]):.2f}"
    names = {"zero_shot": "Zero-shot from Eagle", "scratch_100": "Scratch, 100 circuits", "finetune_100": "Fine-tune, 100 circuits",
             "scratch_300": "Scratch, 300 circuits", "finetune_300": "Fine-tune, 300 circuits",
             "finetune_all": "Fine-tune, all circuits"}
    for m, name in names.items():
        if m in tf:
            rows.append([name] + [ms([s[k]["mape"] for s in tf[m]], 2) for k in T])
            for k in T:
                N[f"tf_{m}_{k}"] = f"{mean([s[k]['mape'] for s in tf[m]]):.2f}"
    for k in T:
        N[f"tf_scratch_all_{k}"] = f"{mean([s[k]['mape'] for s in tf['scratch_all']]):.2f}"
    N["table6"] = rows; N["tor_nseed"] = len(it["hcgnet"]); N["tor_ntrain"] = int(0.7 * 2700)

if os.path.exists(os.path.join(R, "breakdown_s0.json")):
    bd = L("breakdown_s0.json")
    for k in T:
        v = [bd["by_ol"][o][k] for o in ("1", "2", "3")]
        N[f"ol_min_{k}"], N[f"ol_max_{k}"] = f"{min(v):.2f}", f"{max(v):.2f}"
        fam = {f: x[k] for f, x in bd["by_family"].items()}
        fmin, fmax = min(fam, key=fam.get), max(fam, key=fam.get)
        N[f"fam_min_{k}"], N[f"fam_min_{k}_name"] = f"{fam[fmin]:.2f}", fam_names[fmin]
        N[f"fam_max_{k}"], N[f"fam_max_{k}_name"] = f"{fam[fmax]:.2f}", fam_names[fmax]
if os.path.exists(os.path.join(R, "significance_s0.json")):
    sg = L("significance_s0.json")
    def pf(p):
        return "<10{sup:\u2212" + str(int(-np.ceil(np.log10(p)))) + "}" if p < 1e-3 else f"{p:.3f}"
    for st, d in sg.items():
        for k, v in d.items():
            N[f"sig_{st}_{k}_diff"] = f"{v['diff']:.2f}"
            N[f"sig_{st}_{k}_ci"] = f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]"
            N[f"sig_{st}_{k}_p"] = pf(v["p"])
    N["sig_id_ens_gbdt_pmax"] = pf(max(sg["id"][f"hcg_ens_vs_gbdt_{t}"]["p"] for t in T))
    N["sig_ood_ens_gbdt_pmax"] = pf(max(sg["ood"][f"hcg_ens_vs_gbdt_{t}"]["p"] for t in T))
    N["sig_id_net_gbdt_pmin"] = pf(min(sg["id"][f"hcgnet_vs_gbdt_{t}"]["p"] for t in T))
json.dump(N, open(os.path.join(R, "nums.json"), "w"), indent=1, ensure_ascii=False)
print({k: v for k, v in N.items() if not k.startswith("table")})
