"""Experiments for the transpilation-cost surrogate paper.

python train.py --exp iid        # main comparison (5 seeds): ID test + size-OOD test
python train.py --exp ablation   # GNN component ablations (3 seeds)
python train.py --exp lofo       # leave-one-family-out (seed 0)
python train.py --exp torino     # second device + transfer
python train.py --exp timing     # inference cost vs transpilation
"""
import argparse, json, math, os, pickle, time, copy
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from features import glob_vector
from model import HCGNet, Batch

torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '1')))
OUT = "results"
os.makedirs(OUT, exist_ok=True)
TASKS = ["depth", "twoq", "esp"]


# ----------------------------------------------------------------------------- data
def load(path):
    recs = pickle.load(open(path, "rb"))
    for r in recs:
        g = r["graph"]
        r["gvec"] = glob_vector(g["glob"])
        agg = []
        for M in (g["gx"], g["qx"]):
            agg += list(M.mean(0)) if len(M) else [0.0] * M.shape[1]
            agg += list(M.max(0)) if len(M) else [0.0] * M.shape[1]
        r["tab"] = np.concatenate([r["gvec"], np.array(agg, np.float32)])
        r["ood"] = r["uid"] >= 10**6
        L = g["logical"]
        # first-order, device-agnostic structural prior (log of the logical counterpart of each cost)
        r["base"] = np.array([math.log(L["depth"] + 1), math.log1p(L["cx"]), math.log(L["cx"] + 2.5 * g["n"])], np.float32)
        r["y_abs"] = r["y"]
    return recs


RESID = True  # learn log-overhead w.r.t. the structural prior (Sec. III-C)


def Y(recs):
    return np.stack([r["y"] - r["base"] if RESID else r["y"] for r in recs])


def B(recs):
    return np.stack([r["base"] if RESID else np.zeros(3, np.float32) for r in recs])


def split_id(recs, seed, frac=(0.7, 0.15, 0.15)):
    ids = [r for r in recs if not r["ood"]]
    rng = np.random.RandomState(seed)
    p = rng.permutation(len(ids))
    a, b = int(frac[0] * len(ids)), int((frac[0] + frac[1]) * len(ids))
    return [ids[i] for i in p[:a]], [ids[i] for i in p[a:b]], [ids[i] for i in p[b:]]


# ----------------------------------------------------------------------------- metrics
def to_raw(ylog):
    return np.stack([np.exp(ylog[:, 0]), np.expm1(ylog[:, 1]), np.exp(ylog[:, 2])], 1)  # depth, 2q, -lnESP


def metrics(recs, pred):
    """pred: absolute log-targets"""
    yt = np.stack([r["y"] for r in recs])
    rt, rp = to_raw(yt), to_raw(pred)
    out = {}
    for k, t in enumerate(TASKS):
        den = np.maximum(rt[:, k], 1.0) if t == "twoq" else rt[:, k]
        ape = np.abs(rp[:, k] - rt[:, k]) / den
        ss = ((yt[:, k] - pred[:, k]) ** 2).sum() / ((yt[:, k] - yt[:, k].mean()) ** 2).sum()
        out[t] = dict(mape=100 * float(ape.mean()), medape=100 * float(np.median(ape)),
                      within10=100 * float((ape <= 0.10).mean()), r2=float(1 - ss),
                      rho=float(spearmanr(rt[:, k], rp[:, k]).correlation))
    esp_t, esp_p = np.exp(-rt[:, 2]), np.exp(-rp[:, 2])
    out["esp"]["esp_mae"] = float(np.abs(esp_t - esp_p).mean())
    k = max(1, len(recs) // 10)  # top-decile recall for ESP-based screening
    top_t = set(np.argsort(-esp_t)[:k]); top_p = set(np.argsort(-esp_p)[:k])
    out["esp"]["top10_recall"] = 100 * len(top_t & top_p) / k
    return out


# ----------------------------------------------------------------------------- tabular baselines
def scaling_features(recs):
    """Calibrated analytical scaling law: log-cost vs log logical counterpart (+ width, opt level)."""
    rows = []
    for r in recs:
        g = r["graph"]["glob"]
        rows.append([g["log_depth"], g["log_cx"], g["log_u"], math.log(g["n_qubits"]), g["ol2"], g["ol3"]])
    return np.array(rows, np.float32)


def fit_tab(name, train, seed):
    X = np.stack([r["tab"] for r in train]) if name != "scaling" else scaling_features(train)
    y = Y(train)
    models = []
    for k in range(3):
        if name in ("scaling", "ridge"):
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        elif name == "rf":
            m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=2, random_state=seed)
        elif name == "gbdt":
            m = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1e-3, early_stopping=True,
                                              validation_fraction=0.15, n_iter_no_change=40, random_state=seed)
        elif name == "mlp":
            m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 256), alpha=1e-4, learning_rate_init=1e-3,
                                                             max_iter=600, early_stopping=True,
                                                             random_state=seed))
        m.fit(X, y[:, k]); models.append(m)
    return (name, models)


def predict_tab(fm, recs):
    name, models = fm
    X = np.stack([r["tab"] for r in recs]) if name != "scaling" else scaling_features(recs)
    return np.stack([m.predict(X) for m in models], 1) + B(recs)


# ----------------------------------------------------------------------------- GNN
class GNNWrap:
    def __init__(self, model, gs, ym, ys):
        self.model, self.gs, self.ym, self.ys = model, gs, ym, ys

    def predict(self, recs, bs=256):
        self.model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(recs), bs):
                b = recs[i:i + bs]
                bt = Batch([r["graph"] for r in b], self.gs.transform(np.stack([r["gvec"] for r in b])).astype(np.float32))
                out.append(self.model(bt).numpy())
        return np.concatenate(out) * self.ys + self.ym + B(recs)


def fit_gnn(train, val, seed, epochs=60, bs=64, lr=3e-3, h=64, layers=3, task_w=(1, 1, 1),
            init=None, verbose=False, **kw):
    torch.manual_seed(seed); np.random.seed(seed)
    gs = StandardScaler().fit(np.stack([r["gvec"] for r in train])) if init is None else init.gs
    y = Y(train)
    ym, ys = (y.mean(0), y.std(0)) if init is None else (init.ym, init.ys)
    model = HCGNet(len(train[0]["gvec"]), h=h, layers=layers, **kw) if init is None else copy.deepcopy(init.model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(train) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    w = torch.tensor(task_w, dtype=torch.float32)
    # pre-collate validation
    wrap = GNNWrap(model, gs, ym, ys)
    yv = Y(val)
    best, best_state = 1e9, None
    G = [r["graph"] for r in train]
    Gv = gs.transform(np.stack([r["gvec"] for r in train])).astype(np.float32)
    Yt = torch.from_numpy((y - ym) / ys).float()
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(train))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            bt = Batch([G[j] for j in idx], Gv[idx])
            pred = model(bt)
            loss = (torch.nn.functional.smooth_l1_loss(pred, Yt[idx], reduction="none", beta=1.0) * w).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        if ep % 2 == 1 or ep == epochs - 1:
            pv = wrap.predict(val) - B(val)
            vl = float((((pv - yv) / ys) ** 2 * np.array(task_w)).mean())
            if vl < best:
                best, best_state = vl, copy.deepcopy(model.state_dict())
            if verbose:
                print(f"ep {ep} loss {loss.item():.4f} val {vl:.4f}", flush=True)
    model.load_state_dict(best_state)
    return wrap


# ----------------------------------------------------------------------------- experiments
def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, name + ".json"), "w"), indent=1)


def exp_iid(recs, seeds, tag="sherbrooke", baselines=("scaling", "ridge", "rf", "gbdt", "mlp")):
    ood = [r for r in recs if r["ood"]]
    res = {}
    for s in seeds:
        tr, va, te = split_id(recs, s)
        t0 = time.time()
        gnn = fit_gnn(tr, va, s)
        g_te, g_ood = gnn.predict(te), gnn.predict(ood)
        res.setdefault("hcgnet", []).append(dict(id=metrics(te, g_te), ood=metrics(ood, g_ood), train_s=time.time() - t0))
        if s == seeds[0]:
            pickle.dump(dict(te_uid=[r["uid"] for r in te], te_pred=gnn.predict(te),
                             ood_uid=[r["uid"] for r in ood], ood_pred=gnn.predict(ood)),
                        open(os.path.join(OUT, f"preds_{tag}_s{s}.pkl"), "wb"))
            torch.save(dict(state=gnn.model.state_dict(), gs=gnn.gs, ym=gnn.ym, ys=gnn.ys),
                       os.path.join(OUT, f"hcgnet_{tag}_s{s}.pt"))
        for b in baselines:
            t0 = time.time()
            fm = fit_tab(b, tr + va, s)
            b_te, b_ood = predict_tab(fm, te), predict_tab(fm, ood)
            res.setdefault(b, []).append(dict(id=metrics(te, b_te), ood=metrics(ood, b_ood), train_s=time.time() - t0))
            if b == "gbdt":  # HCG-Ens: equal-weight log-space average of HCG-Net and descriptor GBDT
                res.setdefault("hcg_ens", []).append(dict(id=metrics(te, (g_te + b_te) / 2), ood=metrics(ood, (g_ood + b_ood) / 2)))
        print(f"[{tag}] seed {s}: " + " | ".join(
            f"{m} d={res[m][-1]['id']['depth']['mape']:.1f} q={res[m][-1]['id']['twoq']['mape']:.1f} "
            f"e={res[m][-1]['id']['esp']['mape']:.1f}" for m in res), flush=True)
        save(f"iid_{tag}", res)
    return res


def exp_ablation(recs, seeds):
    # "full" is identical (same split, seed, config) to the iid run and is taken from iid_sherbrooke.json
    variants = {"no_seq": dict(use_seq=False), "no_qubit": dict(use_qubit=False),
                "no_glob": dict(use_glob=False), "sum_readout": dict(sum_readout=True), "no_prior": "noprior",
                "single_task": None}
    res = {}
    for s in seeds:
        tr, va, te = split_id(recs, s)
        for v, kw in variants.items():
            if v == "single_task" and s != seeds[0]:
                continue  # single-task variant (3 separate models) is run for the first seed only
            if v == "no_prior":
                global RESID
                RESID = False
                gnn = fit_gnn(tr, va, s); pred = gnn.predict(te); pred_ood = gnn.predict([r for r in recs if r["ood"]])
                gb = fit_tab("gbdt", tr + va, s); oo = [r for r in recs if r["ood"]]
                res.setdefault("gbdt_no_prior", []).append(dict(id=metrics(te, predict_tab(gb, te)), ood=metrics(oo, predict_tab(gb, oo))))
                RESID = True
                res.setdefault(v, []).append(dict(id=metrics(te, pred), ood=metrics(oo, pred_ood)))
                print(f"[abl] seed {s} no_prior done", flush=True); save("ablation", res)
                continue
            if v == "single_task":
                pred = np.zeros((len(te), 3))
                for k in range(3):
                    w = [0, 0, 0]; w[k] = 1
                    pred[:, k] = fit_gnn(tr, va, s, task_w=w).predict(te)[:, k]
            else:
                gnn = fit_gnn(tr, va, s, **kw); pred = gnn.predict(te)
            oo = [r for r in recs if r["ood"]]
            res.setdefault(v, []).append(dict(id=metrics(te, pred), ood=metrics(oo, gnn.predict(oo)) if v != "single_task" else None))
            m = res[v][-1]["id"]
            print(f"[abl] seed {s} {v}: d={m['depth']['mape']:.2f} q={m['twoq']['mape']:.2f} e={m['esp']['mape']:.2f}", flush=True)
            save("ablation", res)


def exp_lofo(recs, seed=0):
    fams = sorted(set(r["family"] for r in recs))
    ids = [r for r in recs if not r["ood"]]
    res = {}
    for f in fams:
        pool = [r for r in ids if r["family"] != f]
        test = [r for r in recs if r["family"] == f]          # all sizes of the unseen family
        rng = np.random.RandomState(seed); p = rng.permutation(len(pool)); cut = int(0.85 * len(pool))
        tr, va = [pool[i] for i in p[:cut]], [pool[i] for i in p[cut:]]
        pg = fit_gnn(tr, va, seed).predict(test); pb = predict_tab(fit_tab("gbdt", tr + va, seed), test)
        res[f] = dict(hcgnet=metrics(test, pg), gbdt=metrics(test, pb), hcg_ens=metrics(test, (pg + pb) / 2),
                      n_test=len(test))
        print(f"[lofo] {f}: gnn d={res[f]['hcgnet']['depth']['mape']:.1f} q={res[f]['hcgnet']['twoq']['mape']:.1f} "
              f"e={res[f]['hcgnet']['esp']['mape']:.1f} | gbdt d={res[f]['gbdt']['depth']['mape']:.1f} "
              f"q={res[f]['gbdt']['twoq']['mape']:.1f} e={res[f]['gbdt']['esp']['mape']:.1f}", flush=True)
        save("lofo", res)


def exp_torino(recs_t, recs_s, seeds):
    exp_iid(recs_t, seeds, tag="torino", baselines=("ridge", "gbdt"))
    # transfer: Sherbrooke-pretrained HCG-Net fine-tuned with k Torino circuits vs. training from scratch
    res = {}
    tr_s, va_s, _ = split_id(recs_s, 0)
    src = fit_gnn(tr_s, va_s, 0)
    for s in seeds:
        tr, va, te = split_id(recs_t, s)
        res.setdefault("zero_shot", []).append(metrics(te, src.predict(te)))
        for k in (100, 300, len(tr)):
            sub = tr[:k]
            res.setdefault(f"scratch_{k if k < len(tr) else 'all'}", []).append(metrics(te, fit_gnn(sub, va, s).predict(te)))
            res.setdefault(f"finetune_{k if k < len(tr) else 'all'}", []).append(
                metrics(te, fit_gnn(sub, va, s, init=src, lr=1e-3, epochs=40).predict(te)))
        print(f"[transfer] seed {s}: " + " | ".join(f"{m} d={v[-1]['depth']['mape']:.1f} q={v[-1]['twoq']['mape']:.1f} "
                                                     f"e={v[-1]['esp']['mape']:.1f}" for m, v in res.items()), flush=True)
        save("transfer_torino", res)


def exp_timing(recs):
    tr, va, te = split_id(recs, 0)
    ck = torch.load(os.path.join(OUT, "hcgnet_sherbrooke_s0.pt"), weights_only=False)
    model = HCGNet(len(tr[0]["gvec"])); model.load_state_dict(ck["state"]); model.eval()
    wrap = GNNWrap(model, ck["gs"], ck["ym"], ck["ys"])
    torch.set_num_threads(1)
    sample = te[:400]
    t_single = []
    for r in sample:  # forward pass at batch size 1 (graph construction time is t_feat, measured in prepare.py)
        t0 = time.perf_counter()
        wrap.predict([r], bs=1)
        t_single.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); wrap.predict(te, bs=256); t_batch = (time.perf_counter() - t0) / len(te)
    tt = np.array([r["t_transpile"] for r in te]); tf = np.array([r["t_feat"] for r in sample])
    by_ol = {ol: float(np.mean([r["t_transpile"] for r in te if r["opt_level"] == ol])) for ol in (1, 2, 3)}
    out = dict(transpile_mean=float(tt.mean()), transpile_median=float(np.median(tt)), transpile_by_ol=by_ol,
               feat_mean=float(tf.mean()), gnn_single_mean=float(np.mean(t_single)), gnn_batch_per_circ=t_batch,
               n=len(te))
    out["speedup_single"] = out["transpile_mean"] / (out["feat_mean"] + out["gnn_single_mean"])
    out["speedup_batch"] = out["transpile_mean"] / (out["feat_mean"] + out["gnn_batch_per_circ"])
    print(out); save("timing", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    S = load("data/sherbrooke/processed.pkl")
    if a.exp == "iid":
        exp_iid(S, list(range(a.seeds)))
    elif a.exp == "ablation":
        exp_ablation(S, list(range(a.seeds)))
    elif a.exp == "lofo":
        exp_lofo(S)
    elif a.exp == "torino":
        exp_torino(load("data/torino/processed.pkl"), S, list(range(a.seeds)))
    elif a.exp == "timing":
        exp_timing(S)
