"""Build graphs/descriptors for every labelled circuit.

python prepare.py data/sherbrooke sherbrooke
"""
import sys, glob, pickle, time, math, os
from multiprocessing import Pool
import numpy as np
from features import load_circ, circuit_graph
import gen_dataset as gd

def work(r):
    qc = load_circ(r["qpy"])
    t0 = time.perf_counter()
    g = circuit_graph(qc, r["opt_level"])
    t_feat = time.perf_counter() - t0
    lab = r["labels"][0]
    return dict(uid=r["uid"], family=r["family"], n=r["n"], opt_level=r["opt_level"], meta=r["meta"],
                y=np.array([math.log(lab["depth"]), math.log1p(lab["twoq"]), math.log(-lab["log_esp"])], np.float32),
                raw=dict(depth=lab["depth"], twoq=lab["twoq"], log_esp=lab["log_esp"]),
                seeds=[(l["depth"], l["twoq"], l["log_esp"]) for l in r["labels"]],
                t_transpile=lab["t_transpile"], t_feat=t_feat, graph=g)


if __name__ == "__main__":
    d, name = sys.argv[1], sys.argv[2]
    recs = []
    for f in sorted(glob.glob(os.path.join(d, "shard_*.pkl"))):
        recs += pickle.load(open(f, "rb"))
    print(len(recs), "records")
    t0 = time.time()
    with Pool(2) as pool:
        out = pool.map(work, recs, chunksize=16)
    out.sort(key=lambda r: r["uid"])
    pickle.dump(out, open(os.path.join(d, "processed.pkl"), "wb"))
    print("done", time.time() - t0)
