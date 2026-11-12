"""Export the labelled datasets as CSV (labels) + QPY (logical circuits, same order).
Floats are written with repr() (full double precision) so the import round trip is bit-exact."""
import glob, pickle, csv, json, math, os, io
from qiskit import qpy
for dev in ("sherbrooke", "torino"):
    recs = []
    for f in sorted(glob.glob(f"data/{dev}/shard_*.pkl")):
        recs += pickle.load(open(f, "rb"))
    recs.sort(key=lambda r: r["uid"])
    os.makedirs("export", exist_ok=True)
    circs = [qpy.load(io.BytesIO(r["qpy"]))[0] for r in recs]
    with open(f"export/{dev}_circuits.qpy", "wb") as fh:
        qpy.dump(circs, fh)
    with open(f"export/{dev}_labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "uid", "family", "n_qubits", "split_pool", "opt_level", "depth", "n_2q", "ln_esp", "esp",
                    "transpile_s", "seed", "extra_seed_labels", "generator_meta"])
        for i, r in enumerate(recs):
            l = r["labels"][0]
            extra = [[x["depth"], x["twoq"], repr(x["log_esp"])] for x in r["labels"][1:]]
            w.writerow([i, r["uid"], r["family"], r["n"], "ood" if r["uid"] >= 10**6 else "id", r["opt_level"], l["depth"],
                        l["twoq"], repr(l["log_esp"]), f"{math.exp(l['log_esp']):.6e}", repr(l["t_transpile"]),
                        l["seed"], json.dumps(extra) if extra else "", json.dumps(r["meta"])])
    print(dev, len(recs))
