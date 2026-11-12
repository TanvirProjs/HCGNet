"""Rebuild the shard files used by prepare.py from the released CSV + QPY dataset.

python import_data.py <dataset_dir> <device> <out_dir>
e.g. python import_data.py hcgnet_dataset sherbrooke data/sherbrooke
then: python prepare.py data/sherbrooke sherbrooke
"""
import sys, os, csv, json, io, pickle
from qiskit import qpy

src, dev, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
with open(os.path.join(src, f"{dev}_circuits.qpy"), "rb") as fh:
    circs = qpy.load(fh)
rows = list(csv.DictReader(open(os.path.join(src, f"{dev}_labels.csv"))))
assert len(rows) == len(circs)
recs = []
for row, qc in zip(rows, circs):
    buf = io.BytesIO(); qpy.dump(qc, buf)
    labels = [dict(depth=int(row["depth"]), twoq=int(row["n_2q"]), log_esp=float(row["ln_esp"]),
                   t_transpile=float(row["transpile_s"]), seed=int(row["seed"]))]
    for d, q, e in (json.loads(row["extra_seed_labels"]) if row["extra_seed_labels"] else []):
        labels.append(dict(depth=d, twoq=q, log_esp=float(e), t_transpile=float("nan"), seed=-1))
    recs.append(dict(uid=int(row["uid"]), family=row["family"], n=int(row["n_qubits"]), gen_seed=None,
                     opt_level=int(row["opt_level"]), meta=json.loads(row["generator_meta"]), qpy=buf.getvalue(),
                     labels=labels))
for i in range(0, len(recs), 500):
    pickle.dump(recs[i:i + 500], open(os.path.join(out, f"shard_{i // 500:03d}.pkl"), "wb"))
print(len(recs), "records written to", out)
