"""Dataset generator for the transpilation-cost surrogate study.

For every logical circuit we store
  * the logical circuit (canonicalised to {u, cx}, no coupling map) as QPY,
  * the post-routing labels produced by the Qiskit preset pass manager on a
    calibrated fake IBM backend: depth, two-qubit gate count, log ESP,
  * wall-clock transpilation time.

Usage: python gen_dataset.py --backend sherbrooke --out data/sherbrooke --n 7000 --workers 2
"""
import argparse, os, time, math, pickle, random, io
import numpy as np
from multiprocessing import Pool

from qiskit import QuantumCircuit, ClassicalRegister, transpile, qpy
from qiskit.circuit.library import QFT, QuantumVolume, EfficientSU2, CDKMRippleCarryAdder, DraperQFTAdder
from qiskit.circuit.random import random_circuit

FAMILIES = ["random", "qft", "ghz", "bv", "qaoa", "su2", "qv", "adder", "trotter"]


def get_backend(name):
    """Calibrated Target of a fake IBM device with calibration-flagged couplers (error >= 0.5)
    removed, restricted to the largest connected component of the remaining coupling graph
    (qubits re-indexed). This mirrors how non-operational couplers are excluded on production
    devices and avoids the disjoint-coupling code path."""
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeTorino
    from qiskit.transpiler import Target
    import rustworkx as rx
    b = {"sherbrooke": FakeSherbrooke, "torino": FakeTorino}[name]()
    src = b.target
    bad = lambda p: p is not None and p.error is not None and p.error >= 0.5
    g = rx.PyGraph(); g.add_nodes_from(range(src.num_qubits))
    for opname in src.operation_names:
        props = src[opname]
        if props is None:
            continue
        for qargs, p in props.items():
            if qargs is not None and len(qargs) == 2 and not bad(p):
                g.add_edge(qargs[0], qargs[1], None)
    comp = sorted(max(rx.connected_components(g), key=len))
    remap = {old: new for new, old in enumerate(comp)}
    qp = [src.qubit_properties[o] for o in comp] if src.qubit_properties else None
    tgt = Target(description=f"{name}-lcc", num_qubits=len(comp), dt=src.dt, granularity=src.granularity,
                 min_length=src.min_length, pulse_alignment=src.pulse_alignment,
                 acquire_alignment=src.acquire_alignment, qubit_properties=qp)
    for opname in src.operation_names:
        if opname in ("for_loop", "if_else", "switch_case", "while_loop", "box"):
            continue  # control flow is not used by the benchmark circuits
        props = src[opname]
        keep = {}
        for qargs, p in props.items():
            if qargs is None:
                keep[None] = p; continue
            if any(q not in remap for q in qargs) or (len(qargs) == 2 and bad(p)):
                continue
            keep[tuple(remap[q] for q in qargs)] = p
        tgt.add_instruction(src.operation_from_name(opname), keep, name=opname)
    return tgt


# ----------------------------------------------------------------------------- circuits
def random_graph(n, rng, kind=None):
    kind = kind or rng.choice(["er", "reg3", "ring", "grid"])
    edges = set()
    if kind == "er":
        p = rng.uniform(0.15, 0.6)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    edges.add((i, j))
    elif kind == "reg3":
        # random near-3-regular graph via configuration-style pairing
        stubs = [i for i in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        for a, b in zip(stubs[::2], stubs[1::2]):
            if a != b:
                edges.add((min(a, b), max(a, b)))
    elif kind == "ring":
        for i in range(n):
            edges.add((min(i, (i + 1) % n), max(i, (i + 1) % n)))
    else:  # 2D grid
        w = max(2, int(math.sqrt(n)))
        for i in range(n):
            if (i + 1) % w and i + 1 < n:
                edges.add((i, i + 1))
            if i + w < n:
                edges.add((i, i + w))
    if not edges:
        edges.add((0, 1))
    return sorted(edges), kind


def make_circuit(family, n, rng):
    meta = {}
    if family == "random":
        d = rng.randint(4, 40)
        qc = random_circuit(n, d, max_operands=2, seed=rng.randint(0, 2**31 - 1))
        meta["depth_param"] = d
    elif family == "qft":
        approx = rng.choice([0, 0, max(0, n // 3), max(0, n // 2)])
        qc = QFT(n, approximation_degree=approx, do_swaps=rng.random() < 0.5)
        meta["approx"] = approx
    elif family == "ghz":
        qc = QuantumCircuit(n)
        qc.h(0)
        topo = rng.choice(["chain", "star", "tree"])
        for i in range(1, n):
            src = {"chain": i - 1, "star": 0, "tree": (i - 1) // 2}[topo]
            qc.cx(src, i)
        meta["topo"] = topo
    elif family == "bv":
        qc = QuantumCircuit(n)
        secret = [rng.random() < 0.5 for _ in range(n - 1)]
        qc.x(n - 1)
        qc.h(range(n))
        for i, s in enumerate(secret):
            if s:
                qc.cx(i, n - 1)
        qc.h(range(n - 1))
        meta["weight"] = int(sum(secret))
    elif family in ("qaoa", "trotter"):
        edges, kind = random_graph(n, rng, None if family == "qaoa" else rng.choice(["ring", "grid"]))
        p = rng.randint(1, 3) if family == "qaoa" else rng.randint(1, 4)
        qc = QuantumCircuit(n)
        if family == "qaoa":
            qc.h(range(n))
        for _ in range(p):
            order = list(edges)
            rng.shuffle(order)
            for (a, b) in order:
                th = rng.uniform(0, math.pi)
                if family == "qaoa":
                    qc.rzz(th, a, b)
                else:  # Heisenberg XX+YY+ZZ Trotter step
                    qc.rxx(th, a, b); qc.ryy(th, a, b); qc.rzz(th, a, b)
            for q in range(n):
                qc.rx(rng.uniform(0, math.pi), q)
        meta.update(graph=kind, p=p, n_edges=len(edges))
    elif family == "su2":
        ent = rng.choice(["linear", "circular", "full", "sca"])
        reps = rng.randint(1, 3)
        qc = EfficientSU2(n, reps=reps, entanglement=ent)
        qc = qc.assign_parameters(np.array([rng.uniform(0, 2 * math.pi) for _ in range(qc.num_parameters)]))
        meta.update(ent=ent, reps=reps)
    elif family == "qv":
        d = rng.randint(2, n)
        qc = QuantumVolume(n, depth=d, seed=rng.randint(0, 2**31 - 1))
        meta["depth_param"] = d
    elif family == "adder":
        k = max(1, (n - 2) // 2)
        if rng.random() < 0.5:
            qc = CDKMRippleCarryAdder(k, kind="full")
        else:
            qc = DraperQFTAdder(k, kind="half")
        meta["k"] = k
    else:
        raise ValueError(family)
    return qc, meta


def canonical_logical(qc):
    """Logical canonical form: basis {u, cx}, light 1q merging, no device information."""
    qc = qc.decompose(reps=4) if qc.num_parameters == 0 else qc
    return transpile(qc, basis_gates=["u", "cx"], optimization_level=1, seed_transpiler=0)


# ----------------------------------------------------------------------------- labels
def esp_of(tqc, target):
    """Estimated success probability (product of gate and readout fidelities)."""
    log_esp = 0.0
    for inst in tqc.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = tuple(tqc.find_bit(q).index for q in inst.qubits)
        try:
            props = target[name][qargs]
        except KeyError:
            props = None
        err = props.error if (props is not None and props.error is not None) else 0.0
        err = min(max(err, 0.0), 0.999)
        log_esp += math.log1p(-err)
    return log_esp


def two_q_count(tqc):
    return sum(1 for inst in tqc.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")


_BACKEND = None


def _init(backend_name):
    global _BACKEND
    _BACKEND = get_backend(backend_name)


def label_job(job):
    """job = (uid, family, n, gen_seed, opt_level, transpile_seeds)"""
    uid, family, n, gseed, ol, tseeds = job
    rng = random.Random(gseed)
    qc, meta = make_circuit(family, n, rng)
    log = canonical_logical(qc)
    meas = log.copy()
    meas.add_register(ClassicalRegister(meas.num_qubits, 'meas'))
    meas.measure(range(meas.num_qubits), range(meas.num_qubits))  # no barrier
    out = []
    for ts in tseeds:
        t0 = time.perf_counter()
        tqc = transpile(meas, target=_BACKEND, optimization_level=ol, seed_transpiler=ts)
        dt = time.perf_counter() - t0
        out.append(dict(depth=tqc.depth(), twoq=two_q_count(tqc), log_esp=esp_of(tqc, _BACKEND),
                        size=tqc.size(), t_transpile=dt, seed=ts))
    buf = io.BytesIO()
    qpy.dump(log, buf)
    return dict(uid=uid, family=family, n=n, gen_seed=gseed, opt_level=ol, meta=meta,
                qpy=buf.getvalue(), labels=out)


def build_jobs(n_total, nmin, nmax, seed, n_seeds_noise=0, noise_frac=0.0, uid0=0):
    rng = random.Random(seed)
    jobs = []
    for k in range(n_total):
        fam = FAMILIES[k % len(FAMILIES)]
        n = rng.randint(nmin, nmax)
        if fam == "adder":
            n = max(n, 4)
        ol = rng.choice([1, 2, 3])
        seeds = [rng.randint(0, 10**6)]
        if n_seeds_noise and rng.random() < noise_frac:
            seeds += [rng.randint(0, 10**6) for _ in range(n_seeds_noise - 1)]
        jobs.append((uid0 + k, fam, n, rng.randint(0, 2**31 - 1), ol, seeds))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sherbrooke")
    ap.add_argument("--out", default="data/sherbrooke")
    ap.add_argument("--n_id", type=int, default=6300)    # in-distribution sizes
    ap.add_argument("--n_ood", type=int, default=900)    # larger circuits for size extrapolation
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = build_jobs(a.n_id, 4, 20, a.seed, n_seeds_noise=5, noise_frac=0.05) + \
        build_jobs(a.n_ood, 21, 28, a.seed + 1, uid0=10**6)
    random.Random(0).shuffle(jobs)
    t0 = time.time()
    shard, res = 0, []
    with Pool(a.workers, initializer=_init, initargs=(a.backend,)) as pool:
        for i, r in enumerate(pool.imap_unordered(label_job, jobs, chunksize=4)):
            res.append(r)
            if len(res) == 500 or i == len(jobs) - 1:
                with open(os.path.join(a.out, f"shard_{shard:03d}.pkl"), "wb") as f:
                    pickle.dump(res, f)
                print(f"{i+1}/{len(jobs)} done  {time.time()-t0:.0f}s", flush=True)
                shard, res = shard + 1, []
