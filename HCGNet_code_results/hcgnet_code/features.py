"""Circuit -> heterogeneous graph + hand-crafted descriptor vector.

Graph (per logical circuit in the {u, cx} basis):
  gate nodes   : one per consolidated two-qubit block (maximal run of CX on one pair)
  qubit nodes  : one per logical qubit
  seq edges    : block i -> block j when j is the next block on one of i's wires
  inc edges    : gate <-> its control and target qubit
Global vector  : circuit-level descriptors (also used by the tabular baselines).
"""
import io, math
import numpy as np
from qiskit import qpy

GATE_F = 10
QUBIT_F = 6
HEAVY_HEX_MAX_DEG = 3


def load_circ(qpy_bytes):
    return qpy.load(io.BytesIO(qpy_bytes))[0]


def circuit_graph(qc, opt_level):
    n = qc.num_qubits
    idx = {q: i for i, q in enumerate(qc.qubits)}
    last2q = [-1] * n          # index of the last 2q block on each wire
    n1q_since = [0] * n        # u gates since last block on wire
    n1q_tot = [0] * n
    layer_q = [0] * n          # ASAP layer (all gates) per wire
    layer2_q = [0] * n         # ASAP layer counting 2q blocks only
    gates, seq_src, seq_dst = [], [], []   # gate = [c, t, mult, n1q_in, pre_c, pre_t, l, l2, first_c, first_t]
    pair_cnt = {}
    n_cx = 0
    for inst in qc.data:
        op = inst.operation
        if op.name in ("barrier", "measure"):
            continue
        qs = [idx[q] for q in inst.qubits]
        if len(qs) == 1:
            q = qs[0]
            n1q_since[q] += 1; n1q_tot[q] += 1; layer_q[q] += 1
            continue
        c, t = qs
        n_cx += 1
        key = (min(c, t), max(c, t))
        pair_cnt[key] = pair_cnt.get(key, 0) + 1
        l = max(layer_q[c], layer_q[t]) + 1
        layer_q[c] = layer_q[t] = l
        b = last2q[c]
        if b >= 0 and b == last2q[t]:
            # same pair as the open block on both wires -> merge (2q-block consolidation)
            gates[b][2] += 1
            gates[b][3] += n1q_since[c] + n1q_since[t]
            n1q_since[c] = n1q_since[t] = 0
            continue
        g = len(gates)
        l2 = max(layer2_q[c], layer2_q[t]) + 1
        gates.append([c, t, 1, 0, n1q_since[c], n1q_since[t], l, l2, int(last2q[c] < 0), int(last2q[t] < 0)])
        for w in (c, t):
            if last2q[w] >= 0:
                seq_src.append(last2q[w]); seq_dst.append(g)
            last2q[w] = g
            n1q_since[w] = 0
        layer2_q[c] = layer2_q[t] = l2

    G = len(gates)
    depth = max(layer_q) if n else 0
    depth2 = max(layer2_q) if n else 0
    # interaction graph statistics
    deg = np.zeros(n); wdeg = np.zeros(n)
    adj = [set() for _ in range(n)]
    for (a, b), w in pair_cnt.items():
        deg[a] += 1; deg[b] += 1; wdeg[a] += w; wdeg[b] += w
        adj[a].add(b); adj[b].add(a)
    clust = np.zeros(n)
    for v in range(n):
        k = len(adj[v])
        if k >= 2:
            nb = list(adj[v]); links = sum(1 for i in range(k) for j in range(i + 1, k) if nb[j] in adj[nb[i]])
            clust[v] = 2 * links / (k * (k - 1))
    # BFS distances in interaction graph (avg shortest path over connected pairs)
    dsum, dcnt, diam = 0, 0, 0
    for s in range(n):
        dist = {s: 0}; frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1; nxt.append(v)
            frontier = nxt
        for v, d in dist.items():
            if v > s:
                dsum += d; dcnt += 1; diam = max(diam, d)
    n_comp = 0; seen = set()
    for s in range(n):
        if s in seen or not adj[s]:
            continue
        n_comp += 1; stack = [s]
        while stack:
            u = stack.pop()
            if u in seen: continue
            seen.add(u); stack.extend(adj[u] - seen)

    # node features
    if G:
        ga = np.array(gates, dtype=np.float64)
        c, t = ga[:, 0].astype(int), ga[:, 1].astype(int)
        gx = np.stack([
            np.log1p(ga[:, 2]), np.log1p(ga[:, 3]), np.log1p(ga[:, 4]), np.log1p(ga[:, 5]),
            ga[:, 6] / max(depth, 1), ga[:, 7] / max(depth2, 1),
            ga[:, 8], ga[:, 9],
            (deg[c] + deg[t]) / (2 * max(n - 1, 1)),
            np.abs(deg[c] - deg[t]) / max(n - 1, 1),
        ], 1).astype(np.float32)
        inc_g = np.concatenate([np.arange(G), np.arange(G)])
        inc_q = np.concatenate([c, t])
    else:
        gx = np.zeros((0, GATE_F), np.float32); inc_g = np.zeros(0, int); inc_q = np.zeros(0, int)
    qx = np.stack([
        deg / max(n - 1, 1), wdeg / max(n_cx, 1), np.log1p(np.array(n1q_tot, float)), clust,
        (deg > HEAVY_HEX_MAX_DEG).astype(float), (deg == 0).astype(float)
    ], 1).astype(np.float32)

    n_u = int(sum(n1q_tot)); E = len(pair_cnt)
    excess = float(np.maximum(deg - HEAVY_HEX_MAX_DEG, 0).sum())
    glob = dict(
        n_qubits=n, log_cx=math.log1p(n_cx), log_blocks=math.log1p(G), log_u=math.log1p(n_u), log_depth=math.log1p(depth),
        log_depth2=math.log1p(depth2), int_edges=E, int_density=E / max(n * (n - 1) / 2, 1),
        max_deg=float(deg.max()) if n else 0.0, mean_deg=float(deg.mean()) if n else 0.0,
        deg_excess=excess, log_deg_excess=math.log1p(excess), avg_dist=dsum / max(dcnt, 1), diam=diam,
        n_comp=n_comp, mean_clust=float(clust.mean()) if n else 0.0,
        cx_per_qubit=n_cx / max(n, 1), cx_per_edge=n_cx / max(E, 1), parallelism=G / max(depth2, 1),
        cx_per_block=n_cx / max(G, 1),
        ol1=float(opt_level == 1), ol2=float(opt_level == 2), ol3=float(opt_level == 3),
    )
    return dict(gx=gx, qx=qx, seq=np.array([seq_src, seq_dst], dtype=np.int64).reshape(2, -1),
                inc=np.stack([inc_g, inc_q]).astype(np.int64), glob=glob, G=G, n=n,
                logical=dict(depth=depth, cx=n_cx, u=n_u, blocks=G))


GLOB_KEYS = None


def glob_vector(glob):
    global GLOB_KEYS
    if GLOB_KEYS is None:
        GLOB_KEYS = list(glob.keys())
    return np.array([glob[k] for k in GLOB_KEYS], np.float32)
