"""HCG-Net: heterogeneous circuit-graph network for transpilation-cost regression (pure PyTorch)."""
import numpy as np
import torch
import torch.nn as nn

from features import GATE_F, QUBIT_F


def scatter_mean(src, index, n):
    out = torch.zeros(n, src.shape[1], dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt = torch.zeros(n, dtype=src.dtype).index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1).unsqueeze(1)


def scatter_sum(src, index, n):
    out = torch.zeros(n, src.shape[1], dtype=src.dtype)
    return out.index_add_(0, index, src)


def scatter_max(src, index, n):
    out = torch.zeros(n, src.shape[1], dtype=src.dtype)
    if src.shape[0] == 0:
        return out
    idx = index.unsqueeze(1).expand_as(src)
    return out.scatter_reduce(0, idx, src, reduce="amax", include_self=False)


def mlp(i, h, o, act=nn.SiLU):
    return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, o))


class Batch:
    """Collated mini-batch of circuit graphs."""

    def __init__(self, graphs, glob):
        gx, qx, seq, inc, gb, qb = [], [], [], [], [], []
        go = qo = 0
        for b, g in enumerate(graphs):
            gx.append(g["gx"]); qx.append(g["qx"])
            seq.append(g["seq"] + go)
            inc.append(g["inc"] + np.array([[go], [qo]]))
            gb.append(np.full(g["G"], b)); qb.append(np.full(g["n"], b))
            go += g["G"]; qo += g["n"]
        self.gx = torch.from_numpy(np.concatenate(gx)).float()
        self.qx = torch.from_numpy(np.concatenate(qx)).float()
        self.seq = torch.from_numpy(np.concatenate(seq, 1)).long()
        self.inc = torch.from_numpy(np.concatenate(inc, 1)).long()
        self.gb = torch.from_numpy(np.concatenate(gb)).long()
        self.qb = torch.from_numpy(np.concatenate(qb)).long()
        self.glob = torch.from_numpy(glob).float()
        self.B, self.NG, self.NQ = len(graphs), go, qo


class Layer(nn.Module):
    def __init__(self, h, use_seq, use_qubit):
        super().__init__()
        self.use_seq, self.use_qubit = use_seq, use_qubit
        self.fg = mlp(4 * h, 2 * h, h)
        self.fq = mlp(2 * h, 2 * h, h)
        self.ng, self.nq = nn.LayerNorm(h), nn.LayerNorm(h)

    def forward(self, hg, hq, bt):
        z = torch.zeros_like(hg)
        if self.use_seq and bt.seq.shape[1]:
            s, d = bt.seq
            m_f = scatter_mean(hg[s], d, bt.NG)    # from predecessors
            m_b = scatter_mean(hg[d], s, bt.NG)    # from successors
        else:
            m_f = m_b = z
        ig, iq = bt.inc
        m_q = scatter_mean(hq[iq], ig, bt.NG) if self.use_qubit else z
        hg = self.ng(hg + self.fg(torch.cat([hg, m_f, m_b, m_q], 1)))
        if self.use_qubit:
            m_g = scatter_mean(hg[ig], iq, bt.NQ)
            hq = self.nq(hq + self.fq(torch.cat([hq, m_g], 1)))
        return hg, hq


class HCGNet(nn.Module):
    def __init__(self, glob_dim, h=64, layers=3, n_out=3, use_seq=True, use_qubit=True, use_glob=True, sum_readout=False):
        super().__init__()
        self.use_glob, self.sum_readout = use_glob, sum_readout
        self.eg, self.eq = mlp(GATE_F, h, h), mlp(QUBIT_F, h, h)
        self.eglob = mlp(glob_dim, h, h)
        self.layers = nn.ModuleList([Layer(h, use_seq, use_qubit) for _ in range(layers)])
        ro = (3 if sum_readout else 2) * 2 * h + (h if use_glob else 0)
        self.head = nn.Sequential(nn.Linear(ro, 2 * h), nn.SiLU(), nn.Dropout(0.05),
                                  nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, n_out))

    def forward(self, bt):
        cg = self.eglob(bt.glob)
        hg = self.eg(bt.gx)
        hq = self.eq(bt.qx)
        if self.use_glob:  # condition node states on circuit-level context (incl. optimisation level)
            hg = hg + cg[bt.gb]; hq = hq + cg[bt.qb]
        for L in self.layers:
            hg, hq = L(hg, hq, bt)
        B = bt.B
        # size-intensive readout (mean/max); extensive size information enters via the global descriptors
        r = [scatter_mean(hg, bt.gb, B), scatter_max(hg, bt.gb, B), scatter_mean(hq, bt.qb, B), scatter_max(hq, bt.qb, B)]
        if self.sum_readout:
            r += [scatter_sum(hg, bt.gb, B) / 100.0, scatter_sum(hq, bt.qb, B) / 10.0]
        if self.use_glob:
            r.append(cg)
        return self.head(torch.cat(r, 1))
