"""Fig. 1: worked example of the method — (a) logical circuit with consolidated 2q blocks,
(b) the resulting heterogeneous circuit graph, (c) HCG-Net and the structural-prior decoder."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"],
                     "mathtext.fontset": "stix", "font.size": 6.5})
BLUE, GREEN, ORANGE, PURPLE, GREY = "#2b5c8a", "#3d7a3d", "#c26a1f", "#6b3d7a", "#555555"
BCOL = ["#dbe8f5", "#e3f0dc", "#f8e6d4", "#efe0f3", "#f3f0d2"]

fig = plt.figure(figsize=(3.38, 2.88), dpi=600)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 85); ax.axis("off")


def label(x, y, t):
    ax.text(x, y, t, fontsize=7, fontweight="bold", va="top", ha="left")


# ---------------- (a) logical circuit
label(0.5, 84.5, "(a)")
ys = {0: 77, 1: 71, 2: 65, 3: 59}
for q, y in ys.items():
    ax.plot([7, 97], [y, y], color="k", lw=0.5, zorder=1)
    ax.text(5.8, y, f"$q_{q}$", ha="right", va="center", fontsize=6.5)
# blocks: (x0, x1, qa, qb, name, colour index)
blocks = [(10, 31, 0, 1, "$b_1$", 0), (34, 43, 1, 2, "$b_2$", 1), (46, 65, 2, 3, "$b_3$", 2),
          (68, 77, 0, 3, "$b_4$", 3), (80, 89, 1, 2, "$b_5$", 4)]
for x0, x1, qa, qb, nm, c in blocks:
    ax.add_patch(FancyBboxPatch((x0, ys[qb] - 2.6), x1 - x0, ys[qa] - ys[qb] + 5.2,
                                boxstyle="round,pad=0,rounding_size=1.2", fc=BCOL[c], ec="#888888", lw=0.4, zorder=0))
    ax.text((x0 + x1) / 2, ys[qa] + 4.0, nm, ha="center", va="bottom", fontsize=6.5)


def cx(x, c, t):
    ax.plot([x, x], [ys[c], ys[t]], color="k", lw=0.6, zorder=2)
    ax.add_patch(Circle((x, ys[c]), 0.9, color="k", zorder=3))
    ax.add_patch(Circle((x, ys[t]), 1.6, fc="white", ec="k", lw=0.6, zorder=3))
    ax.plot([x - 1.6, x + 1.6], [ys[t], ys[t]], color="k", lw=0.6, zorder=4)
    ax.plot([x, x], [ys[t] - 1.6, ys[t] + 1.6], color="k", lw=0.6, zorder=4)


def u(x, q):
    ax.add_patch(Rectangle((x - 2.2, ys[q] - 2.0), 4.4, 4.0, fc="white", ec="k", lw=0.5, zorder=3))
    ax.text(x, ys[q], "u", ha="center", va="center", fontsize=5.5, zorder=4)


u(14, 0); cx(19, 0, 1); u(24, 1); cx(28.5, 0, 1)
cx(38.5, 1, 2)
cx(51, 2, 3); u(55.5, 3); cx(60.5, 2, 3)
# non-adjacent CX q0 -> q3 (wire crossings drawn as a long control line)
cx(72.5, 0, 3)
cx(84.5, 1, 2)

# ---------------- (b) heterogeneous graph
label(0.5, 54.5, "(b)")
P = {"b1": (12, 44), "b2": (25, 44), "b3": (38, 44), "b4": (25, 33.5), "b5": (44.5, 33.5)}
Q = {0: (8, 13), 1: (20, 13), 2: (32, 13), 3: (44, 13)}
seq = [("b1", "b2"), ("b1", "b4"), ("b2", "b3"), ("b3", "b4"), ("b2", "b5"), ("b3", "b5")]
inc = [("b1", 0), ("b1", 1), ("b2", 1), ("b2", 2), ("b3", 2), ("b3", 3), ("b4", 0), ("b4", 3), ("b5", 1), ("b5", 2)]
for b, q in inc:
    ax.plot([P[b][0], Q[q][0]], [P[b][1], Q[q][1] + 2.2], color="#b0b0b0", lw=0.45, zorder=1, ls=(0, (2, 1.2)))
for a, b in seq:
    ax.add_patch(FancyArrowPatch(P[a], P[b], arrowstyle="-|>", mutation_scale=5, lw=0.7, color=GREEN,
                                 shrinkA=6.2, shrinkB=6.2, zorder=2))
for i, (k, (x, y)) in enumerate(P.items()):
    ax.add_patch(Circle((x, y), 3.3, fc=BCOL[i], ec=GREEN, lw=0.7, zorder=3))
    ax.text(x, y, "$b_%s$" % k[1], ha="center", va="center", fontsize=6.3, zorder=4)
for q, (x, y) in Q.items():
    ax.add_patch(FancyBboxPatch((x - 2.6, y - 2.2), 5.2, 4.4, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="#fbeee2", ec=ORANGE, lw=0.7, zorder=3))
    ax.text(x, y, f"$q_{q}$", ha="center", va="center", fontsize=6.3, zorder=4)
ax.plot([2, 6], [4.5, 4.5], color=GREEN, lw=0.7); ax.text(7, 4.5, "sequence edge", va="center", fontsize=5.5)
ax.plot([27, 31], [4.5, 4.5], color="#b0b0b0", lw=0.6, ls=(0, (2, 1.2))); ax.text(32, 4.5, "incidence", va="center", fontsize=5.5)
ax.text(8, 52.6, "block nodes (10 features), qubit nodes (6)", fontsize=5.4, color=GREY, va="center")

# ---------------- (c) model
label(52, 54.5, "(c)")


def box(x, y, w, h, t, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=5.8, linespacing=1.2)


def arr(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, lw=0.6, color=GREY))


box(56, 41, 42, 9, "$L$ = 3 rounds: block$\\leftrightarrow$block\n(pred./succ.), block$\\leftrightarrow$qubit", BLUE, "#e8eff7")
box(56, 28, 19, 9, "global $\\mathbf{g}$ (23)\nincl. opt. level", ORANGE, "#fbeee2")
box(79, 28, 19, 9, "mean + max\nreadout, MLP", BLUE, "#e8eff7")
box(56, 13, 42, 10, "$\\hat{\\mathbf{y}} = \\mathbf{b}(C) + f_\\theta(C,o)$\nprior $\\mathbf{b}$: logical depth, CX, width", PURPLE, "#f3e9f6")
box(56, 2, 42, 7.5, "depth,  $N_{2q}$,  ESP", "k", "white")
arr(49, 30, 55.5, 45); arr(77, 40.8, 88.5, 37.2); arr(75.2, 32.5, 78.8, 32.5); arr(88.5, 27.8, 80, 23.2)
arr(77, 12.8, 77, 9.7)
plt.savefig("results/fig_arch.png", dpi=600)
print("ok")
