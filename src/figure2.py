"""Generates Figure 2: delta-band connectivity differences between conditions.

Panel A  Heatmap of pairwise delta-band coherence differences across all 171
         electrode pairs, with significant pairs marked.
Panel B  Topographic network graph on a standard head schematic, with an inset
         reporting leave-one-out cross-validation performance.

Appearance follows the published figure caption: warm tones mark higher
coherence in the treatment-simulated condition, cool tones lower; significant
pairs carry a white border; red and blue edges mark increases and decreases,
with width scaled to effect magnitude.
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from montage import CHANNELS, N_CHANNELS, pair_list
from simulate import simulate_cohort
import analysis

# Approximate 10-20 scalp positions on a unit head, (x right, y anterior).
POSITIONS = {
    "Fp1": (-0.27, 0.83), "Fp2": (0.27, 0.83),
    "F7": (-0.70, 0.48), "F3": (-0.37, 0.48), "Fz": (0.00, 0.48),
    "F4": (0.37, 0.48), "F8": (0.70, 0.48),
    "T3": (-0.87, 0.00), "C3": (-0.44, 0.00), "Cz": (0.00, 0.00),
    "C4": (0.44, 0.00), "T4": (0.87, 0.00),
    "T5": (-0.70, -0.48), "P3": (-0.37, -0.48), "Pz": (0.00, -0.48),
    "P4": (0.37, -0.48), "T6": (0.70, -0.48),
    "O1": (-0.27, -0.83), "O2": (0.27, -0.83),
}


def square_matrix(values):
    """Fold a 171-vector of pair values into a symmetric 19 x 19 matrix."""
    M = np.zeros((N_CHANNELS, N_CHANNELS))
    for k, (i, j) in enumerate(pair_list()):
        M[i, j] = M[j, i] = values[k]
    return M


def panel_a(ax, diff, reject):
    M = square_matrix(diff)
    S = square_matrix(reject.astype(float))
    lim = 0.15   # fixed scale, matching the published figure
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)

    for k, (i, j) in enumerate(pair_list()):
        if reject[k]:
            for (r, c) in ((i, j), (j, i)):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       fill=False, edgecolor="white", linewidth=0.8))

    ax.set_xticks(range(N_CHANNELS))
    ax.set_yticks(range(N_CHANNELS))
    ax.set_xticklabels(CHANNELS, rotation=90, fontsize=6)
    ax.set_yticklabels(CHANNELS, fontsize=6)
    ax.text(-0.13, 1.04, "A", transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\Delta$ coherence (treatment $-$ sham)", fontsize=7)
    cb.ax.tick_params(labelsize=6)


def panel_b(ax, diff, reject, d, clf):
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="0.4", linewidth=1.2))
    # Nose and ears.
    ax.plot([-0.09, 0, 0.09], [0.995, 1.09, 0.995], color="0.4", linewidth=1.2)
    for s in (-1, 1):
        ax.add_patch(Circle((s * 1.0, 0), 0.07, fill=False, edgecolor="0.4", linewidth=1.2))

    sig_d = np.abs(d[reject])
    wmax = sig_d.max() if sig_d.size else 1.0

    for k, (i, j) in enumerate(pair_list()):
        if not reject[k]:
            continue
        x1, y1 = POSITIONS[CHANNELS[i]]
        x2, y2 = POSITIONS[CHANNELS[j]]
        colour = "#c0392b" if diff[k] > 0 else "#2471a3"
        ax.plot([x1, x2], [y1, y2], color=colour,
                linewidth=0.4 + 2.6 * abs(d[k]) / wmax, alpha=0.75, zorder=1)

    for name in CHANNELS:
        x, y = POSITIONS[name]
        ax.add_patch(Circle((x, y), 0.062, facecolor="white",
                            edgecolor="0.3", linewidth=0.8, zorder=2))
        ax.text(x, y, name, ha="center", va="center", fontsize=5, zorder=3)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-0.02, 1.04, "B", transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom")

    ax.text(0.80, 0.96, "%d significant pairs\n(FDR q < 0.05)" % int(reject.sum()),
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5)

    txt = ("LOO-CV:  accuracy %.0f%%    sensitivity %.0f%%    specificity %.0f%%"
           % (100 * clf["accuracy"], 100 * clf["sensitivity"], 100 * clf["specificity"]))
    ax.text(0.5, 0.035, txt, transform=ax.transAxes, ha="center", va="center",
            fontsize=7, bbox=dict(boxstyle="square,pad=0.45", facecolor="white",
                                  edgecolor="0.35", linewidth=0.8))


def main():
    data, labels = simulate_cohort()
    X = analysis.build_features(data)
    stats = analysis.compare_conditions(X, labels)
    clf = analysis.loocv(X, labels)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    panel_a(axes[0], stats["diff"], stats["reject"])
    panel_b(axes[1], stats["diff"], stats["reject"], stats["d"], clf)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig("results/figure2.%s" % ext, dpi=300, bbox_inches="tight")
    print("wrote results/figure2.png and results/figure2.pdf")


if __name__ == "__main__":
    main()
