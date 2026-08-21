"""Redraw the all-families accuracy figure in the dissertation's standard
figure style, from the stored results of make_all_families_1d.py.

Reads output/families/all_families_1d.npz and re-renders only -- no network
is retrained here, so the plotted numbers are byte-identical to those already
reported in the text.

Differences from the original rendering: dissertation rcParams (serif, cm
mathtext, matched font sizes and tick style), no in-figure title (the caption
carries it), legend entries reduced to the orbit number alone, and 300 dpi.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
OUT = out("families")
FIGDIR = figure_dir()

OMEGA, E_0 = 0.057, 0.0534
U_P = E_0 ** 2 / (4 * OMEGA ** 2)

mpl.rcParams.update({
    "font.size": 7.2, "axes.labelsize": 7.6, "axes.titlesize": 8.0,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "mathtext.fontset": "cm", "font.family": "serif",
})

# eight distinguishable colours, ordered so early and late returns separate
COLORS = ["#2d5591", "#b23a48", "#3f8f5a", "#c9781f",
          "#6e4d9a", "#8a6d3b", "#c0489a", "#3f3f46"]


def main():
    d = np.load(f"{OUT}/all_families_1d.npz")
    fams = sorted(int(k.split("_")[1]) for k in d.files if k.startswith("err_"))

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, f in enumerate(fams):
        E = d[f"p_{f}"] ** 2 / (2 * U_P)
        ax.semilogy(E, d[f"err_{f}"], lw=0.85, color=COLORS[i % len(COLORS)],
                    label=f"{f}")
    ax.set_xlabel(r"final energy  ($U_p$)")
    ax.set_ylabel(r"$|\Delta\theta_s|$")
    ax.tick_params(direction="in", top=True, right=True, length=2.4)

    # legend boxed inside the axes; the curves leave the upper-left corner
    # free, and two rows of four keep it compact
    leg = ax.legend(loc="upper left", ncol=4, frameon=True, fontsize=6.0,
                    handlelength=0.9, columnspacing=0.7, handletextpad=0.4,
                    labelspacing=0.25, borderpad=0.35,
                    borderaxespad=0.4, framealpha=0.92)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor("0.4")

    fig.tight_layout()
    fig.savefig(f"{OUT}/all_families_error_1d.png", dpi=300, bbox_inches="tight")
    if FIGDIR:
        fig.savefig(f"{FIGDIR}/all_families_error_1d.png", dpi=300, bbox_inches="tight")
    print("wrote all_families_error_1d.png (replot only, no retraining)")
    print("per-family median |d theta_s| (unchanged):")
    for f in fams:
        print(f"  orbit {f}: {np.median(d[f'err_{f}']):.3e}")


if __name__ == "__main__":
    main()
