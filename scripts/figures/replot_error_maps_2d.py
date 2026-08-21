"""Redraw the eight-family 2D error-map figure in the dissertation's standard
figure style, from the stored results of make_error_maps_2d.py.

Reads output/families/error_maps_2d.npz and re-renders only -- no network is
retrained here, so the plotted numbers are byte-identical to those already
reported in the text.

Differences from the original rendering: the same rcParams as
replot_all_families_1d.py (serif, cm mathtext, matched font sizes and tick
style), a canvas sized to the width the figure is actually printed at
(the full \\textwidth of a two-column-spanning figure*, 17.6 cm) so that point
sizes on the page match Fig. 10 exactly rather than being shrunk by
\\includegraphics, and 300 dpi.
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

# replot_all_families_1d.py sizes, scaled down slightly: the panels here are
# larger than that figure's single axes, so matched point sizes read as heavy
mpl.rcParams.update({
    "font.size": 6.2, "axes.labelsize": 6.6, "axes.titlesize": 7.0,
    "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "legend.fontsize": 5.6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "mathtext.fontset": "cm", "font.family": "serif",
})

# printed width: full \textwidth (17.6 cm) = 6.93 in
FIGSIZE = (6.93, 3.25)


def main():
    d = np.load(f"{OUT}/error_maps_2d.npz")
    p_grid, perp_grid = d["p_par"], d["p_perp"]
    fams = sorted(int(k.split("_")[1]) for k in d.files if k.startswith("err_"))

    fig, axes = plt.subplots(2, 4, figsize=FIGSIZE, sharex=True, sharey=True)
    for f, ax in zip(fams, axes.ravel()):
        em = d[f"err_{f}"]
        im = ax.pcolormesh(p_grid, perp_grid, np.log10(em + 1e-16),
                           shading="auto", cmap="viridis", vmin=-5, vmax=-1,
                           rasterized=True)
        ax.set_title(f"{f}", pad=2.5)
        ax.tick_params(direction="in", top=True, right=True, length=2.4,
                       color="w")
        ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
        ax.set_yticks([0.0, 0.4, 0.8, 1.2])

    fig.supxlabel(r"$p_\parallel$ (a.u.)", fontsize=6.6, y=0.02)
    fig.supylabel(r"$p_\perp$ (a.u.)", fontsize=6.6, x=0.015)
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015, aspect=26)
    cb.set_label(r"$\log_{10}|\Delta\theta_s|$", fontsize=6.6)
    cb.ax.tick_params(labelsize=5.8, direction="in", length=2.4, width=0.6)
    cb.outline.set_linewidth(0.6)

    fig.savefig(f"{OUT}/error_maps_2d.png", dpi=300, bbox_inches="tight")
    if FIGDIR:
        fig.savefig(f"{FIGDIR}/error_maps_2d.png", dpi=300, bbox_inches="tight")
    print("wrote error_maps_2d.png (replot only, no retraining)")
    print("per-family median |d theta_s| (unchanged):")
    med = []
    for f in fams:
        m = np.median(d[f"err_{f}"])
        med.append(m)
        print(f"  family {f}: {m:.3e}")
    print(f"median across all 8: {np.median(med):.3e}  "
          f"[{min(med):.3e}, {max(med):.3e}]")


if __name__ == "__main__":
    main()
