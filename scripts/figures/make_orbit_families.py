import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir

OUT = out("families")
FIGDIR = figure_dir()

RATIO = 0.3            # A_2 / A_1, for the schematic bichromatic curve
PHI = 0.6 * np.pi

mpl.rcParams.update({
    "font.size": 7.2, "axes.labelsize": 7.6, "axes.titlesize": 8.0,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.0, "mathtext.fontset": "cm", "font.family": "serif",
})

PAIR_COLOURS = ["#2d5591", "#8a6d1f"]
GREY = "#6e6e6e"

# Idealised (label, colour index, release phase, return phase), in units
# of theta/pi, hand-placed and deliberately spread apart so every point
# reads cleanly on the curve. Only the shortest-delay pair (1,2) and the
# longest-delay pair (7,8 of 8) are shown.
MONO = [
    (1, 0, 0.50, 0.86),
    (2, 0, 0.62, 1.34),
    (7, 1, 1.50, 4.58),
    (8, 1, 1.64, 5.18),
]

# Bichromatic: half-cycle symmetry broken, so the release phases of the
# shortest and longest pair sit in different clusters rather than the same
# two as panel (a). 20 solutions exist in the real system within the same
# delay window (see caption); only the shortest- and longest-delay pair
# are sketched here.
BICH = [
    (1, 0, 0.14, 0.52),
    (2, 0, 0.24, 0.98),
    (19, 0, 2.50, 3.18),
    (20, 0, 2.64, 3.78),
]


def field(th, bich):
    return np.cos(th) + (RATIO * np.cos(2 * th + PHI) if bich else 0.0)


def draw_panel(ax, rows, bich, xlo, xhi, title):
    """Vector potential with each family's release and return phase marked
    directly on the curve, labelled individually since the hand-picked
    phases are already spread apart.
    """
    th = np.linspace(xlo * np.pi, xhi * np.pi, 3000)
    ax.plot(th / np.pi, field(th, bich), color="k", lw=0.9, zorder=2)
    ax.axhline(0, color=GREY, lw=0.4, zorder=1)

    for label, ci, ts, tr in rows:
        colour = GREY if bich else PAIR_COLOURS[ci]
        ax.plot(ts, field(ts * np.pi, bich), "o", ms=4.5, color=colour,
                mew=0, zorder=4)
        ax.plot(tr, field(tr * np.pi, bich), "s", ms=4.5, mfc="white",
                mec=colour, mew=0.9, zorder=4)
        ax.annotate(str(label), (ts, field(ts * np.pi, bich)),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=6.6, color=colour)
        ax.annotate(str(label), (tr, field(tr * np.pi, bich)),
                    textcoords="offset points", xytext=(0, -12), ha="center",
                    fontsize=6.6, color=colour)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-2.0, 2.0)
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"phase $\theta/\pi$")
    ax.set_title(title, loc="left")


def main():
    xlo, xhi = 0.0, 5.4

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 3.3), sharex=True,
                             sharey=True)
    draw_panel(axes[0], MONO, False, xlo, xhi,
               "(a) monochromatic: shortest & longest pair (of 8)")
    draw_panel(axes[1], BICH, True, xlo, xhi,
               r"(b) $\omega$--$2\omega$, $\phi=0.6\pi$: shortest & "
               r"longest pair (of 20)")
    axes[0].set_xlabel("")
    for a in axes:
        a.set_ylabel(r"$A/A_1$")

    axes[0].plot([], [], "o", ms=4.5, color=GREY, mew=0,
                 label=r"release $\mathrm{Re}\,\theta_s$")
    axes[0].plot([], [], "s", ms=4.5, mfc="white", mec=GREY, mew=0.9,
                 label=r"return $\mathrm{Re}\,\theta_r$")
    axes[0].legend(loc="upper right", frameon=False, ncol=2,
                   handletextpad=0.4, columnspacing=1.2, borderaxespad=0.3)

    fig.tight_layout()
    png = OUT / "orbit_families.png"
    fig.savefig(png, dpi=400)
    plt.close(fig)
    print(f"wrote {png}")
    if FIGDIR is not None:
        shutil.copy(png, FIGDIR / "orbit_families.png")
        print(f"copied to {FIGDIR / 'orbit_families.png'}")


if __name__ == "__main__":
    main()
