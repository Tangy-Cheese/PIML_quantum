import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
OUT = out("polish")
FIGDIR = figure_dir()

mpl.rcParams.update({
    "font.size": 7.2, "axes.labelsize": 7.6, "axes.titlesize": 8.0,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "mathtext.fontset": "cm", "font.family": "serif",
})

C_PHI = "#b23a48"    # phi*
C_MONO = "#2d5591"   # ordinary monochromatic
C_REF = "#3f3f46"


def load(case):
    return np.load(f"{OUT}/polish_{case}.npz")


def band(ax, x, Y, color, label, ls="-"):
    """Median across seeds with a min-max band."""
    med = np.median(Y, axis=0)
    ax.plot(x, med, ls, color=color, lw=1.1, label=label)
    ax.fill_between(x, Y.min(axis=0), Y.max(axis=0), color=color, alpha=0.18,
                    lw=0)
    return med


def main():
    d_phi, d_mono = load("phistar"), load("mono")

    # ---------------- figure 1: residual + separation vs distance -----------
    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(6.95, 2.9))

    for d, col, lab in ((d_mono, C_MONO, "ordinary"), (d_phi, C_PHI, r"$\phi^\ast$")):
        dist = d["dist"]
        # both branches pooled: residual is a per-branch quantity
        R = np.concatenate([d["A_res_pinn"], d["B_res_pinn"]], axis=0)
        band(ax_r, dist, R, col, lab)
    ax_r.set_xscale("log"); ax_r.set_yscale("log")
    ax_r.invert_xaxis()
    ax_r.set_xlabel(r"$p_c-p_\parallel$  (a.u.)")
    ax_r.set_ylabel(r"physics residual $\|\mathbf{F}\|$")
    ax_r.set_title("(a) residual on approach")
    ax_r.legend(frameon=False, loc="lower left")
    ax_r.tick_params(direction="in", top=True, right=True, length=2.4)

    for d, col, lab in ((d_mono, C_MONO, "ordinary"), (d_phi, C_PHI, r"$\phi^\ast$")):
        dist, gap = d["dist"], d["gap"]
        ax_s.plot(dist, gap, "-", color=col, lw=1.1, label=f"{lab}, reference")
        sep_nn = np.abs(d["A_ths"] - d["B_ths"])     # (seeds, N)
        band(ax_s, dist, sep_nn, col, f"{lab}, PINN", ls="--")
    g = d_phi["gap"]; dd = d_phi["dist"]
    ax_s.plot(dd, g[-1] * np.sqrt(dd / dd[-1]), ":", color=C_REF, lw=0.9,
              label=r"$\sqrt{p_c-p}$")
    ax_s.set_xscale("log"); ax_s.set_yscale("log")
    ax_s.invert_xaxis()
    ax_s.set_xlabel(r"$p_c-p_\parallel$  (a.u.)")
    ax_s.set_ylabel(r"branch separation $|\theta_s^A-\theta_s^B|$")
    ax_s.set_title("(b) branch separation on approach")
    ax_s.legend(frameon=False, loc="lower left", ncol=1)
    ax_s.tick_params(direction="in", top=True, right=True, length=2.4)

    fig.tight_layout()
    fig.savefig(f"{OUT}/polish_approach.png", dpi=300)
    if FIGDIR:
        fig.savefig(f"{FIGDIR}/polish_approach.png", dpi=300)
    print(f"wrote polish_approach.png")
    plt.close(fig)

    # ---------------- figure 2: polish improvement --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(6.95, 2.9))
    for ax, d, col, title in ((axes[0], d_mono, C_MONO, "(a) ordinary coalescence"),
                              (axes[1], d_phi, C_PHI, r"(b) $\phi^\ast$ coalescence")):
        dist = d["dist"]
        EP = np.concatenate([d["A_err_pinn"], d["B_err_pinn"]], axis=0)
        EQ = np.concatenate([d["A_err_pol"], d["B_err_pol"]], axis=0)
        band(ax, dist, EP, col, "PINN")
        band(ax, dist, np.maximum(EQ, 1e-17), C_REF, "PINN + 4 Newton steps",
             ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"$p_c-p_\parallel$  (a.u.)")
        ax.set_ylabel(r"$|\Delta\theta_s|$")
        ax.set_title(title)
        ax.legend(frameon=False, loc="center left", framealpha=0.9)
        ax.tick_params(direction="in", top=True, right=True, length=2.4)
    fig.tight_layout()
    fig.savefig(f"{OUT}/polish_comparison.png", dpi=300)
    if FIGDIR:
        fig.savefig(f"{FIGDIR}/polish_comparison.png", dpi=300)
    print(f"wrote polish_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
