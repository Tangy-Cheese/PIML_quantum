import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
import bichromatic as B

OUT = out("diagnostics")
GT = out("ground_truth")
FIGDIR = figure_dir()

# Matches replot_error_maps_2d.py / replot_all_families_1d.py.
mpl.rcParams.update({
    "font.size": 6.2, "axes.labelsize": 6.6, "axes.titlesize": 7.0,
    "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "legend.fontsize": 5.6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "mathtext.fontset": "cm", "font.family": "serif",
    "xtick.direction": "in", "ytick.direction": "in",
})

# The dissertation's figure hues. Blue/red only: the document's green sits at
# dEcvd 6.3 against the red under protanopia, so it is not used as a third
# categorical identity here.
BLUE, RED, GREY = "#2D5591", "#B23A48", "#6E6E6E"

# printed at 0.85\linewidth of a two-column-spanning figure* (17.6 cm)
FIGSIZE = (5.89, 2.45)

PHI_LO, PHI_HI, N_PHI = 0.20 * np.pi, 0.95 * np.pi, 301


def track(phi_star, sol_star):
    """Continue the caustic in phi, warm-started from the phi* solution."""
    B.RATIO = 0.3
    phis = np.linspace(PHI_LO, PHI_HI, N_PHI)
    j0 = int(np.argmin(np.abs(phis - phi_star)))

    pc = np.full(N_PHI, np.nan, dtype=complex)
    for direction in (+1, -1):
        v = np.asarray(sol_star, dtype=float).copy()
        idx = range(j0, N_PHI) if direction > 0 else range(j0, -1, -1)
        for j in idx:
            B.PHI = phis[j]
            sol, ok = B.caustic_point(list(v), 0.0, phis[j])
            if not ok:
                break
            v = sol
            pc[j] = sol[4] + 1j * sol[5]
    return phis, pc


def main():
    crit = np.load(GT / "critical_phi.npz", allow_pickle=True)
    phi_star = float(crit["phi_star"])
    sol_star = np.asarray(crit["sol"], dtype=float)
    print(f"phi* = {phi_star / np.pi:.4f} pi, "
          f"p_c = {float(crit['pc_re']):.6f} + {float(crit['pc_im']):.2e}i")

    phis, pc = track(phi_star, sol_star)
    good = np.isfinite(pc)
    print(f"tracked {good.sum()}/{N_PHI} phases, "
          f"phi/pi in [{phis[good].min() / np.pi:.3f}, {phis[good].max() / np.pi:.3f}]")

    # Independently computed validation phases (make_bichrom_phase_data.py).
    val = []
    for tag in ("0.5pi", "0.6pi", "0.9pi"):
        z = np.load(out("bichromatic") / f"bichrom_{tag}_pinn.npz")
        val.append((float(z["phi"]), float(z["pc_re"]), float(z["pc_im"]), tag))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=FIGSIZE)

    # ---- (a) locus in the complex momentum plane ------------------------
    ax0.axhline(0.0, color=GREY, lw=0.6, zorder=1)
    ax0.plot(pc[good].real, pc[good].imag, "-", color=BLUE, lw=1.2, zorder=3)
    ax0.plot(float(crit["pc_re"]), 0.0, "o", ms=4.5, mfc=RED, mec="white",
             mew=0.7, zorder=5)
    ax0.annotate(r"$\phi^\ast$", (float(crit["pc_re"]), 0.0),
                 textcoords="offset points", xytext=(1, -9),
                 color=RED, fontsize=6.4)
    for _, r, i, tag in val:
        ax0.plot(r, i, "s", ms=3.2, mfc="white", mec=BLUE, mew=0.8, zorder=4)
        ax0.annotate(tag.replace("pi", r"$\pi$"), (r, i),
                     textcoords="offset points", xytext=(3.5, -1.5),
                     color=GREY, fontsize=5.4)
    # direction of increasing phi
    k = np.flatnonzero(good)[len(np.flatnonzero(good)) // 2]
    ax0.annotate("", xy=(pc[k + 4].real, pc[k + 4].imag),
                 xytext=(pc[k].real, pc[k].imag),
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9,
                                 shrinkA=0, shrinkB=0))
    ax0.annotate(r"increasing $\phi$", (pc[k].real, pc[k].imag),
                 textcoords="offset points", xytext=(-4, 7),
                 color=BLUE, fontsize=5.6, ha="right")
    ax0.set_xlabel(r"$\mathrm{Re}\,p_c$ (a.u.)")
    ax0.set_ylabel(r"$\mathrm{Im}\,p_c$ (a.u.)")
    ax0.set_title("(a) caustic locus in complex $p$", loc="left")
    ax0.grid(True, lw=0.3, color=GREY, alpha=0.25)
    ax0.set_axisbelow(True)

    # ---- (b) |Im p_c| against phi ---------------------------------------
    ax1.semilogy(phis[good] / np.pi, np.abs(pc[good].imag), "-",
                 color=BLUE, lw=1.2, zorder=3)
    ax1.axvline(phi_star / np.pi, color=RED, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax1.annotate(r"$\phi^\ast=0.2489\pi$", (phi_star / np.pi, 1.0),
                 xycoords=("data", "axes fraction"),
                 textcoords="offset points", xytext=(3, -8),
                 color=RED, fontsize=6.0)
    for p, _, i, tag in val:
        ax1.plot(p / np.pi, abs(i), "s", ms=3.2, mfc="white", mec=BLUE,
                 mew=0.8, zorder=4)
    ax1.set_xlabel(r"relative phase $\phi/\pi$")
    ax1.set_ylabel(r"$|\mathrm{Im}\,p_c|$ (a.u.)")
    ax1.set_title(r"(b) approach to the real axis", loc="left")
    ax1.grid(True, which="both", lw=0.3, color=GREY, alpha=0.25)
    ax1.set_axisbelow(True)

    h = [plt.Line2D([], [], color=BLUE, lw=1.2),
         plt.Line2D([], [], ls="none", marker="s", ms=3.2, mfc="white",
                    mec=BLUE, mew=0.8),
         plt.Line2D([], [], ls="none", marker="o", ms=4.5, mfc=RED,
                    mec="white", mew=0.7)]
    ax1.legend(h, ["continued caustic", "validation phases (Sec. 6.2)",
                   r"$\phi^\ast$"],
               loc="lower right", frameon=False, handlelength=1.6,
               borderaxespad=0.3)

    fig.tight_layout(pad=0.35, w_pad=1.1)
    for d in (FIGDIR, OUT):
        fig.savefig(Path(d) / "phistar_construction.png", dpi=300)
    print("wrote phistar_construction.png")


if __name__ == "__main__":
    main()
