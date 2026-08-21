"""
Pair A: ionisation and return times in the complex plane, with residuals.

TOP PANEL:  Im(theta) vs Re(theta) -- complex-plane trajectories.
            p_parallel is the *parameter* along each curve, not an axis.
            Four curves: {ionisation, return} x {short, long}.
            Continuation = solid reference. PINN = dashed overlay.
            Markers show the coalescence point (where short and long meet).

BOTTOM:     Errors vs p_parallel. Two quantities, deliberately together:
              |dtheta| = |theta_PINN - theta_continuation|   (true error)
              ||F||    = physics residual of the PINN solution
            The gap between them near the caustic is the point: the
            residual stays small while the true error rises, i.e. the
            residual fails to detect the error floor.

USAGE
-----
Fill in load_data() with your own arrays, then:
    python plot_pair_a_complex.py
To preview the layout with fake data:
    python plot_pair_a_complex.py --demo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out

OUT = out("future_work_pmd")

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
C_ION = "#452A7A"    # ionisation theta_s
C_RET = "#C8506E"    # return theta_r
C_GREY = "#6B6570"
PC_RE = 2.197280     # Re(p_c), for the vertical marker in the lower panel


# ----------------------------------------------------------------------
# DATA LOADING -- replace this with your own arrays
# ----------------------------------------------------------------------
def load_data():
    """
    Loads output/future_work_pmd/pair_a_theta_1d.npz, produced by
    make_pair_a_theta_1d.py. Key names match exactly -- no translation needed.
    """
    d = np.load(f"{OUT}/pair_a_theta_1d.npz")
    return {k: d[k] for k in d.files}


def make_demo_data(n=400):
    """SYNTHETIC data with the right qualitative shape. NOT physics."""
    pc = PC_RE
    p = np.linspace(0.4, pc - 1e-3, n)
    d = np.sqrt(pc - p)                      # fold scaling

    # Coalescence points (arbitrary but plausible-looking)
    tc_s = 1.55 + 0.62j
    tc_r = 4.30 + 0.16j

    cs = (0.95 + 0.22j)
    cr = (1.35 - 0.30j)

    cont_s_short = tc_s + cs * d + 0.05 * (pc - p)
    cont_s_long = tc_s - cs * d + 0.05 * (pc - p)
    cont_r_short = tc_r + cr * d - 0.03 * (pc - p)
    cont_r_long = tc_r - cr * d - 0.03 * (pc - p)

    # PINN: tracks well, then rounds off the cusp near the fold
    #       (bounded-Lipschitz smoothing -> separation saturates)
    floor = 0.02
    soft = np.sqrt(d**2 + floor**2) - floor + 1e-4
    pinn_s_short = tc_s + cs * soft + 0.05 * (pc - p)
    pinn_s_long = tc_s - cs * soft + 0.05 * (pc - p)
    pinn_r_short = tc_r + cr * soft - 0.03 * (pc - p)
    pinn_r_long = tc_r - cr * soft - 0.03 * (pc - p)

    rng = np.random.default_rng(0)
    for a in (pinn_s_short, pinn_s_long, pinn_r_short, pinn_r_long):
        a += (rng.normal(0, 3e-4, n) + 1j * rng.normal(0, 3e-4, n))

    # Residual stays small even where the true error grows -- the point.
    res_short = 10 ** (-5.6 + 0.25 * rng.normal(0, 1, n)) + 2e-7 / (d + 1e-3)
    res_long = 10 ** (-5.4 + 0.25 * rng.normal(0, 1, n)) + 2e-7 / (d + 1e-3)

    return {
        "p": p,
        "cont_s_short": cont_s_short, "cont_s_long": cont_s_long,
        "cont_r_short": cont_r_short, "cont_r_long": cont_r_long,
        "pinn_s_short": pinn_s_short, "pinn_s_long": pinn_s_long,
        "pinn_r_short": pinn_r_short, "pinn_r_long": pinn_r_long,
        "res_short": res_short, "res_long": res_long,
        "_demo": True,
    }


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def make_figure(D, outfile="pair_a_complex.png"):
    is_demo = D.get("_demo", False)
    p = D["p"]

    fig, (ax, axe) = plt.subplots(
        2, 1, figsize=(7.4, 8.2),
        gridspec_kw={"height_ratios": [2.15, 1.0], "hspace": 0.30},
    )

    # ---------------- TOP: complex plane ----------------
    # Continuation reference: solid
    ax.plot(D["cont_s_short"].real, D["cont_s_short"].imag,
            color=C_ION, lw=2.0, zorder=3)
    ax.plot(D["cont_s_long"].real, D["cont_s_long"].imag,
            color=C_ION, lw=2.0, zorder=3)
    ax.plot(D["cont_r_short"].real, D["cont_r_short"].imag,
            color=C_RET, lw=2.0, zorder=3)
    ax.plot(D["cont_r_long"].real, D["cont_r_long"].imag,
            color=C_RET, lw=2.0, zorder=3)

    # PINN: dashed overlay
    for key, col in (("pinn_s_short", C_ION), ("pinn_s_long", C_ION),
                     ("pinn_r_short", C_RET), ("pinn_r_long", C_RET)):
        ax.plot(D[key].real, D[key].imag, color=col, lw=1.5,
                ls=(0, (4, 2.6)), alpha=0.95, zorder=4)

    # Mark the start (low p) and the coalescence (high p) ends
    for key, col in (("cont_s_short", C_ION), ("cont_s_long", C_ION),
                     ("cont_r_short", C_RET), ("cont_r_long", C_RET)):
        ax.plot(D[key].real[0], D[key].imag[0], "o", color=col,
                ms=5, mfc="white", mew=1.5, zorder=6)
    for key, col in (("cont_s_short", C_ION), ("cont_r_short", C_RET)):
        ax.plot(D[key].real[-1], D[key].imag[-1], "*", color=col,
                ms=15, zorder=7)

    ax.annotate("short and long merge here\n(coalescence, $p\\to p_c$)",
                xy=(D["cont_s_short"].real[-1], D["cont_s_short"].imag[-1]),
                xytext=(18, 26), textcoords="offset points",
                fontsize=8.5, color=C_GREY,
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.1))

    ax.set_xlabel(r"$\mathrm{Re}\,\theta$   (rad)")
    ax.set_ylabel(r"$\mathrm{Im}\,\theta$   (rad)")
    ax.set_title("Pair A in the complex time plane\n"
                 r"(curves parameterised by $p_\parallel$)", fontsize=11)
    ax.grid(alpha=0.2, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [
        Line2D([], [], color=C_ION, lw=2.0, label=r"ionisation $\theta_s$"),
        Line2D([], [], color=C_RET, lw=2.0, label=r"return $\theta_r$"),
        Line2D([], [], color="k", lw=2.0, label="continuation (reference)"),
        Line2D([], [], color="k", lw=1.5, ls=(0, (4, 2.6)), label="PINN"),
        Line2D([], [], color=C_GREY, marker="o", mfc="white", ls="none",
               label=r"start (low $p_\parallel$)"),
        Line2D([], [], color=C_GREY, marker="*", ls="none", ms=11,
               label="coalescence"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False,
              loc="best", ncol=2)

    # ---------------- BOTTOM: errors vs p ----------------
    err_s = np.abs(D["pinn_s_short"] - D["cont_s_short"])
    err_l = np.abs(D["pinn_s_long"] - D["cont_s_long"])

    axe.semilogy(p, err_s, color=C_ION, lw=1.8,
                 label=r"$|\Delta\theta_s|$ vs continuation, short")
    axe.semilogy(p, err_l, color=C_ION, lw=1.8, ls=(0, (4, 2.6)),
                 label=r"$|\Delta\theta_s|$ vs continuation, long")
    axe.semilogy(p, D["res_short"], color=C_GREY, lw=1.4,
                 label=r"physics residual $\|F\|$, short")
    axe.semilogy(p, D["res_long"], color=C_GREY, lw=1.4, ls=(0, (4, 2.6)),
                 label=r"physics residual $\|F\|$, long")

    axe.axvline(PC_RE, color=C_RET, lw=1.2, ls=":")
    axe.annotate(r"$\mathrm{Re}\,p_c$", xy=(PC_RE, axe.get_ylim()[1]),
                 xytext=(-30, -14), textcoords="offset points",
                 fontsize=8.5, color=C_RET)

    axe.set_xlabel(r"$p_\parallel$  (a.u.)")
    axe.set_ylabel("error / residual")
    axe.set_title("Error against continuation, and physics residual",
                  fontsize=10)
    axe.grid(alpha=0.2, lw=0.6, which="both")
    axe.set_axisbelow(True)
    axe.legend(fontsize=7.5, frameon=False, ncol=2)
    for s in ("top", "right"):
        axe.spines[s].set_visible(False)

    if is_demo:
        fig.text(0.5, 0.5, "SYNTHETIC DEMO DATA", fontsize=34,
                 color="red", alpha=0.14, ha="center", va="center",
                 rotation=27, weight="bold", zorder=50)

    fig.savefig(outfile, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"wrote {outfile}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="preview layout with synthetic data")
    ap.add_argument("-o", "--out", default=f"{OUT}/pair_a_complex.png")
    a = ap.parse_args()

    D = make_demo_data() if a.demo else load_data()
    make_figure(D, a.out)
