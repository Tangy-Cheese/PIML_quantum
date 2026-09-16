"""Bichromatic phase-sweep figure: orbit pair A at three ordinary relative
phases, matching the layout of the monochromatic three-intensity figure.

Rows are relative phase (0.5 pi, 0.6 pi, 0.9 pi), columns are Re theta,
Im theta and the network's own physics residual -- the same three columns as
the intensity figure, so the two read as a pair: the intensity figure asks
whether the method holds as the coalescence relocates within one field, this
one asks whether it holds across the second field's own parameter space.

Also prints the summary table body for the write-up.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
OUT = out("bichromatic")
FIGDIR = figure_dir()
TAGS = ["0.5pi", "0.6pi", "0.9pi"]
LABELS = {"0.5pi": r"$\phi=0.5\pi$", "0.6pi": r"$\phi=0.6\pi$",
          "0.9pi": r"$\phi=0.9\pi$"}

mpl.rcParams.update({
    "font.size": 7.2, "axes.labelsize": 7.6, "axes.titlesize": 8.0,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "mathtext.fontset": "cm", "font.family": "serif",
})
C_S = "#2d5591"
C_R = "#b23a48"
C_PC = "#6e6e6e"


def load(tag):
    g = np.load(f"{OUT}/bichrom_{tag}_pinn.npz")
    n = int(g["n_seeds"])
    d = dict(p=g["p_par"], p_c=float(g["pc_re"]), pc_im=float(g["pc_im"]),
             kappa=float(g["kappa"]), n=n)
    for b in ("short", "long"):
        d[f"{b}_cont_s"] = g[f"{b}_cont_s"]
        d[f"{b}_cont_r"] = g[f"{b}_cont_r"]
        d[f"{b}_ths"] = np.array([g[f"{b}_ths_{i}"] for i in range(n)])
        d[f"{b}_thr"] = np.array([g[f"{b}_thr_{i}"] for i in range(n)])
        d[f"{b}_res"] = np.array([g[f"{b}_res_{i}"] for i in range(n)])
        d[f"{b}_err"] = np.array([g[f"{b}_err_{i}"] for i in range(n)])
    return d


def main():
    data = {t: load(t) for t in TAGS}
    fig, axes = plt.subplots(3, 3, figsize=(6.95, 6.55))

    for row, tag in enumerate(TAGS):
        d = data[tag]
        p, p_c = d["p"], d["p_c"]
        ax_re, ax_im, ax_rs = axes[row]

        for b, ls in (("short", "-"), ("long", "--")):
            med_err = np.median(d[f"{b}_err"], axis=1)
            rep = int(np.argsort(med_err)[len(med_err) // 2])
            cs, cr = d[f"{b}_cont_s"], d[f"{b}_cont_r"]
            ps, pr = d[f"{b}_ths"][rep], d[f"{b}_thr"][rep]
            sl = slice(4, None, 10)

            ax_re.plot(p, cs.real, ls, color=C_S, lw=1.0)
            ax_re.plot(p, cr.real, ls, color=C_R, lw=1.0)
            ax_re.plot(p[sl], ps.real[sl], "o", ms=1.9, mfc="none", mec=C_S, mew=0.55)
            ax_re.plot(p[sl], pr.real[sl], "o", ms=1.9, mfc="none", mec=C_R, mew=0.55)

            ax_im.plot(p, cs.imag, ls, color=C_S, lw=1.0)
            ax_im.plot(p, cr.imag, ls, color=C_R, lw=1.0)
            ax_im.plot(p[sl], ps.imag[sl], "o", ms=1.9, mfc="none", mec=C_S, mew=0.55)
            ax_im.plot(p[sl], pr.imag[sl], "o", ms=1.9, mfc="none", mec=C_R, mew=0.55)

            R = d[f"{b}_res"]
            ax_rs.semilogy(p, np.median(R, axis=0), ls, color="#3f3f46", lw=0.95)

        for ax in (ax_re, ax_im, ax_rs):
            ax.axvline(p_c, color=C_PC, ls=":", lw=0.8, zorder=0)
            ax.set_xlim(p[0], p[-1])
            ax.tick_params(direction="in", top=True, right=True, length=2.4)
            ax.set_box_aspect(1)
            ax.set_xlabel(r"$p_\parallel$  (a.u.)")

        allre = np.concatenate([d[f"{b}_cont_{k}"].real
                                for b in ("short", "long") for k in ("s", "r")])
        pad = 0.08 * (allre.max() - allre.min())
        ax_re.set_ylim(allre.min() - pad, allre.max() + pad)
        ax_re.set_ylabel(r"$\mathrm{Re}\,\theta$  (rad)")
        ax_im.set_ylabel(r"$\mathrm{Im}\,\theta$  (rad)")
        ax_rs.set_ylabel(r"$\|\mathbf{F}\|$")
        ax_re.text(0.035, 0.93, LABELS[tag], transform=ax_re.transAxes,
                   va="top", ha="left", fontsize=7.0,
                   bbox=dict(fc="white", ec="0.75", lw=0.4, pad=1.6))
        ax_rs.text(p_c, ax_rs.get_ylim()[0], r"$p_c$", ha="center",
                   va="bottom", fontsize=6.4, color=C_PC)
        if row == 0:
            ax_re.set_title("real parts")
            ax_im.set_title("imaginary parts")
            ax_rs.set_title("network residual")

    handles = [
        Line2D([], [], color=C_S, lw=1.0, label=r"$\theta_s$"),
        Line2D([], [], color=C_R, lw=1.0, label=r"$\theta_r$"),
        Line2D([], [], color="0.3", lw=1.0, ls="-", label="short branch"),
        Line2D([], [], color="0.3", lw=1.0, ls="--", label="long branch"),
        Line2D([], [], color="0.3", ls="none", marker="o", ms=2.4, mfc="none",
               mew=0.6, label="PINN"),
        Line2D([], [], color=C_PC, ls=":", lw=0.9, label=r"$p_c$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.005), handlelength=1.6, columnspacing=1.5)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"{OUT}/bichrom_phases.png", dpi=300)
    if FIGDIR:
        fig.savefig(f"{FIGDIR}/bichrom_phases.png", dpi=300)
    print("wrote bichrom_phases.png")
    plt.close(fig)

    print("\n" + "=" * 74)
    print("phase sweep: median over momentum, then across 5 seeds [min, max]")
    print("=" * 74)
    print(f"{'phi':>7} {'|Im p_c|':>10} {'kappa':>9} {'branch':<6} "
          f"{'||F||':<26} {'|d theta_s|'}")
    for tag in TAGS:
        d = data[tag]
        for b in ("short", "long"):
            mr = np.median(d[f"{b}_res"], axis=1)
            me = np.median(d[f"{b}_err"], axis=1)
            head = f"{tag:>7} {abs(d['pc_im']):10.2e} {d['kappa']:9.0f}" if b == "short" \
                   else " " * 28
            print(f"{head} {b:<6} {np.median(mr):.2e} [{mr.min():.2e},{mr.max():.2e}]  "
                  f"{np.median(me):.2e} [{me.min():.2e},{me.max():.2e}]")

    print("\nLaTeX table body:")
    for tag in TAGS:
        d = data[tag]
        for b in ("short", "long"):
            mr = np.median(d[f"{b}_res"], axis=1)
            me = np.median(d[f"{b}_err"], axis=1)
            r, e = np.median(mr), np.median(me)
            er, ee = int(np.floor(np.log10(r))), int(np.floor(np.log10(e)))
            lab = LABELS[tag] if b == "short" else ""
            print(f"    {lab:<14} & {b:<6} & ${r/10**er:.2f}\\times10^{{{er}}}$ & "
                  f"$[{mr.min()/10**er:.2f},\\,{mr.max()/10**er:.2f}]$ & "
                  f"${e/10**ee:.2f}\\times10^{{{ee}}}$ & "
                  f"$[{me.min()/10**ee:.2f},\\,{me.max()/10**ee:.2f}]$ \\\\")


if __name__ == "__main__":
    main()
