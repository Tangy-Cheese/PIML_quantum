import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
from classical_solvers import monochromatic as C
from pinn_common import PINN, train

torch.set_default_dtype(torch.float64)
OUT = out("pair_a")
FIGDIR = figure_dir()

mpl.rcParams.update({
    "font.size": 7.2,
    "axes.labelsize": 7.6,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.4,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "mathtext.fontset": "cm",
    "font.family": "serif",
})

C_S = "#2d5591"
C_R = "#b23a48"
C_PC = "#6e6e6e"

OMEGA = 0.057
I_P = 0.5
EPOCHS = 2500
INTENSITIES = [0.5e14, 1.0e14, 1.5e14]
P_LO_FRAC, P_HI_FRAC = 0.17, 1.16


def set_field(E_0):
    C.E_0 = E_0
    C.A_0 = E_0 / OMEGA
    C.U_P = E_0 ** 2 / (4 * OMEGA ** 2)


def box(vals, pad=3.0, floor=1e-3):
    lo, hi = vals.min(0), vals.max(0)
    c = 0.5 * (lo + hi)
    h = np.maximum(0.5 * (hi - lo) * pad, floor)
    return list(c), list(h)


def run_intensity(I_Wcm2):
    E0 = np.sqrt(I_Wcm2 / 3.5094e16)
    set_field(E0)
    A0 = C.A_0
    p_cut = np.sqrt(20 * C.U_P)

    scale = p_cut / 2.0948
    v0 = [4.96186, 0.96739, 9.26187, -0.01340,
          2.197280 * scale, 0.017697 * scale]
    sol, ok = C.caustic_point(v0, 0.0)
    p_c = sol[4] if ok else 1.05 * p_cut

    p_grid = np.linspace(P_LO_FRAC * p_c, P_HI_FRAC * p_c, 220)

    seeds = C.find_orbits(p_grid[0], 0.0)
    seeds_sorted = sorted(seeds, key=lambda x: x[2] - x[0])
    sA, sB = seeds_sorted[1], seeds_sorted[2]

    xa, xb = sA.copy(), sB.copy()
    XA, XB = [], []
    for p in p_grid:
        ya, oka = C.newton(xa, p, 0.0)
        yb, okb = C.newton(xb, p, 0.0)
        if oka:
            xa = ya
        if okb:
            xb = yb
        XA.append(xa.copy())
        XB.append(xb.copy())
    XA, XB = np.array(XA), np.array(XB)

    p_t = torch.tensor(p_grid)
    pp0 = torch.zeros_like(p_t)
    z = ((p_t - p_grid.mean()) / p_grid.std()).unsqueeze(1)

    out = {"p": p_grid, "p_c": p_c}
    for tag, Xg in (("short", XA), ("long", XB)):
        c, w = box(Xg, pad=3.0)
        torch.manual_seed(0)
        m = PINN(c, w, input_dim=1)
        loss = train(m, z, p_t, pp0, A0, I_P, EPOCHS)
        with torch.no_grad():
            ths, thr = m(z)
        ths, thr = ths.numpy(), thr.numpy()
        res = np.array([
            np.linalg.norm(C.F_real([s.real, s.imag, r.real, r.imag], p, 0.0))
            for s, r, p in zip(ths, thr, p_grid)])
        cont_s = Xg[:, 0] + 1j * Xg[:, 1]
        cont_r = Xg[:, 2] + 1j * Xg[:, 3]
        err = np.abs(ths - cont_s)
        out[tag] = dict(cont_s=cont_s, cont_r=cont_r, pinn_s=ths, pinn_r=thr, res=res)
        print(f"  I={I_Wcm2:.1e} {tag:5s}: loss {loss:.2e}  "
              f"median |F| {np.median(res):.2e}  median err {np.median(err):.2e}")

    return out


def main():
    fig, axes = plt.subplots(3, 3, figsize=(6.95, 6.55))

    for row, I_Wcm2 in enumerate(INTENSITIES):
        print(f"I = {I_Wcm2:.1e} W/cm^2")
        d = run_intensity(I_Wcm2)
        p, p_c = d["p"], d["p_c"]
        ax_re, ax_im, ax_rs = axes[row]

        for br, ls in (("short", "-"), ("long", "--")):
            ax_re.plot(p, d[br]["cont_s"].real, ls, color=C_S, lw=1.0)
            ax_re.plot(p, d[br]["cont_r"].real, ls, color=C_R, lw=1.0)
            sl = slice(4, None, 10)
            ax_re.plot(p[sl], d[br]["pinn_s"].real[sl], "o", ms=1.9, mfc="none", mec=C_S, mew=0.55)
            ax_re.plot(p[sl], d[br]["pinn_r"].real[sl], "o", ms=1.9, mfc="none", mec=C_R, mew=0.55)

            ax_im.plot(p, d[br]["cont_s"].imag, ls, color=C_S, lw=1.0)
            ax_im.plot(p, d[br]["cont_r"].imag, ls, color=C_R, lw=1.0)
            ax_im.plot(p[sl], d[br]["pinn_s"].imag[sl], "o", ms=1.9, mfc="none", mec=C_S, mew=0.55)
            ax_im.plot(p[sl], d[br]["pinn_r"].imag[sl], "o", ms=1.9, mfc="none", mec=C_R, mew=0.55)

            ax_rs.semilogy(p, d[br]["res"], ls, color="#3f3f46", lw=0.95)

        for ax in (ax_re, ax_im, ax_rs):
            ax.axvline(p_c, color=C_PC, ls=":", lw=0.8, zorder=0)
            ax.set_xlim(p[0], p[-1])
            ax.tick_params(direction="in", top=True, right=True, length=2.4)
            ax.set_box_aspect(1)

        allre = np.concatenate([d[br][f"cont_{k}"].real
                                for br in ("short", "long") for k in ("s", "r")])
        pad = 0.08 * (allre.max() - allre.min())
        ax_re.set_ylim(allre.min() - pad, allre.max() + pad)
        ax_re.set_ylabel(r"$\mathrm{Re}\,\theta$  (rad)")
        ax_im.set_ylabel(r"$\mathrm{Im}\,\theta$  (rad)")
        ax_rs.set_ylabel(r"$\|\mathbf{F}\|$")

        ax_re.text(0.035, 0.93, rf"$I={I_Wcm2/1e14:.1f}\times10^{{14}}$",
                   transform=ax_re.transAxes, va="top", ha="left", fontsize=7.0,
                   bbox=dict(fc="white", ec="0.75", lw=0.4, pad=1.6))
        ax_rs.text(p_c, ax_rs.get_ylim()[0], r"$p_c$",
                   ha="center", va="bottom", fontsize=6.4, color=C_PC)

        if row == 0:
            ax_re.set_title(r"real parts")
            ax_im.set_title(r"imaginary parts")
            ax_rs.set_title(r"network residual")

    for row in axes:
        for ax in row:
            ax.set_xlabel(r"$p_\parallel$  (a.u.)")

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
    fig.savefig(f"{OUT}/pair_a_intensities.png", dpi=300)
    fig.savefig(f"{OUT}/pair_a_intensities.pdf")
    if FIGDIR:
        fig.savefig(f"{FIGDIR}/pair_a_intensities.png", dpi=300)
    print(f"wrote {OUT}/pair_a_intensities.{{png,pdf}}"
          + (f" and {FIGDIR}/pair_a_intensities.png" if FIGDIR else ""))


if __name__ == "__main__":
    main()
