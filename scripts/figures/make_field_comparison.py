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
from classical_solvers import bichromatic as B
from pinn_common import PINN, train, train_bichrom

torch.set_default_dtype(torch.float64)
OUT = out("comparison")
GT = out("ground_truth")
BICHROM = out("bichromatic")
FIGDIR = figure_dir()

I_P = 0.5
OMEGA = 0.057
N_SEEDS = 5
P_LO_FRAC, P_HI_FRAC = 0.17, 1.16

# published seed-0 values for the monochromatic I=1.0e14 row of the pair A
# figure, used as a regression check that nothing has drifted
PUBLISHED_MONO = {
    "short": dict(loss=5.43e-05, res=3.43e-03, err=5.61e-04),
    "long":  dict(loss=3.62e-05, res=3.00e-03, err=3.55e-04),
}

mpl.rcParams.update({
    "font.size": 7.2, "axes.labelsize": 7.6, "axes.titlesize": 8.0,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "mathtext.fontset": "cm", "font.family": "serif",
})
C_S = "#2d5591"
C_R = "#b23a48"
C_PC = "#6e6e6e"


def box(vals, pad=3.0, floor=1e-3):
    lo, hi = vals.min(0), vals.max(0)
    c = 0.5 * (lo + hi)
    h = np.maximum(0.5 * (hi - lo) * pad, floor)
    return list(c), list(h)


# ------------------------------------------------------------------ mono ----
def run_mono(I_Wcm2=1.0e14, epochs=2500):
    """Replicates make_pair_a_intensities.py's I=1.0e14 row exactly, then adds seeds."""
    E0 = np.sqrt(I_Wcm2 / 3.5094e16)
    C.E_0 = E0
    C.A_0 = E0 / OMEGA
    C.U_P = E0 ** 2 / (4 * OMEGA ** 2)
    A0 = C.A_0
    p_cut = np.sqrt(20 * C.U_P)
    scale = p_cut / 2.0948
    v0 = [4.96186, 0.96739, 9.26187, -0.01340, 2.197280 * scale, 0.017697 * scale]
    sol, ok = C.caustic_point(v0, 0.0)
    p_c = sol[4] if ok else 1.05 * p_cut
    p_grid = np.linspace(P_LO_FRAC * p_c, P_HI_FRAC * p_c, 220)

    seeds = C.find_orbits(p_grid[0], 0.0)
    ss = sorted(seeds, key=lambda x: x[2] - x[0])
    sA, sB = ss[1], ss[2]
    xa, xb = sA.copy(), sB.copy()
    XA, XB = [], []
    for p in p_grid:
        ya, oa = C.newton(xa, p, 0.0)
        yb, ob = C.newton(xb, p, 0.0)
        if oa:
            xa = ya
        if ob:
            xb = yb
        XA.append(xa.copy())
        XB.append(xb.copy())
    XA, XB = np.array(XA), np.array(XB)

    p_t = torch.tensor(p_grid)
    pp0 = torch.zeros_like(p_t)
    z = ((p_t - p_grid.mean()) / p_grid.std()).unsqueeze(1)

    out = {"p": p_grid, "p_c": p_c}
    print(f"monochromatic I={I_Wcm2:.1e}: p_c = {p_c:.6f}")
    for tag, Xg in (("short", XA), ("long", XB)):
        c, w = box(Xg, pad=3.0)
        cont_s = Xg[:, 0] + 1j * Xg[:, 1]
        cont_r = Xg[:, 2] + 1j * Xg[:, 3]
        S = []
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            m = PINN(c, w, input_dim=1)
            loss = train(m, z, p_t, pp0, A0, I_P, epochs)
            with torch.no_grad():
                ths, thr = m(z)
            ths, thr = ths.numpy(), thr.numpy()
            res = np.array([np.linalg.norm(
                C.F_real([a.real, a.imag, b.real, b.imag], p, 0.0))
                for a, b, p in zip(ths, thr, p_grid)])
            err = np.abs(ths - cont_s)
            S.append(dict(loss=loss, ths=ths, thr=thr, res=res, err=err))
            flag = ""
            if seed == 0:
                pub = PUBLISHED_MONO[tag]
                ok_l = abs(loss - pub["loss"]) / pub["loss"] < 0.02
                ok_r = abs(np.median(res) - pub["res"]) / pub["res"] < 0.02
                ok_e = abs(np.median(err) - pub["err"]) / pub["err"] < 0.02
                flag = ("   [seed 0 reproduces published values]" if (ok_l and ok_r and ok_e)
                        else "   [WARNING: seed 0 DIFFERS from published values]")
            print(f"  {tag:5s} seed {seed}: loss {loss:.3e}  "
                  f"median |F| {np.median(res):.3e}  "
                  f"median |d theta_s| {np.median(err):.3e}{flag}")
        out[tag] = dict(seeds=S, cont_s=cont_s, cont_r=cont_r)
    save_case("mono", out, epochs)


# --------------------------------------------------------------- phistar ----
def run_phistar(epochs=9000):
    g = np.load(f"{GT}/gt_phistar_pair.npz")
    pg, XA, XB = g["p_par"], g["XA"], g["XB"]
    phi = float(g["phi_star"])
    p_c = float(g["pc"])
    B.RATIO = 0.3
    A_1, A_2 = B.A_1, B.A_2()

    p_t = torch.tensor(pg)
    pp0 = torch.zeros_like(p_t)
    z = ((p_t - pg.mean()) / pg.std()).unsqueeze(1)

    out = {"p": pg, "p_c": p_c}
    print(f"phi* = {phi:.6f} rad: p_c = {p_c:.6f}")
    for tag, Xg in (("short", XA), ("long", XB)):
        lo = Xg[:, :4].min(0)
        hi = Xg[:, :4].max(0)
        c = list(0.5 * (lo + hi))
        w = list(np.maximum(0.5 * (hi - lo) * 3.0, 0.02))
        cont_s = Xg[:, 0] + 1j * Xg[:, 1]
        cont_r = Xg[:, 2] + 1j * Xg[:, 3]
        S = []
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            m = PINN(c, w, input_dim=1)
            loss = train_bichrom(m, z, p_t, pp0, A_1, A_2, phi, I_P, epochs,
                                 patience=500)
            with torch.no_grad():
                ths, thr = m(z)
            ths, thr = ths.numpy(), thr.numpy()
            res = np.array([np.linalg.norm(
                B.F_real([a.real, a.imag, b.real, b.imag], p, 0.0, phi))
                for a, b, p in zip(ths, thr, pg)])
            err = np.abs(ths - cont_s)
            S.append(dict(loss=loss, ths=ths, thr=thr, res=res, err=err))
            print(f"  {tag:5s} seed {seed}: loss {loss:.3e}  "
                  f"median |F| {np.median(res):.3e}  "
                  f"median |d theta_s| {np.median(err):.3e}")
        out[tag] = dict(seeds=S, cont_s=cont_s, cont_r=cont_r)
    save_case("phistar", out, epochs)


def save_case(name, out, epochs):
    d = dict(p=out["p"], p_c=out["p_c"], epochs=epochs, n_seeds=N_SEEDS)
    for tag in ("short", "long"):
        d[f"{tag}_cont_s"] = out[tag]["cont_s"]
        d[f"{tag}_cont_r"] = out[tag]["cont_r"]
        for i, s in enumerate(out[tag]["seeds"]):
            d[f"{tag}_ths_{i}"] = s["ths"]
            d[f"{tag}_thr_{i}"] = s["thr"]
            d[f"{tag}_res_{i}"] = s["res"]
            d[f"{tag}_err_{i}"] = s["err"]
        d[f"{tag}_loss"] = np.array([s["loss"] for s in out[tag]["seeds"]])
    np.savez(f"{OUT}/field_{name}.npz", **d)
    print(f"wrote {OUT}/field_{name}.npz")


# ------------------------------------------------------------------ figure --
# phi* is deliberately absent: it is an engineered degeneracy, reported on its
# own terms in Appendix D from the polish-test pipeline, not as a validation
# case alongside the two physical fields.
CASES = [
    ("mono", "monochromatic"),
    ("bichrom", r"bichromatic, $\phi=0.6\pi$"),
]


def load_case(name):
    if name == "bichrom":
        # produced by make_bichrom_phase_data.py; phi = 0.6 pi is the middle of the
        # three swept phases and the one quoted in the comparison table
        g = np.load(f"{BICHROM}/bichrom_0.6pi_pinn.npz")
        n = int(g["n_seeds"])
        d = dict(p=g["p_par"], p_c=float(g["pc_re"]), n=n,
                 epochs=int(g["epochs"]))
    else:
        g = np.load(f"{OUT}/field_{name}.npz")
        n = int(g["n_seeds"])
        d = dict(p=g["p"], p_c=float(g["p_c"]), n=n, epochs=int(g["epochs"]))
    for tag in ("short", "long"):
        d[f"{tag}_cont_s"] = g[f"{tag}_cont_s"]
        d[f"{tag}_cont_r"] = g[f"{tag}_cont_r"]
        d[f"{tag}_ths"] = np.array([g[f"{tag}_ths_{i}"] for i in range(n)])
        d[f"{tag}_thr"] = np.array([g[f"{tag}_thr_{i}"] for i in range(n)])
        d[f"{tag}_res"] = np.array([g[f"{tag}_res_{i}"] for i in range(n)])
        d[f"{tag}_err"] = np.array([g[f"{tag}_err_{i}"] for i in range(n)])
    return d


def make_figure():
    data = {n: load_case(n) for n, _ in CASES}
    fig, axes = plt.subplots(len(CASES), 3,
                             figsize=(6.95, 2.35 * len(CASES) + 0.5))

    for row, (name, label) in enumerate(CASES):
        d = data[name]
        p, p_c = d["p"], d["p_c"]
        ax_re, ax_im, ax_rs = axes[row]

        for tag, ls in (("short", "-"), ("long", "--")):
            # representative seed = median by median trajectory error
            med_err = np.median(d[f"{tag}_err"], axis=1)
            rep = int(np.argsort(med_err)[len(med_err) // 2])
            cs, cr = d[f"{tag}_cont_s"], d[f"{tag}_cont_r"]
            ps, pr = d[f"{tag}_ths"][rep], d[f"{tag}_thr"][rep]
            sl = slice(4, None, max(1, len(p) // 22))

            ax_re.plot(p, cs.real, ls, color=C_S, lw=1.0)
            ax_re.plot(p, cr.real, ls, color=C_R, lw=1.0)
            ax_re.plot(p[sl], ps.real[sl], "o", ms=1.9, mfc="none", mec=C_S, mew=0.55)
            ax_re.plot(p[sl], pr.real[sl], "o", ms=1.9, mfc="none", mec=C_R, mew=0.55)

            ax_im.plot(p, cs.imag, ls, color=C_S, lw=1.0)
            ax_im.plot(p, cr.imag, ls, color=C_R, lw=1.0)
            ax_im.plot(p[sl], ps.imag[sl], "o", ms=1.9, mfc="none", mec=C_S, mew=0.55)
            ax_im.plot(p[sl], pr.imag[sl], "o", ms=1.9, mfc="none", mec=C_R, mew=0.55)

            R = d[f"{tag}_res"]
            ax_rs.semilogy(p, np.median(R, axis=0), ls, color="#3f3f46", lw=0.95)
            ax_rs.fill_between(p, R.min(axis=0), R.max(axis=0),
                               color="#3f3f46", alpha=0.16, lw=0)

        for ax in (ax_re, ax_im, ax_rs):
            ax.axvline(p_c, color=C_PC, ls=":", lw=0.8, zorder=0)
            ax.set_xlim(p[0], p[-1])
            ax.tick_params(direction="in", top=True, right=True, length=2.4)
            ax.set_box_aspect(1)
            ax.set_xlabel(r"$p_\parallel$  (a.u.)")

        allre = np.concatenate([d[f"{t}_cont_{k}"].real
                                for t in ("short", "long") for k in ("s", "r")])
        pad = 0.08 * (allre.max() - allre.min())
        ax_re.set_ylim(allre.min() - pad, allre.max() + pad)
        ax_re.set_ylabel(r"$\mathrm{Re}\,\theta$  (rad)")
        ax_im.set_ylabel(r"$\mathrm{Im}\,\theta$  (rad)")
        ax_rs.set_ylabel(r"$\|\mathbf{F}\|$")
        ax_re.text(0.035, 0.93, label, transform=ax_re.transAxes, va="top",
                   ha="left", fontsize=6.6,
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
    fig.savefig(f"{OUT}/field_comparison.png", dpi=300)
    if FIGDIR:
        fig.savefig(f"{FIGDIR}/field_comparison.png", dpi=300)
    print(f"wrote field_comparison.png")
    plt.close(fig)
    summary(data)


def summary(data):
    print("\n" + "=" * 78)
    print("SUMMARY: median over momentum, then across 5 seeds [min, max]")
    print("=" * 78)
    print(f"{'case':<26} {'branch':<6} {'physics residual ||F||':<30} "
          f"{'trajectory error |d theta_s|'}")
    rows = []
    for name, label in CASES:
        d = data[name]
        for tag in ("short", "long"):
            mr = np.median(d[f"{tag}_res"], axis=1)
            me = np.median(d[f"{tag}_err"], axis=1)
            print(f"{label if tag=='short' else '':<26} {tag:<6} "
                  f"{np.median(mr):.2e} [{mr.min():.2e}, {mr.max():.2e}]   "
                  f"{np.median(me):.2e} [{me.min():.2e}, {me.max():.2e}]")
            rows.append((label, tag, np.median(mr), mr.min(), mr.max(),
                         np.median(me), me.min(), me.max()))
    def sci(v, ex):
        """mantissa of v at a fixed exponent ex"""
        return v / 10.0 ** ex

    print("\nLaTeX table body:")
    for label, tag, r, rlo, rhi, e, elo, ehi in rows:
        er = int(np.floor(np.log10(r)))
        ee = int(np.floor(np.log10(e)))
        print(f"    {label if tag=='short' else '':<28} & {tag:<6} & "
              f"${sci(r,er):.2f}\\times10^{{{er}}}$ & "
              f"$[{sci(rlo,er):.2f}, {sci(rhi,er):.2f}]\\times10^{{{er}}}$ & "
              f"${sci(e,ee):.2f}\\times10^{{{ee}}}$ & "
              f"$[{sci(elo,ee):.2f}, {sci(ehi,ee):.2f}]\\times10^{{{ee}}}$ \\\\")
    print("\nepochs per case:", {n: data[n]["epochs"] for n, _ in CASES})


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "mono":
        run_mono()
    elif stage == "phistar":
        run_phistar()
    elif stage == "fig":
        make_figure()
