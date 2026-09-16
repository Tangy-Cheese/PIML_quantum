import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams["font.size"] = 16
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
from classical_solvers import monochromatic as C

OUT = out("diagnostics")


def main():
    seeds = C.find_orbits(0.40, 0.0)
    short, long_ = seeds[1].copy(), seeds[2].copy()

    p_scan = np.linspace(0.35, 2.60, 400)
    cond_s, cond_l, sep, det_s = [], [], [], []
    a, b = short.copy(), long_.copy()
    for p in p_scan:
        ra, oa = C.newton(a, p, 0.0)
        rb, ob = C.newton(b, p, 0.0)
        if oa:
            a = ra
        if ob:
            b = rb
        cond_s.append(C.cond_fixed_p(a, p, 0.0))
        cond_l.append(C.cond_fixed_p(b, p, 0.0))
        sep.append(abs((a[0] + 1j * a[1]) - (b[0] + 1j * b[1])))
        det_s.append(abs(C.hess_det(a[0] + 1j * a[1], a[2] + 1j * a[3], p, 0.0)))

    v0 = [4.96186, 0.96739, 9.26187, -0.01340, 2.197280, 0.017697]
    locus = C.caustic_locus(v0, np.linspace(0.0, 1.35, 60))
    pp_c = np.array([t[0] for t in locus])
    pc = np.array([t[1] for t in locus])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(p_scan, cond_s, label="short branch")
    ax.plot(p_scan, cond_l, label="long branch")
    ax.axvline(pc[0].real, ls="--", c="k", lw=1, label="$10.007U_p+0.538I_p$")
    ax.axvline(np.sqrt(20 * C.U_P), ls=":", c="gray", lw=1, label="$10U_p$")
    ax.set_xlabel(r"$p_\parallel$ (a.u.)")
    ax.set_ylabel("condition number")
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="y", style="plain")
    ax.legend()
    plt.savefig(f"{OUT}/newton_conditioning.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    plt.semilogy(p_scan, sep)
    plt.axvline(pc[0].real, ls="--", c="k", lw=1)
    plt.xlabel(r"$p_\parallel$ (a.u.)")
    plt.ylabel(r"$|\theta_s^{(a)} - \theta_s^{(b)}|$")
    plt.title("Separation of the coalescing branch pair")
    plt.savefig(f"{OUT}/branch_separation.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    plt.semilogy(p_scan, det_s)
    plt.axvline(pc[0].real, ls="--", c="k", lw=1)
    plt.xlabel(r"$p_\parallel$ (a.u.)")
    plt.ylabel(r"$|\det \mathcal{H}|$")
    plt.title("Hessian determinant along the short branch")
    plt.savefig(f"{OUT}/hessian_det.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 5))
    plt.plot(pc.real, pp_c, lw=2, label="caustic ring")
    plt.plot(pc.real, -pp_c, lw=2, c="C0")
    plt.plot(-pc.real, pp_c, lw=2, c="C0")
    plt.plot(-pc.real, -pp_c, lw=2, c="C0")
    plt.xlabel(r"Re$(p_\parallel)$ (a.u.)")
    plt.ylabel(r"$p_\perp$ (a.u.)")
    plt.title("Caustic locus in the momentum plane")
    plt.legend()
    plt.savefig(f"{OUT}/caustic_ring.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(pp_c, pc.imag, lw=2)
    plt.xlabel(r"$p_\perp$ (a.u.)")
    plt.ylabel(r"Im$(p_c)$ (a.u.)")
    plt.title("Distance of the caustic off the real momentum plane")
    plt.savefig(f"{OUT}/caustic_imag.png", dpi=150, bbox_inches="tight")
    plt.close()

    path, tang, cond = C.continue_branch(seeds[1], 0.40, 0.0, p_lo=0.3, p_hi=2.6,
                                         ds=2e-3, ds_max=1e-2)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(path[:, 0], path[:, 1], ".-", ms=3)
    plt.xlabel(r"Re$(\theta_s)$")
    plt.ylabel(r"Im$(\theta_s)$")
    plt.title("Continuation path of the ionisation phase")
    plt.savefig(f"{OUT}/branch_path.png", dpi=150, bbox_inches="tight")
    plt.close()

    np.savez(f"{OUT}/caustic_conditioning.npz", p_scan=p_scan, cond_short=cond_s,
             cond_long=cond_l, separation=sep, det_short=det_s,
             caustic_pperp=pp_c, caustic_p=pc, path=path, cond_path=cond)

    print("caustic at p_perp = 0: p_c = %.6f %+.6fj" % (pc[0].real, pc[0].imag))
    print("ring extent in Re(p_c): %.4f .. %.4f" % (pc.real.min(), pc.real.max()))
    print("Im(p_c) range: %.5f .. %.5f" % (pc.imag.min(), pc.imag.max()))
    print("max condition number on the real axis: %.3e" % max(cond_s))
    print("min |det H| on the real axis: %.4e" % min(det_s))
    print("min branch separation: %.4e" % min(sep))


if __name__ == "__main__":
    main()
