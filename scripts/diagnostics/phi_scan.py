import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
from classical_solvers import bichromatic as B

OUT = out("diagnostics")


def caustics_at_phi(phi, p_lo=0.3, p_hi=3.0, n_scan=260, max_families=6):
    """Locate caustics near the real momentum axis at fixed relative phase.

    Orbit families are tracked across p_par; wherever |det H| shows a local
    minimum the discriminant system is solved to place the caustic exactly in
    complex p_par.
    """
    B.PHI = phi
    seeds = B.find_orbits(p_lo + 0.05, 0.0, phi)[:max_families]
    p_scan = np.linspace(p_lo, p_hi, n_scan)

    out = []
    for x0 in seeds:
        x = x0.copy()
        dets, xs = [], []
        for p in p_scan:
            y, ok = B.newton(x, p, 0.0, phi)
            if ok:
                x = y
            dets.append(abs(B.hess_det(x[0] + 1j * x[1], x[2] + 1j * x[3], p, 0.0, phi)))
            xs.append(x.copy())
        dets = np.array(dets)

        for j in range(1, len(dets) - 1):
            if dets[j] < dets[j - 1] and dets[j] < dets[j + 1]:
                v0 = list(xs[j]) + [p_scan[j], 0.0]
                sol, ok = B.caustic_point(v0, 0.0, phi)
                if not ok:
                    continue
                pc = sol[4] + 1j * sol[5]
                if not (p_lo - 0.5 < pc.real < p_hi + 0.5):
                    continue
                if any(abs(pc - q) < 1e-4 for q, _ in out):
                    continue
                out.append((pc, sol))
    return out


def main():
    B.RATIO = 0.3
    phis = np.linspace(0.0, 2 * np.pi, 25)

    rows = []
    print("  phi/pi   n_caustics   min |Im p_c|   at Re p_c")
    for phi in phis:
        cs = caustics_at_phi(phi)
        if not cs:
            print(f"  {phi/np.pi:6.3f}       0            --")
            continue
        ims = np.array([abs(pc.imag) for pc, _ in cs])
        j = np.argmin(ims)
        pc = cs[j][0]
        rows.append((phi, len(cs), ims[j], pc.real))
        print(f"  {phi/np.pi:6.3f}       {len(cs)}         {ims[j]:.6f}     {pc.real:8.4f}")

    R = np.array(rows)
    np.savez(f"{OUT}/phi_scan.npz", phi=R[:, 0], n=R[:, 1], im_pc=R[:, 2],
             re_pc=R[:, 3])

    j = np.argmin(R[:, 2])
    print(f"\nclosest approach over phi: |Im p_c| = {R[j,2]:.6f} at phi = {R[j,0]/np.pi:.3f} pi")
    print(f"monochromatic reference   : |Im p_c| = 0.017697")

    plt.figure(figsize=(7.5, 4.5))
    plt.semilogy(R[:, 0] / np.pi, R[:, 2], "o-", label="bichromatic")
    plt.axhline(0.017697, ls="--", c="k", lw=1, label="monochromatic")
    plt.xlabel(r"relative phase $\phi/\pi$")
    plt.ylabel(r"$|\mathrm{Im}\,p_c|$ (a.u.)")
    plt.legend()
    plt.title(r"Closest caustic approach to real momentum vs $\phi$")
    plt.savefig(f"{OUT}/phi_scan_imag.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(R[:, 0] / np.pi, R[:, 3], "o-")
    plt.xlabel(r"relative phase $\phi/\pi$")
    plt.ylabel(r"$\mathrm{Re}\,p_c$ (a.u.)")
    plt.title(r"Caustic momentum vs relative phase")
    plt.savefig(f"{OUT}/phi_scan_real.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(R[:, 0] / np.pi, R[:, 1], "o-")
    plt.xlabel(r"relative phase $\phi/\pi$")
    plt.ylabel("number of caustics located")
    plt.title(r"Caustic count vs relative phase")
    plt.savefig(f"{OUT}/phi_scan_count.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
