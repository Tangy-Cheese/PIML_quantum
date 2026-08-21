"""Locate the critical bichromatic phase phi* and generate ground truth there.

phi* is the relative phase at which the caustic crosses the real momentum axis,
so that Im(p_c) = 0 and the fixed-momentum Jacobian becomes genuinely singular.

Produces:
    critical_phi.npz      phi*, p_c, and the coalescing solution
    gt_phistar_pair.npz   both branches on a uniform grid
    gt_phistar_fine.npz   both branches on a log grid approaching p_c

Runtime: about 5 minutes.
"""

import numpy as np
from scipy.optimize import brentq

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
import bichromatic as B

OUT = out("ground_truth")


def caustics_at_phi(phi, p_lo=0.3, p_hi=3.0, n_scan=260, max_families=6):
    """Locate caustics near the real momentum axis at fixed relative phase."""
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
            dets.append(abs(B.hess_det(x[0] + 1j * x[1], x[2] + 1j * x[3],
                                       p, 0.0, phi)))
            xs.append(x.copy())
        dets = np.array(dets)

        for j in range(1, len(dets) - 1):
            if dets[j] < dets[j - 1] and dets[j] < dets[j + 1]:
                sol, ok = B.caustic_point(list(xs[j]) + [p_scan[j], 0.0], 0.0, phi)
                if not ok:
                    continue
                pc = sol[4] + 1j * sol[5]
                if not (p_lo - 0.5 < pc.real < p_hi + 0.5):
                    continue
                if any(abs(pc - q) < 1e-4 for q, _ in out):
                    continue
                out.append((pc, sol))
    return out


def im_pc(phi):
    cs = caustics_at_phi(phi)
    if not cs:
        return np.nan
    j = int(np.argmin([abs(pc.imag) for pc, _ in cs]))
    return cs[j][0].imag


def main():
    B.RATIO = 0.3

    print("bracketing the zero crossing of Im(p_c) in phi ...")
    phi_star = brentq(im_pc, 0.2475 * np.pi, 0.2500 * np.pi, xtol=1e-12)
    cs = caustics_at_phi(phi_star)
    j = int(np.argmin([abs(pc.imag) for pc, _ in cs]))
    pc, sol = cs[j]

    print(f"  phi*   = {phi_star:.9f} rad = {phi_star/np.pi:.7f} pi")
    print(f"  p_c    = {pc.real:.8f} {pc.imag:+.3e}j")
    print(f"  kappa  = {B.cond_fixed_p(sol[:4], pc.real, 0.0, phi_star):.3e}")
    np.savez(f"{OUT}/critical_phi.npz", phi_star=phi_star,
             pc_re=pc.real, pc_im=pc.imag, sol=np.array(sol))

    B.PHI = phi_star
    print("\ntracing both branches of the coalescing pair ...")
    p0 = 1.60
    path0, _ = B.continue_branch(np.array(sol[:4]), pc.real, 0.0,
                                 p_lo=1.55, p_hi=2.45, phi=phi_star,
                                 ds=1e-3, ds_max=5e-3)
    x0 = B.sample_path(path0, np.array([p0]), 0.0, phi_star)[0]

    path, tang = B.continue_branch(x0, p0, 0.0, p_lo=1.50, p_hi=2.45,
                                   phi=phi_star, ds=5e-4, ds_max=2e-3)
    dp = tang[:, 4]
    folds = np.where(np.sign(dp[:-1]) * np.sign(dp[1:]) < 0)[0]
    jf = folds[0]
    b1, b2 = path[:jf + 1], path[jf:]

    p_grid = np.linspace(1.60, 1.9660, 260)
    sA = B.sample_path(b1, p_grid, 0.0, phi_star)
    sB = B.sample_path(b2, p_grid, 0.0, phi_star)
    common = sorted(set(sA) & set(sB))
    XA = np.array([sA[j] for j in common])
    XB = np.array([sB[j] for j in common])
    pg = p_grid[common]
    sep = np.abs((XA[:, 0] + 1j * XA[:, 1]) - (XB[:, 0] + 1j * XB[:, 1]))
    kA = np.array([B.k_of(x[0] + 1j * x[1], x[2] + 1j * x[3], phi_star) for x in XA])
    kB = np.array([B.k_of(x[0] + 1j * x[1], x[2] + 1j * x[3], phi_star) for x in XB])
    print(f"  uniform grid: {len(pg)} points")
    np.savez(f"{OUT}/gt_phistar_pair.npz", p_par=pg, XA=XA, XB=XB,
             kA=kA, kB=kB, sep=sep, phi_star=phi_star, pc=pc.real)

    print("\nlog grid approaching the caustic ...")
    d = np.logspace(np.log10(1e-7), np.log10(0.36), 300)
    pgl = np.sort(pc.real - d)
    a, b = XA[0].copy(), XB[0].copy()
    LA, LB, LP = [], [], []
    for p in pgl:
        ra, oa = B.newton(a, p, 0.0, phi_star, tol=1e-13, max_iter=200)
        rb, ob = B.newton(b, p, 0.0, phi_star, tol=1e-13, max_iter=200)
        if oa and ob and np.linalg.norm(ra - rb) > 1e-12:
            a, b = ra, rb
            LA.append(a.copy())
            LB.append(b.copy())
            LP.append(p)
    LA, LB, LP = np.array(LA), np.array(LB), np.array(LP)
    s = np.abs((LA[:, 0] + 1j * LA[:, 1]) - (LB[:, 0] + 1j * LB[:, 1]))
    q = np.polyfit(np.log(pc.real - LP), np.log(s), 1)
    print(f"  {len(LP)} points, closest p_c - p = {(pc.real-LP).min():.2e}")
    print(f"  separation ~ (p_c - p)^{q[0]:.4f}   (fold predicts 0.5)")
    np.savez(f"{OUT}/gt_phistar_fine.npz", p_par=LP, XA=LA, XB=LB,
             pc=pc.real, phi=phi_star)


if __name__ == "__main__":
    main()
