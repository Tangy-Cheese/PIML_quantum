"""
Generates output/pair_a_pmd.npz and pairA_pmd_{gt,pinn}.png -- the actual
photoelectron momentum distribution for pair A, built from
pair_a_theta_2d.npz using the amplitude machinery ALREADY IN continuation.py
(action, hess_det, pair_amplitude / uniform_sweep_stable). No new physics;
this is the assembly step that was missing.

pair_amplitude() operates on one p_par sweep at fixed p_perp at a time (it
was written and tested for exactly that: a 1D phi*-style sweep through a
coalescence). Building the 2D PMD means calling it once per p_perp row,
which matches how make_gt_2d.py itself is structured.

Scope: pair A only (short + long, the coalescing pair). This is a
deliberate choice -- pair_amplitude()'s Stokes/Airy blending is built for a
two-orbit coalescing pair specifically, not an arbitrary number of orbits,
so extending to all four pairs is future work, not a missing piece here.

Caveat inherited from continuation.py's own orbit_amplitude() docstring:
"V_pk and d set to unity for a first pass" -- the dipole and scattering
matrix elements are not yet physical, only the phase (exp(iS)) and the
Gaussian/Airy normalisation are. State this next to the figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import continuation as C

OUT = out("future_work_pmd")


def build_pmd(S, R, K, p_par, p_perp):
    """S, R, K: (N_PERP, N_PAR) complex arrays for ONE branch (short or
    long). Returns nothing -- this just packages (theta_s, theta_r) into
    the 4-vector form pair_amplitude expects, per grid point.
    """
    n_perp, n_par = S.shape
    out = np.empty((n_perp, n_par), dtype=object)
    for i in range(n_perp):
        for j in range(n_par):
            out[i, j] = np.array([S[i, j].real, S[i, j].imag,
                                  R[i, j].real, R[i, j].imag])
    return out


def pmd_for_branch_pair(s_short, r_short, s_long, r_long, p_par, p_perp):
    """Coherent pair amplitude, row by row in p_perp, using
    continuation.pair_amplitude (Airy uniform near the fold, filtered
    ordinary sum away from it, blended -- already built and tested).
    """
    n_perp, n_par = s_short.shape
    M = np.zeros((n_perp, n_par), dtype=complex)
    for i, pp in enumerate(p_perp):
        xa_list = [np.array([s_short[i, j].real, s_short[i, j].imag,
                             r_short[i, j].real, r_short[i, j].imag])
                  for j in range(n_par)]
        xb_list = [np.array([s_long[i, j].real, s_long[i, j].imag,
                             r_long[i, j].real, r_long[i, j].imag])
                  for j in range(n_par)]
        m_row, zeta = C.pair_amplitude(xa_list, xb_list, p_par, pp)
        M[i] = m_row
    return M


def main():
    d = np.load(f"{OUT}/pair_a_theta_2d.npz")
    p_par, p_perp = d["p_par"], d["p_perp"]

    print("assembling GT (continuation) PMD ...")
    M_gt = pmd_for_branch_pair(d["cont_s_short"], d["cont_r_short"],
                               d["cont_s_long"], d["cont_r_long"],
                               p_par, p_perp)

    print("assembling PINN PMD ...")
    M_pinn = pmd_for_branch_pair(d["pinn_s_short"], d["pinn_r_short"],
                                 d["pinn_s_long"], d["pinn_r_long"],
                                 p_par, p_perp)

    I_gt, I_pinn = np.abs(M_gt) ** 2, np.abs(M_pinn) ** 2
    norm = I_gt.max()
    I_gt, I_pinn = I_gt / norm, I_pinn / norm

    finite = np.isfinite(I_gt) & (I_gt > 1e-8)
    rel = np.where(finite, np.abs(I_pinn - I_gt) / (I_gt + 1e-12), np.nan)
    print(f"\nPMD agreement (points > 1e-8 of peak): "
          f"median relative diff {np.nanmedian(rel):.3e}")

    # mirror to p_perp < 0 for display (the usual PMD presentation convention)
    pf = np.concatenate([-p_perp[::-1][:-1], p_perp])
    purple = LinearSegmentedColormap.from_list(
        "p", ["#0b0416", "#241040", "#452a7a", "#6d4bb0", "#9b7fd4", "#c9b6ec", "#ece4f8"])

    np.savez(f"{OUT}/pair_a_pmd.npz", p_par=p_par, p_perp=p_perp,
             I_gt=I_gt, I_pinn=I_pinn, M_gt=M_gt, M_pinn=M_pinn)

    for tag, I, label in (("gt", I_gt, "continuation"), ("pinn", I_pinn, "PINN")):
        full = np.vstack([I[::-1][:-1], I])
        plt.figure(figsize=(7.2, 5))
        plt.pcolormesh(p_par, pf, np.log10(full + 1e-12),
                       shading="gouraud", cmap=purple, vmin=-6, vmax=0)
        plt.colorbar(label=r"$\log_{10}|M_p|^2$")
        plt.xlabel(r"$p_\parallel$ (a.u.)")
        plt.ylabel(r"$p_\perp$ (a.u.)")
        plt.title(f"Pair A rescattered PMD ({label})\n"
                  r"(unit $d$, $V_{pk}$ -- phase + Gaussian/Airy prefactor only)",
                  fontsize=10)
        plt.savefig(f"{OUT}/pairA_pmd_{tag}.png", dpi=180, bbox_inches="tight")
        plt.close()

    print("\nfigures written: pair_a_pmd_gt.png, pair_a_pmd_pinn.png")


if __name__ == "__main__":
    main()
