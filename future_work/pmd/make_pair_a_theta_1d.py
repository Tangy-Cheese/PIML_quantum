"""
Generates pair_a_theta_1d.npz -- the 1D counterpart of make_pair_a_theta_2d.py:
continuation reference AND PINN predictions for (theta_s, theta_r) vs p_par
at p_perp = 0, for pair A (family 1 = short, family 2 = long), plus the
PINN's own physics residual ||F|| (via C.F_real on the PINN's own output,
NOT the training loss curve and NOT compared against continuation).

Reuses all_families.py's PINN class, loss_fn, train, and ground_truth
unchanged -- same 1D architecture and training recipe already validated
across all 8 families, just kept here instead of retrained inline.

Also writes the three requested 1D figures directly (Re vs p, Im vs p,
residual vs p) so this doubles as the script behind those plots, and saves
pair_a_theta_1d.npz in the exact schema plot_pair_a_complex.py already
expects (its load_data() docstring names these keys).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy"))
from paths import out, figure_dir
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy"))
import continuation as C
from all_families import PINN, train, ground_truth

torch.set_default_dtype(torch.float64)
OUT = out("future_work_pmd")

C_ION = "#452A7A"
C_RET = "#C8506E"
PC_RE = 2.197280


def run_branch(fam, p_grid):
    P, X = ground_truth(fam, p_grid)
    k = np.array([C.k_of(x[0] + 1j * x[1], x[2] + 1j * x[3]) for x in X])

    p = torch.tensor(P)
    z = ((p - P.mean()) / P.std()).unsqueeze(1)
    lo = np.array([X[:, 0].min(), X[:, 1].min(), X[:, 2].min(), X[:, 3].min(),
                   k.real.min(), k.imag.min()])
    hi = np.array([X[:, 0].max(), X[:, 1].max(), X[:, 2].max(), X[:, 3].max(),
                   k.real.max(), k.imag.max()])
    c = list(0.5 * (lo + hi))
    w = list(np.maximum(0.5 * (hi - lo) * 3.0, 0.05))

    torch.manual_seed(0)
    m = PINN(c, w)
    loss = train(m, z, p)
    with torch.no_grad():
        ths, thr, kk = m(z)
    ths, thr = ths.numpy(), thr.numpy()

    # PINN's own physics residual -- plug its OWN (th_s, th_r) into F_real,
    # not the error against continuation.
    res = np.array([
        np.linalg.norm(C.F_real([s.real, s.imag, r.real, r.imag], pp, 0.0))
        for s, r, pp in zip(ths, thr, P)
    ])

    cont_s = X[:, 0] + 1j * X[:, 1]
    cont_r = X[:, 2] + 1j * X[:, 3]
    print(f"family {fam}: n={len(P):3d}  loss={loss:.3e}  "
          f"median|dth_s|={np.median(np.abs(ths - cont_s)):.3e}  "
          f"median||F||={np.median(res):.3e}")

    return dict(p=P, cont_s=cont_s, cont_r=cont_r, pinn_s=ths, pinn_r=thr, res=res)


def main():
    p_grid = np.linspace(0.35, 2.55, 200)

    print("training 1D PINN, family 1 (short branch of pair A) ...")
    short = run_branch(1, p_grid)
    print("training 1D PINN, family 2 (long branch of pair A) ...")
    long_ = run_branch(2, p_grid)

    # Common grid: intersect the two branches' surviving continuation points
    # (sample_path can drop different indices near the fold for each branch)
    # so short/long line up point-for-point for plotting and saving.
    p, i_s, i_l = np.intersect1d(short["p"], long_["p"], return_indices=True)

    def trim(d, idx):
        return {k: v[idx] for k, v in d.items() if k != "p"}

    S, L = trim(short, i_s), trim(long_, i_l)

    np.savez(
        f"{OUT}/pair_a_theta_1d.npz",
        p=p,
        cont_s_short=S["cont_s"], cont_r_short=S["cont_r"],
        cont_s_long=L["cont_s"], cont_r_long=L["cont_r"],
        pinn_s_short=S["pinn_s"], pinn_r_short=S["pinn_r"],
        pinn_s_long=L["pinn_s"], pinn_r_long=L["pinn_r"],
        res_short=S["res"], res_long=L["res"],
    )
    print(f"\nwrote {OUT}/pair_a_theta_1d.npz")

    # ---------------- Re(theta) vs p ----------------
    fig, (axs, axr) = plt.subplots(2, 1, figsize=(7.2, 7.5), sharex=True)
    axs.plot(p, S["cont_s"].real, color=C_ION, lw=2.0, label="continuation, short")
    axs.plot(p, L["cont_s"].real, color=C_ION, lw=2.0, ls="--", label="continuation, long")
    axs.plot(p, S["pinn_s"].real, color=C_ION, lw=1.3, ls=(0, (1, 1)), label="PINN, short")
    axs.plot(p, L["pinn_s"].real, color=C_ION, lw=1.3, ls=(0, (4, 2)), label="PINN, long")
    axs.set_ylabel(r"$\mathrm{Re}\,\theta_s$")
    axs.axvline(PC_RE, color=C_RET, lw=1.0, ls=":")
    axs.legend(fontsize=7.5, frameon=False, ncol=2)
    axs.set_title("Pair A: ionisation phase")

    axr.plot(p, S["cont_r"].real, color=C_RET, lw=2.0, label="continuation, short")
    axr.plot(p, L["cont_r"].real, color=C_RET, lw=2.0, ls="--", label="continuation, long")
    axr.plot(p, S["pinn_r"].real, color=C_RET, lw=1.3, ls=(0, (1, 1)), label="PINN, short")
    axr.plot(p, L["pinn_r"].real, color=C_RET, lw=1.3, ls=(0, (4, 2)), label="PINN, long")
    axr.set_ylabel(r"$\mathrm{Re}\,\theta_r$")
    axr.set_xlabel(r"$p_\parallel$ (a.u.)")
    axr.axvline(PC_RE, color=C_RET, lw=1.0, ls=":")
    axr.legend(fontsize=7.5, frameon=False, ncol=2)
    axr.set_title("Pair A: return phase")

    fig.tight_layout()
    fig.savefig(f"{OUT}/pair_a_theta_real.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Im(theta) vs p ----------------
    fig, (axs, axr) = plt.subplots(2, 1, figsize=(7.2, 7.5), sharex=True)
    axs.plot(p, S["cont_s"].imag, color=C_ION, lw=2.0, label="continuation, short")
    axs.plot(p, L["cont_s"].imag, color=C_ION, lw=2.0, ls="--", label="continuation, long")
    axs.plot(p, S["pinn_s"].imag, color=C_ION, lw=1.3, ls=(0, (1, 1)), label="PINN, short")
    axs.plot(p, L["pinn_s"].imag, color=C_ION, lw=1.3, ls=(0, (4, 2)), label="PINN, long")
    axs.set_ylabel(r"$\mathrm{Im}\,\theta_s$")
    axs.axvline(PC_RE, color=C_RET, lw=1.0, ls=":")
    axs.legend(fontsize=7.5, frameon=False, ncol=2)
    axs.set_title("Pair A: ionisation phase")

    axr.plot(p, S["cont_r"].imag, color=C_RET, lw=2.0, label="continuation, short")
    axr.plot(p, L["cont_r"].imag, color=C_RET, lw=2.0, ls="--", label="continuation, long")
    axr.plot(p, S["pinn_r"].imag, color=C_RET, lw=1.3, ls=(0, (1, 1)), label="PINN, short")
    axr.plot(p, L["pinn_r"].imag, color=C_RET, lw=1.3, ls=(0, (4, 2)), label="PINN, long")
    axr.set_ylabel(r"$\mathrm{Im}\,\theta_r$")
    axr.set_xlabel(r"$p_\parallel$ (a.u.)")
    axr.axvline(PC_RE, color=C_RET, lw=1.0, ls=":")
    axr.legend(fontsize=7.5, frameon=False, ncol=2)
    axr.set_title("Pair A: return phase")

    fig.tight_layout()
    fig.savefig(f"{OUT}/pair_a_theta_imag.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---------------- PINN physics residual vs p ----------------
    plt.figure(figsize=(7.2, 4.5))
    plt.semilogy(p, S["res"], color="#452A7A", lw=1.6, label="short")
    plt.semilogy(p, L["res"], color="#9B7FD4", lw=1.6, label="long")
    plt.axvline(PC_RE, color=C_RET, lw=1.0, ls=":", label=r"$\mathrm{Re}\,p_c$")
    plt.xlabel(r"$p_\parallel$ (a.u.)")
    plt.ylabel(r"PINN physics residual $\|F\|$")
    plt.title("Pair A: PINN's own residual (not compared to continuation)")
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/pair_a_residual.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("\nfigures written: pair_a_theta_real.png, pair_a_theta_imag.png, "
          "pair_a_residual.png")


if __name__ == "__main__":
    main()
