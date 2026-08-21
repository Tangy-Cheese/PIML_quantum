"""
Generates output/pair_a_theta_2d.npz -- continuation reference AND PINN
predictions for (theta_s, theta_r, k) across the full (p_par, p_perp) grid,
for pair A (families 1 and 2). This is the 2D counterpart of
make_pair_a_theta_1d.py, needed as input to make_pair_a_pmd.py.

error_maps.py already trains a 2D PINN per family on this same grid, but
only keeps the error map (err_s = |theta_s_PINN - theta_s_GT|) and discards
theta_r, k, and theta_s itself. This script reuses error_maps.py's PINN
class, loss, and training loop unchanged, and keeps everything.
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy"))
from paths import out, figure_dir
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy"))
import continuation as C
from error_maps import PINN, train, ground_truth_2d, P_LO, P_HI, N_PAR, PERP_MAX, N_PERP

torch.set_default_dtype(torch.float64)
OUT = out("future_work_pmd")


def run_family(fam, p_grid, perp_grid):
    Xg = ground_truth_2d(fam, p_grid, perp_grid)          # (N_PERP, N_PAR, 4)
    Xflat = Xg.reshape(-1, 4)
    k_gt = np.array([C.k_of(x[0] + 1j * x[1], x[2] + 1j * x[3]) for x in Xflat])

    PP, PT = np.meshgrid(p_grid, perp_grid, indexing="xy")
    fp, ft = PP.ravel(), PT.ravel()
    z = torch.tensor(np.stack([(fp - fp.mean()) / fp.std(),
                               (ft - ft.mean()) / (ft.std() + 1e-12)], axis=1))
    p_t, pp_t = torch.tensor(fp), torch.tensor(ft)

    lo = np.array([Xflat[:, 0].min(), Xflat[:, 1].min(), Xflat[:, 2].min(),
                   Xflat[:, 3].min(), k_gt.real.min(), k_gt.imag.min()])
    hi = np.array([Xflat[:, 0].max(), Xflat[:, 1].max(), Xflat[:, 2].max(),
                   Xflat[:, 3].max(), k_gt.real.max(), k_gt.imag.max()])
    c = list(0.5 * (lo + hi))
    w = list(np.maximum(0.5 * (hi - lo) * 3.0, 0.05))

    torch.manual_seed(0)
    m = PINN(c, w)
    loss = train(m, z, p_t, pp_t)
    with torch.no_grad():
        ths, thr, kk = m(z)
    ths, thr, kk = ths.numpy(), thr.numpy(), kk.numpy()

    err_s = np.abs(ths - (Xflat[:, 0] + 1j * Xflat[:, 1]))
    print(f"family {fam}: loss={loss:.3e}  median|dth_s|={np.median(err_s):.3e}  "
          f"max={err_s.max():.3e}")

    shp = (len(perp_grid), len(p_grid))
    return dict(
        cont_s=(Xflat[:, 0] + 1j * Xflat[:, 1]).reshape(shp),
        cont_r=(Xflat[:, 2] + 1j * Xflat[:, 3]).reshape(shp),
        cont_k=k_gt.reshape(shp),
        pinn_s=ths.reshape(shp), pinn_r=thr.reshape(shp), pinn_k=kk.reshape(shp),
        loss=loss,
    )


def main():
    p_grid = np.linspace(P_LO, P_HI, N_PAR)
    perp_grid = np.linspace(0.0, PERP_MAX, N_PERP)

    print("training 2D PINN, family 1 (short branch of pair A) ...")
    short = run_family(1, p_grid, perp_grid)
    print("training 2D PINN, family 2 (long branch of pair A) ...")
    long_ = run_family(2, p_grid, perp_grid)

    np.savez(
        f"{OUT}/pair_a_theta_2d.npz",
        p_par=p_grid, p_perp=perp_grid,
        cont_s_short=short["cont_s"], cont_r_short=short["cont_r"], cont_k_short=short["cont_k"],
        cont_s_long=long_["cont_s"], cont_r_long=long_["cont_r"], cont_k_long=long_["cont_k"],
        pinn_s_short=short["pinn_s"], pinn_r_short=short["pinn_r"], pinn_k_short=short["pinn_k"],
        pinn_s_long=long_["pinn_s"], pinn_r_long=long_["pinn_r"], pinn_k_long=long_["pinn_k"],
    )
    print(f"\nwrote {OUT}/pair_a_theta_2d.npz")


if __name__ == "__main__":
    main()
