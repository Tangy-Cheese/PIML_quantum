import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
import continuation as C
from pinn_common import PINN, train

torch.set_default_dtype(torch.float64)
OUT = out("families")

I_P = 0.5
E_0 = 0.0534
OMEGA = 0.057
A_0 = E_0 / OMEGA
U_P = E_0**2 / (4 * OMEGA**2)
SEED_P = 0.40
EPOCHS = 2000

P_LO, P_HI, N_PAR = 0.35, 2.55, 140
PERP_MAX, N_PERP = 1.20, 48


def track_family(seed0, p_grid, p_perp):
    y, ok = C.newton(seed0, SEED_P, p_perp)
    seed = y if ok else seed0
    samples = {}
    for d in (+1.0, -1.0):
        path, _, _ = C.continue_branch(seed, SEED_P, p_perp,
                                       p_lo=p_grid[0] - 0.02, p_hi=p_grid[-1] + 0.02,
                                       ds=3e-3, ds_max=1.5e-2, direction=d)
        for j, xs in C.sample_path(path, p_grid, p_perp).items():
            if j not in samples:
                samples[j] = xs[0]
    prev = seed
    row, valid = [], np.zeros(len(p_grid), dtype=bool)
    for j in range(len(p_grid)):
        if j in samples:
            prev = samples[j]
            valid[j] = True
        row.append(prev.copy())
    return row, seed, valid


def ground_truth_2d(family, p_grid, perp_grid):
    seeds = C.find_orbits(SEED_P, 0.0)
    seed = seeds[family]
    X = np.zeros((len(perp_grid), len(p_grid), 4))
    for i, pp in enumerate(perp_grid):
        row, seed, valid = track_family(seed, p_grid, pp)
        X[i] = np.array(row)
    return X


def main():
    p_grid = np.linspace(P_LO, P_HI, N_PAR)
    perp_grid = np.linspace(0.0, PERP_MAX, N_PERP)
    PP, PT = np.meshgrid(p_grid, perp_grid, indexing="xy")
    fp, ft = PP.ravel(), PT.ravel()

    z_np = np.stack([(fp - fp.mean()) / fp.std(),
                     (ft - ft.mean()) / (ft.std() + 1e-12)], axis=1)
    z = torch.tensor(z_np)
    p_t = torch.tensor(fp)
    pp_t = torch.tensor(ft)

    results = {}
    for fam in range(1, 9):
        Xg = ground_truth_2d(fam, p_grid, perp_grid)
        Xflat = Xg.reshape(-1, 4)

        lo = np.array([Xflat[:, 0].min(), Xflat[:, 1].min(), Xflat[:, 2].min(), Xflat[:, 3].min()])
        hi = np.array([Xflat[:, 0].max(), Xflat[:, 1].max(), Xflat[:, 2].max(), Xflat[:, 3].max()])
        c = list(0.5 * (lo + hi))
        w = list(np.maximum(0.5 * (hi - lo) * 3.0, 0.05))

        torch.manual_seed(0)
        m = PINN(c, w, input_dim=2)
        loss = train(m, z, p_t, pp_t, A_0, I_P, EPOCHS)
        with torch.no_grad():
            ths, thr = m(z)
        err = np.abs(ths.numpy() - (Xflat[:, 0] + 1j * Xflat[:, 1]))
        err_map = err.reshape(len(perp_grid), len(p_grid))

        results[fam] = dict(err_map=err_map, loss=loss)
        print(f"family {fam}: loss={loss:.3e}  median|dth_s|={np.median(err):.3e}  "
              f"max={err.max():.3e}")

    np.savez(f"{OUT}/error_maps_2d.npz", p_par=p_grid, p_perp=perp_grid,
             **{f"err_{f}": results[f]["err_map"] for f in results})

    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.2), sharex=True, sharey=True)
    for f, ax in zip(range(1, 9), axes.ravel()):
        em = results[f]["err_map"]
        im = ax.pcolormesh(p_grid, perp_grid, np.log10(em + 1e-16),
                           shading="auto", cmap="viridis", vmin=-5, vmax=-1)
        ax.set_title(f"family {f}", fontsize=13)
    fig.supxlabel(r"$p_\parallel$ (a.u.)")
    fig.supylabel(r"$p_\perp$ (a.u.)")
    fig.colorbar(im, ax=axes, label=r"$\log_{10}|\Delta\theta_s|$", shrink=0.8)
    plt.savefig(f"{OUT}/error_maps_2d.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nsummary:")
    med = [np.median(results[f]["err_map"]) for f in results]
    print(f"  median error across all 8: {np.median(med):.3e}")
    print(f"  range: {min(med):.3e} to {max(med):.3e}")


if __name__ == "__main__":
    main()
