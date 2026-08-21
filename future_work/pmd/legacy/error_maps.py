import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams["font.size"] = 16
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from paths import out, figure_dir
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import continuation as C

torch.set_default_dtype(torch.float64)
OUT = out("future_work_pmd")

I_P = 0.5
E_0 = 0.0534
OMEGA = 0.057
A_0 = E_0 / OMEGA
U_P = E_0**2 / (4 * OMEGA**2)
SEED_P = 0.40
EPOCHS = 2000

P_LO, P_HI, N_PAR = 0.35, 2.55, 90
PERP_MAX, N_PERP = 1.20, 18


def A_t(th):
    return A_0 * torch.cos(th)


class PINN(nn.Module):
    def __init__(self, centres, halfwidths, hidden=64, depth=3):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 6)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("c", torch.tensor(centres))
        self.register_buffer("w", torch.tensor(halfwidths))

    def forward(self, z):
        o = self.c + self.w * torch.tanh(self.net(z))
        return (o[:, 0] + 1j * o[:, 1], o[:, 2] + 1j * o[:, 3],
                o[:, 4] + 1j * o[:, 5])


def loss_fn(model, z, p, pp):
    th_s, th_r, k = model(z)
    R1 = 0.5 * (k + A_t(th_s)) ** 2 + I_P
    R2 = k * (th_r - th_s) + A_0 * (torch.sin(th_r) - torch.sin(th_s))
    R3 = 0.5 * (p + A_t(th_r)) ** 2 + 0.5 * pp ** 2 - 0.5 * (k + A_t(th_r)) ** 2
    return (R1.abs() ** 2).mean() + (R2.abs() ** 2).mean() + (R3.abs() ** 2).mean()


def train(model, z, p, pp):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=250)
    best, state = np.inf, None
    for _ in range(EPOCHS):
        opt.zero_grad()
        L = loss_fn(model, z, p, pp)
        L.backward()
        opt.step()
        sch.step(L.item())
        if L.item() < best:
            best = L.item()
            state = {n: v.detach().clone() for n, v in model.state_dict().items()}
    model.load_state_dict(state)
    return best


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
        k = np.array([C.k_of(x[0] + 1j * x[1], x[2] + 1j * x[3]) for x in Xflat])

        lo = np.array([Xflat[:, 0].min(), Xflat[:, 1].min(), Xflat[:, 2].min(),
                       Xflat[:, 3].min(), k.real.min(), k.imag.min()])
        hi = np.array([Xflat[:, 0].max(), Xflat[:, 1].max(), Xflat[:, 2].max(),
                       Xflat[:, 3].max(), k.real.max(), k.imag.max()])
        c = list(0.5 * (lo + hi))
        w = list(np.maximum(0.5 * (hi - lo) * 3.0, 0.05))

        torch.manual_seed(0)
        m = PINN(c, w)
        loss = train(m, z, p_t, pp_t)
        with torch.no_grad():
            ths, thr, kk = m(z)
        err = np.abs(ths.numpy() - (Xflat[:, 0] + 1j * Xflat[:, 1]))
        err_map = err.reshape(len(perp_grid), len(p_grid))

        results[fam] = dict(err_map=err_map, loss=loss,
                            fwd=np.abs(kk.numpy() - fp).min())
        print(f"family {fam}: loss={loss:.3e}  median|dth_s|={np.median(err):.3e}  "
              f"max={err.max():.3e}  min|k-p|={results[fam]['fwd']:.2f}")

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
