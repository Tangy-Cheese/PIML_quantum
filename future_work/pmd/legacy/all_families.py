import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
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
A_0 = 0.0534 / 0.057
U_P = 0.0534**2 / (4 * 0.057**2)
SEED_P = 0.40
EPOCHS = 3000


def A_t(th):
    return A_0 * torch.cos(th)


class PINN(nn.Module):
    def __init__(self, centres, halfwidths, hidden=64, depth=3):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
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


def loss_fn(model, z, p):
    th_s, th_r, k = model(z)
    R1 = 0.5 * (k + A_t(th_s)) ** 2 + I_P
    R2 = k * (th_r - th_s) + A_0 * (torch.sin(th_r) - torch.sin(th_s))
    R3 = 0.5 * (p + A_t(th_r)) ** 2 - 0.5 * (k + A_t(th_r)) ** 2
    return (R1.abs() ** 2).mean() + (R2.abs() ** 2).mean() + (R3.abs() ** 2).mean()


def train(model, z, p):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=300)
    best, state = np.inf, None
    for _ in range(EPOCHS):
        opt.zero_grad()
        L = loss_fn(model, z, p)
        L.backward()
        opt.step()
        sch.step(L.item())
        if L.item() < best:
            best = L.item()
            state = {n: v.detach().clone() for n, v in model.state_dict().items()}
    model.load_state_dict(state)
    return best


def ground_truth(family, p_grid):
    seeds = C.find_orbits(SEED_P, 0.0)
    seed = seeds[family]
    samples = {}
    for d in (+1.0, -1.0):
        path, _, _ = C.continue_branch(seed, SEED_P, 0.0,
                                       p_lo=p_grid[0] - 0.02, p_hi=p_grid[-1] + 0.02,
                                       ds=2e-3, ds_max=1e-2, direction=d)
        for j, xs in C.sample_path(path, p_grid, 0.0).items():
            if j not in samples:
                samples[j] = xs[0]
    idx = sorted(samples)
    return p_grid[idx], np.array([samples[j] for j in idx])


def main():
    p_grid = np.linspace(0.35, 2.55, 200)
    results = {}

    for fam in range(1, 9):
        P, X = ground_truth(fam, p_grid)
        if len(P) < 50:
            print(f"family {fam}: only {len(P)} points, skipped")
            continue
        k = np.array([C.k_of(x[0] + 1j * x[1], x[2] + 1j * x[3]) for x in X])
        res = max(np.linalg.norm(C.F_real(x, pp, 0.0)) for x, pp in zip(X, P))

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
        err_s = np.abs(ths.numpy() - (X[:, 0] + 1j * X[:, 1]))
        err_r = np.abs(thr.numpy() - (X[:, 2] + 1j * X[:, 3]))
        fwd = np.abs(kk.numpy() - P).min()

        results[fam] = dict(P=P, err_s=err_s, err_r=err_r, loss=loss,
                            delay=X[:, 2].mean() - X[:, 0].mean(), gt_res=res, fwd=fwd)
        print(f"family {fam}: n={len(P):3d}  delay={results[fam]['delay']:6.2f}  "
              f"|F|gt={res:.1e}  loss={loss:.3e}  "
              f"median|dth_s|={np.median(err_s):.3e}  max={err_s.max():.3e}  "
              f"min|k-p|={fwd:.2f}")

    print("\nsummary over all trained families:")
    med = np.array([np.median(results[f]["err_s"]) for f in results])
    los = np.array([results[f]["loss"] for f in results])
    print(f"  families trained     : {sorted(results)}")
    print(f"  median |d theta_s|   : {np.median(med):.3e}  "
          f"(range {med.min():.3e} to {med.max():.3e})")
    print(f"  physics loss         : {np.median(los):.3e}  "
          f"(range {los.min():.3e} to {los.max():.3e})")

    np.savez(f"{OUT}/all_families_1d.npz",
             **{f"err_{f}": results[f]["err_s"] for f in results},
             **{f"p_{f}": results[f]["P"] for f in results},
             **{f"loss_{f}": results[f]["loss"] for f in results})

    plt.figure(figsize=(7.5, 4.5))
    for f in sorted(results):
        E = results[f]["P"] ** 2 / (2 * U_P)
        plt.semilogy(E, results[f]["err_s"], lw=1,
                     label=f"orbit {f} (delay {results[f]['delay']:.1f})")
    plt.xlabel(r"final energy ($U_p$)")
    plt.ylabel(r"$|\Delta\theta_s|$")
    plt.legend(fontsize=7, ncol=2)
    plt.title("PINN accuracy against continuation, all orbit families")
    plt.savefig(f"{OUT}/all_families_error_1d.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    fams = sorted(results)
    plt.bar([str(f) for f in fams], [np.median(results[f]["err_s"]) for f in fams],
            color="#452a7a")
    plt.yscale("log")
    plt.xlabel("orbit family")
    plt.ylabel(r"median $|\Delta\theta_s|$")
    plt.title("Per-family accuracy")
    plt.savefig(f"{OUT}/all_families_summary_1d.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
