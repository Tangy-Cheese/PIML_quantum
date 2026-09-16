import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
from pinn_common import PINN

torch.set_default_dtype(torch.float64)
OUT = out("windowing")
GT = out("ground_truth")

OMEGA, E_0, I_P = 0.057, 0.0534, 0.5
A_0 = E_0 / OMEGA
EPOCHS = 1500
N_SEEDS = 5


def A_t(th):
    return A_0 * torch.cos(th)


def residual_pieces(th_s, th_r, p):
    """Tunnelling and rescattering residuals (the two that enter the loss),
    plus the return residual as a diagnostic only (not optimised)."""
    k = -A_0 * (torch.sin(th_r) - torch.sin(th_s)) / (th_r - th_s)
    R1 = 0.5 * (k + A_t(th_s)) ** 2 + I_P
    R3 = 0.5 * (p + A_t(th_r)) ** 2 - 0.5 * (k + A_t(th_r)) ** 2
    R2_diag = k * (th_r - th_s) + A_0 * (torch.sin(th_r) - torch.sin(th_s))
    return R1, R2_diag, R3


def train(centres, halfwidths, z, p, unbounded, seed):
    torch.manual_seed(seed)
    m = PINN(centres, halfwidths, input_dim=1, unbounded=unbounded)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=250)
    hist, best, best_state = [], np.inf, None
    for _ in range(EPOCHS):
        opt.zero_grad()
        th_s, th_r = m(z)
        R1, R2_diag, R3 = residual_pieces(th_s, th_r, p)
        l1, l3 = (R1.abs() ** 2).mean(), (R3.abs() ** 2).mean()
        L = l1 + l3
        L.backward()
        opt.step()
        sched.step(L.item())
        l2 = (R2_diag.abs() ** 2).mean()
        hist.append([L.item(), l1.item(), l2.item(), l3.item()])
        if L.item() < best:
            best = L.item()
            best_state = {n: v.detach().clone() for n, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    return best, m, np.array(hist)


def main():
    gt = np.load(f"{GT}/gt_pair_a_short.npz")
    p_np, X, k_gt = gt["p_par"][::3], gt["X"][::3], gt["k"][::3]
    ths_gt = X[:, 0] + 1j * X[:, 1]

    p = torch.tensor(p_np)
    z = ((p - p_np.mean()) / p_np.std()).unsqueeze(1)

    lo = np.array([X[:, 0].min(), X[:, 1].min(), X[:, 2].min(), X[:, 3].min()])
    hi = np.array([X[:, 0].max(), X[:, 1].max(), X[:, 2].max(), X[:, 3].max()])
    centre = list(0.5 * (lo + hi))
    half = 0.5 * (hi - lo)

    configs = [
        ("baseline", [1.0] * 4, True),
        ("1x", list(np.maximum(half * 1.0, 0.05)), False),
        ("3x", list(np.maximum(half * 3.0, 0.05)), False),
        ("10x", list(np.maximum(half * 10.0, 0.05)), False),
    ]

    results = {name: {"loss": [], "err": [], "hist": []} for name, _, _ in configs}
    for seed in range(N_SEEDS):
        for name, w, unbounded in configs:
            best, m, hist = train(centre, w, z, p, unbounded, seed)
            with torch.no_grad():
                ths, _ = m(z)
            err = np.median(np.abs(ths.numpy() - ths_gt))
            results[name]["loss"].append(best)
            results[name]["err"].append(err)
            results[name]["hist"].append(hist)
        print(f"seed {seed} done")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 4.8))
    colors = {"baseline": "#d05070", "1x": "#452a7a", "3x": "#241040", "10x": "#9b7fd4"}
    for name in ("baseline", "1x", "3x", "10x"):
        H = np.stack(results[name]["hist"])[:, :, 0]
        med = np.median(H, axis=0)
        lo_, hi_ = H.min(axis=0), H.max(axis=0)
        ep = np.arange(H.shape[1])
        ax1.semilogy(ep, med, color=colors[name], lw=1.6, label=f"{name} (median)")
        ax1.fill_between(ep, lo_, hi_, color=colors[name], alpha=0.15)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("physics loss")
    ax1.legend()

    Hr = np.stack(results["1x"]["hist"])
    eq_names = ["tunnelling", "return (diagnostic only)", "rescattering"]
    eq_colors = ["#d05070", "#452a7a", "#9b7fd4"]
    ep = np.arange(Hr.shape[1])
    for i, (name, c) in enumerate(zip(eq_names, eq_colors)):
        comp = Hr[:, :, i + 1]
        med = np.median(comp, axis=0)
        ax2.semilogy(ep, med, color=c, lw=1.6, label=name)
        ax2.fill_between(ep, comp.min(axis=0), comp.max(axis=0), color=c, alpha=0.15)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("residual MSE")
    ax2.legend()

    plt.savefig(f"{OUT}/window_ablation_loss.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("averaged loss-curve figure written")

    print(f"\n=== {N_SEEDS} seeds, {EPOCHS} epochs, {len(p_np)} points ===")
    print(f"{'config':10s} {'median loss':>12s} {'loss range':>24s} {'median err':>12s} {'err range':>22s}")
    for name in results:
        L = np.array(results[name]["loss"])
        E = np.array(results[name]["err"])
        print(f"{name:10s} {np.median(L):12.3e} "
              f"[{L.min():.2e}, {L.max():.2e}]   "
              f"{np.median(E):12.3e} [{E.min():.2e}, {E.max():.2e}]")

    base = np.array(results["1x"]["loss"])
    print(f"\npaired ratio vs 1x windowing (median of per-seed ratios):")
    for name in ("3x", "10x", "baseline"):
        r = base / np.array(results[name]["loss"])
        wins = (r > 1).sum()
        print(f"  {name:10s}: {np.median(r):8.2f}x   (1x better in {N_SEEDS-wins}/{N_SEEDS} seeds)")

    print(f"\nreturn-residual (diagnostic) range across all epochs/seeds, 1x config: "
          f"{Hr[:, :, 2].min():.3e} to {Hr[:, :, 2].max():.3e}")

    np.savez(f"{OUT}/window_ablation.npz",
             **{f"loss_{n}": np.array(results[n]["loss"]) for n in results},
             **{f"err_{n}": np.array(results[n]["err"]) for n in results})

    order = ["baseline", "1x", "3x", "10x"]
    data = [results[n]["loss"] for n in order]
    plt.figure(figsize=(7.5, 4.5))
    plt.boxplot(data, tick_labels=order, widths=0.6)
    for i, d in enumerate(data):
        plt.plot(np.full(len(d), i + 1) + np.random.uniform(-0.1, 0.1, len(d)),
                 d, "o", ms=4, alpha=0.6, color="#452a7a")
    plt.yscale("log")
    plt.ylabel("final physics loss")
    plt.title(f"Window width, {N_SEEDS} seeds each")
    plt.savefig(f"{OUT}/window_ablation_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    data_e = [results[n]["err"] for n in order]
    plt.boxplot(data_e, tick_labels=order, widths=0.6)
    for i, d in enumerate(data_e):
        plt.plot(np.full(len(d), i + 1) + np.random.uniform(-0.1, 0.1, len(d)),
                 d, "o", ms=4, alpha=0.6, color="#241040")
    plt.yscale("log")
    plt.ylabel(r"median $|\Delta\theta_s|$ vs continuation")
    plt.title(f"Window width, accuracy, {N_SEEDS} seeds each")
    plt.savefig(f"{OUT}/window_ablation_error.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("\nfigures written")


if __name__ == "__main__":
    main()
