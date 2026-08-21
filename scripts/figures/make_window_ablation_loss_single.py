"""Single-column loss-curve figure for the window half-width ablation.

Same 4-config (baseline/1x/3x/10x) x 5-seed ablation, same seeds and same
training setup as make_window_ablation.py, so the numbers are identical to
the ones already quoted in Table 3 -- only the plotting differs:

  * left panel only (the per-config physics-loss curves); the right-hand
    per-residual diagnostic panel is not produced here,
  * sized for a single column rather than the full two-column width,
  * seeds are combined with the *mean* rather than the median.

Also writes window_ablation_hist.npz (the full per-epoch, per-seed loss
histories), which make_window_ablation.py discards -- with that on disk the
figure can be restyled later without retraining.

Writes window_ablation_loss_single.png and window_ablation_hist.npz into
output/windowing/.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out
from pinn_common import PINN

torch.set_default_dtype(torch.float64)
OUT = out("windowing")
GT = out("ground_truth")

OMEGA, E_0, I_P = 0.057, 0.0534, 0.5
A_0 = E_0 / OMEGA
EPOCHS = 1500
N_SEEDS = 5

# single text column of the dissertation is 8.55 cm ~= 3.37 in
FIGSIZE = (3.37, 2.6)


def A_t(th):
    return A_0 * torch.cos(th)


def residual_pieces(th_s, th_r, p):
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
    p_np, X = gt["p_par"][::3], gt["X"][::3]

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

    hists = {name: [] for name, _, _ in configs}
    for seed in range(N_SEEDS):
        for name, w, unbounded in configs:
            _, _, hist = train(centre, w, z, p, unbounded, seed)
            hists[name].append(hist)
        print(f"seed {seed} done", flush=True)

    np.savez(f"{OUT}/window_ablation_hist.npz",
             **{f"hist_{n}": np.stack(hists[n]) for n in hists})

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7})

    colors = {"baseline": "#d05070", "1x": "#452a7a", "3x": "#241040", "10x": "#9b7fd4"}
    labels = {"baseline": "unbounded", "1x": r"$1\times$",
              "3x": r"$3\times$", "10x": r"$10\times$"}

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name in ("baseline", "1x", "3x", "10x"):
        H = np.stack(hists[name])[:, :, 0]
        mean = H.mean(axis=0)
        ep = np.arange(H.shape[1])
        ax.semilogy(ep, mean, color=colors[name], lw=1.3, label=labels[name])
        ax.fill_between(ep, H.min(axis=0), H.max(axis=0),
                        color=colors[name], alpha=0.15, lw=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("physics loss")
    ax.legend(frameon=False, loc="upper right", handlelength=1.4, borderaxespad=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/window_ablation_loss_single.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    print("\nfinal-epoch physics loss, mean over seeds [min, max]:")
    for name in ("baseline", "1x", "3x", "10x"):
        H = np.stack(hists[name])[:, :, 0]
        f = H[:, -1]
        print(f"  {name:9s} {f.mean():.3e}  [{f.min():.2e}, {f.max():.2e}]")

    Hr = np.stack(hists["1x"])
    print(f"\nreturn-residual (diagnostic) range, 1x config: "
          f"{Hr[:, :, 2].min():.3e} to {Hr[:, :, 2].max():.3e}")
    print("\nfigure written")


if __name__ == "__main__":
    main()
