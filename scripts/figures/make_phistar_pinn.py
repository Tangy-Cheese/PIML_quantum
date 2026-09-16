import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
import bichromatic as B
from pinn_common import PINN

torch.set_default_dtype(torch.float64)
OUT = out("phistar")
GT = out("ground_truth")

I_P = 0.5
A_1 = 0.0534 / 0.057
EPOCHS = 9000


def windows(X, pad):
    lo = np.array([X[:, 0].min(), X[:, 1].min(), X[:, 2].min(), X[:, 3].min()])
    hi = np.array([X[:, 0].max(), X[:, 1].max(), X[:, 2].max(), X[:, 3].max()])
    return list(0.5 * (lo + hi)), list(np.maximum(0.5 * (hi - lo) * pad, 0.02))


def main():
    gt = np.load(f"{GT}/gt_phistar_pair.npz")
    pg, XA, XB = gt["p_par"], gt["XA"], gt["XB"]
    phi = float(gt["phi_star"])
    pc = float(gt["pc"])

    p = torch.tensor(pg)
    pp0 = torch.zeros_like(p)
    z = ((p - pg.mean()) / pg.std()).unsqueeze(1)

    res = {}
    for tag, X in (("A", XA), ("B", XB)):
        c, w = windows(X, 3.0)
        torch.manual_seed(0)
        m = PINN(c, w, input_dim=1)

        # bichromatic residuals need the phase phi baked in, so wrap
        # directly here rather than forcing pinn_common's fixed-field
        # residual signature.
        def residuals(th_s, th_r, p_par):
            A1 = A_1
            A2 = 0.3 * A1
            A_s = A1 * torch.cos(th_s) + A2 * torch.cos(2 * th_s + phi)
            A_r = A1 * torch.cos(th_r) + A2 * torch.cos(2 * th_r + phi)
            intA_s = A1 * torch.sin(th_s) + 0.5 * A2 * torch.sin(2 * th_s + phi)
            intA_r = A1 * torch.sin(th_r) + 0.5 * A2 * torch.sin(2 * th_r + phi)
            k = -(intA_r - intA_s) / (th_r - th_s)
            R1 = 0.5 * (k + A_s) ** 2 + I_P
            R2 = 0.5 * (p_par + A_r) ** 2 - 0.5 * (k + A_r) ** 2
            return R1, R2

        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=500)
        hist, best, best_state = [], np.inf, None
        for _ in range(EPOCHS):
            opt.zero_grad()
            th_s, th_r = m(z)
            R1, R2 = residuals(th_s, th_r, p)
            L = (R1.abs() ** 2).mean() + (R2.abs() ** 2).mean()
            L.backward()
            opt.step()
            sch.step(L.item())
            hist.append(L.item())
            if L.item() < best:
                best = L.item()
                best_state = {n: v.detach().clone() for n, v in m.state_dict().items()}
        m.load_state_dict(best_state)

        with torch.no_grad():
            ths, thr = m(z)
        ths_np, thr_np = ths.numpy(), thr.numpy()
        err_s = np.abs(ths_np - (X[:, 0] + 1j * X[:, 1]))
        pinn_res = np.array([
            np.linalg.norm(B.F_real([s.real, s.imag, r.real, r.imag], pp, 0.0, phi))
            for s, r, pp in zip(ths_np, thr_np, pg)])
        res[tag] = dict(loss=best, err_s=err_s, ths=ths_np, hist=np.array(hist),
                        pinn_res=pinn_res)
        print(f"=== branch {tag} at phi* (k-eliminated) ===")
        print(f"  final physics loss : {best:.3e}")
        print(f"  median |d th_s|    : {np.median(err_s):.3e}")
        print(f"  max    |d th_s|    : {err_s.max():.3e}")
        print(f"  median ||F||_PINN  : {np.median(pinn_res):.3e}")
        print(f"  ||F||_PINN at p_c  : {pinn_res[-1]:.3e}")

    near = pc - pg < 5e-3
    print(f"\n=== within 5e-3 of the caustic ({near.sum()} points) ===")
    for tag in ("A", "B"):
        e = res[tag]["err_s"][near]
        r = res[tag]["pinn_res"][near]
        print(f"  branch {tag}: median |d th_s| {np.median(e):.3e}, max {e.max():.3e}  "
              f"median ||F|| {np.median(r):.3e}")

    sep_gt = gt["sep"]
    sep_nn = np.abs(res["A"]["ths"] - res["B"]["ths"])
    ok = sep_gt > 0
    print(f"\nbranch separation, PINN vs continuation:")
    print(f"  median relative error : {np.median(np.abs(sep_nn[ok]-sep_gt[ok])/sep_gt[ok]):.3e}")
    print(f"  at closest approach   : PINN {sep_nn[-1]:.4e} vs truth {sep_gt[-1]:.4e}")
    print(f"  p_c - p at closest approach point: {pc - pg[-1]:.4e}")

    np.savez(f"{OUT}/phistar_pinn.npz", p_par=pg, err_A=res["A"]["err_s"],
             err_B=res["B"]["err_s"], sep_gt=sep_gt, sep_nn=sep_nn,
             loss_A=res["A"]["hist"], loss_B=res["B"]["hist"], pc=pc,
             res_A=res["A"]["pinn_res"], res_B=res["B"]["pinn_res"])

    plt.figure(figsize=(7.5, 4.5))
    plt.semilogy(pc - pg, res["A"]["err_s"], label="branch A")
    plt.semilogy(pc - pg, res["B"]["err_s"], label="branch B")
    plt.gca().invert_xaxis()
    plt.xscale("log")
    plt.xlabel(r"$p_c - p_\parallel$ (a.u.)")
    plt.ylabel(r"$|\Delta\theta_s|$")
    plt.legend()
    plt.title(r"PINN accuracy approaching the caustic at $\phi^*$")
    plt.savefig(f"{OUT}/phistar_error.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    plt.loglog(pc - pg, sep_gt, label="continuation")
    plt.loglog(pc - pg, sep_nn, "--", label="PINN")
    plt.loglog(pc - pg, sep_gt[0] * np.sqrt((pc - pg) / (pc - pg[0])), ":",
               c="k", label=r"$\sqrt{p_c-p}$")
    plt.gca().invert_xaxis()
    plt.xlabel(r"$p_c - p_\parallel$ (a.u.)")
    plt.ylabel(r"branch separation $|\theta_s^A-\theta_s^B|$")
    plt.legend()
    plt.title(r"Coalescence resolved by the PINN at $\phi^*$")
    plt.savefig(f"{OUT}/phistar_separation.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    plt.semilogy(res["A"]["hist"], label="branch A")
    plt.semilogy(res["B"]["hist"], label="branch B")
    plt.xlabel("epoch")
    plt.ylabel("physics loss")
    plt.legend()
    plt.title(r"Training at the critical phase $\phi^*=0.2489\pi$")
    plt.savefig(f"{OUT}/phistar_loss.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
