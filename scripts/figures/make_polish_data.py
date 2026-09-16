import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
import continuation as C
import bichromatic as B
from pinn_common import PINN, train, train_bichrom

torch.set_default_dtype(torch.float64)
OUT = out("polish")
GT = out("ground_truth")

I_P = 0.5
EPOCHS = 4000
N_SEEDS = 5
N_STEPS = 4          # damped Newton steps, as in the original hybrid test
DAMPING = 1.0
# below this the polished error is at the double-precision floor for a
# quantity of order 1 rad, and improvement ratios stop being meaningful
FLOOR = 1e-14

# monochromatic pair A caustic (same starting vector as make_pair_a_intensities.py)
V0_MONO = [4.96186, 0.96739, 9.26187, -0.01340, 2.197280, 0.017697]


class MonoField:
    """Adapter: monochromatic field, continuation.py."""
    name = "mono"
    label = "monochromatic, ordinary pair A"

    def __init__(self):
        self.A_0 = C.A_0
        self.phi = None

    def newton(self, x, p, **kw):
        return C.newton(x, p, 0.0, **kw)

    def F_real(self, x, p):
        return C.F_real(x, p, 0.0)

    def J_real(self, x, p):
        return C.J_real(x, p, 0.0)[:, :4]

    def train(self, m, z, p_t, pp0):
        return train(m, z, p_t, pp0, self.A_0, I_P, EPOCHS, patience=400)


class PhiStarField:
    """Adapter: bichromatic field at phi*, bichromatic.py."""
    name = "phistar"
    label = r"bichromatic at $\phi^\ast$"

    def __init__(self, phi):
        B.RATIO = 0.3
        self.phi = phi
        self.A_1, self.A_2 = B.A_1, B.A_2()

    def newton(self, x, p, **kw):
        return B.newton(x, p, 0.0, self.phi, **kw)

    def F_real(self, x, p):
        return B.F_real(x, p, 0.0, self.phi)

    def J_real(self, x, p):
        return B.J_real(x, p, 0.0, self.phi)[:, :4]

    def train(self, m, z, p_t, pp0):
        return train_bichrom(m, z, p_t, pp0, self.A_1, self.A_2, self.phi,
                             I_P, EPOCHS, patience=400)


def build_mono_fine():
    """Log grid approaching the monochromatic pair A caustic, both branches.
    Mirrors make_gt_phistar.py's fine-grid construction."""
    sol, ok = C.caustic_point(V0_MONO, 0.0)
    if not ok:
        raise RuntimeError("monochromatic caustic solve failed")
    pc = sol[4]
    print(f"monochromatic pair A caustic: p_c = {pc:.6f} {sol[5]:+.4e}j")
    print(f"  kappa at caustic = {C.cond_fixed_p(sol[:4], pc, 0.0):.4e}")

    # start both branches well below the fold and track inward
    p0 = 0.55 * pc
    seeds = C.find_orbits(p0, 0.0)
    th_c = sol[0] + 1j * sol[1]
    # track every family in, then keep the two that approach the caustic
    tracks = []
    d = np.logspace(np.log10(1e-7), np.log10(pc - p0), 300)
    pgl = np.sort(pc - d)
    for s in seeds:
        x = np.array(s, float)
        row, okall = [], True
        for p in pgl:
            y, o = C.newton(x, p, 0.0, tol=1e-13, max_iter=200)
            if o:
                x = y
            row.append(x.copy())
        tracks.append(np.array(row))
    dist = [abs((t[-1, 0] + 1j * t[-1, 1]) - th_c) for t in tracks]
    order = np.argsort(dist)
    iA, iB = order[0], order[1]
    print(f"  pair A branches identified: families {iA}, {iB} "
          f"(|d theta_s| to caustic {dist[iA]:.3e}, {dist[iB]:.3e}; "
          f"next {dist[order[2]]:.3e})")
    LA, LB = tracks[iA], tracks[iB]
    s = np.abs((LA[:, 0] + 1j * LA[:, 1]) - (LB[:, 0] + 1j * LB[:, 1]))
    keep = s > 1e-12
    LA, LB, LP, s = LA[keep], LB[keep], pgl[keep], s[keep]
    print(f"  {len(LP)} points, closest p_c - p = {(pc-LP).min():.2e}, "
          f"separation there = {s[-1]:.4e}")
    q = np.polyfit(np.log(pc - LP), np.log(s), 1)
    print(f"  separation ~ (p_c - p)^{q[0]:.4f}  "
          f"(a true fold gives 0.5; this caustic is off-axis so it flattens)")
    np.savez(f"{GT}/gt_mono_fine.npz", p_par=LP, XA=LA, XB=LB, pc=pc)
    print(f"  wrote {GT}/gt_mono_fine.npz")


def box(vals, pad=3.0, floor=1e-3):
    lo, hi = vals.min(0), vals.max(0)
    c = 0.5 * (lo + hi)
    h = np.maximum(0.5 * (hi - lo) * pad, floor)
    return list(c), list(h)


def polish(field, x0, p):
    x = np.array(x0, float)
    for _ in range(N_STEPS):
        F = field.F_real(x, p)
        if not np.isfinite(F).all():
            break
        J = field.J_real(x, p)
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(dx).all():
            break
        x = x + DAMPING * dx
    return x


def run(case):
    if case == "mono":
        g = np.load(f"{GT}/gt_mono_fine.npz")
        field = MonoField()
    else:
        g = np.load(f"{GT}/gt_phistar_fine.npz")
        field = PhiStarField(float(g["phi"]))

    P, XA, XB, pc = g["p_par"], g["XA"], g["XB"], float(g["pc"])
    sA = XA[:, 0] + 1j * XA[:, 1]
    sB = XB[:, 0] + 1j * XB[:, 1]
    gap = np.abs(sA - sB)
    dist = pc - P

    p_t = torch.tensor(P)
    pp0 = torch.zeros_like(p_t)
    z = ((p_t - P.mean()) / P.std()).unsqueeze(1)

    print(f"\n=== case: {case} ({field.label}) ===")
    print(f"p_c = {pc:.6f},  {len(P)} points, "
          f"p_c - p in [{dist.min():.2e}, {dist.max():.2e}]")

    store = {}
    for tag, Xg, s_gt in (("A", XA, sA), ("B", XB, sB)):
        c, w = box(Xg, pad=3.0, floor=1e-3)
        s_other = sB if tag == "A" else sA
        err_pinn = np.zeros((N_SEEDS, len(P)))
        err_pol = np.zeros((N_SEEDS, len(P)))
        res_pinn = np.zeros((N_SEEDS, len(P)))
        correct = np.zeros((N_SEEDS, len(P)), dtype=bool)
        losses = np.zeros(N_SEEDS)
        # theta_s of the network itself, kept so that the PINN's own branch
        # separation can be reconstructed against the reference (Appendix D)
        ths_all = np.zeros((N_SEEDS, len(P)), dtype=complex)

        print(f"\n--- branch {tag}, {N_SEEDS} seeds ---")
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            m = PINN(c, w, input_dim=1)
            losses[seed] = field.train(m, z, p_t, pp0)
            with torch.no_grad():
                ths, thr = m(z)
            ths, thr = ths.numpy(), thr.numpy()
            ths_all[seed] = ths
            err_pinn[seed] = np.abs(ths - s_gt)
            res_pinn[seed] = [np.linalg.norm(
                field.F_real([a.real, a.imag, b.real, b.imag], p))
                for a, b, p in zip(ths, thr, P)]
            for j in range(len(P)):
                x0 = [ths[j].real, ths[j].imag, thr[j].real, thr[j].imag]
                xp = polish(field, x0, P[j])
                th_p = xp[0] + 1j * xp[1]
                err_pol[seed, j] = abs(th_p - s_gt[j])
                correct[seed, j] = abs(th_p - s_gt[j]) < abs(th_p - s_other[j])
            print(f"  seed {seed}: loss {losses[seed]:.2e}  "
                  f"median err PINN {np.median(err_pinn[seed]):.3e} -> "
                  f"polished {np.median(err_pol[seed]):.3e}  "
                  f"wrong branch {100*(1-correct[seed].mean()):.1f}%")

        store[tag] = dict(err_pinn=err_pinn, err_pol=err_pol,
                          res_pinn=res_pinn, correct=correct, losses=losses,
                          ths=ths_all)

    # summary: ratio of PINN error to polished error, across seeds
    print(f"\n=== {case}: improvement factor (median over seeds) ===")
    print(f"{'p_c - p':>10} {'gap':>10} {'PINN err':>11} {'polished':>11} "
          f"{'ratio':>10} {'branch ok':>10}")
    idx = np.unique(np.concatenate([
        np.linspace(0, len(P) - 1, 12).astype(int), [len(P) - 1]]))
    A = store["A"]
    for j in idx:
        ep = np.median(A["err_pinn"][:, j])
        eq = np.median(A["err_pol"][:, j])
        # polished error routinely reaches the double-precision floor, where a
        # ratio is meaningless; floor it at EPS and say so rather than print
        # a spurious 1e12x
        if eq <= FLOOR:
            rtxt = "-> mach eps"
        else:
            rtxt = f"{ep/eq:9.1f}x"
        print(f"{dist[j]:10.2e} {gap[j]:10.2e} {ep:11.3e} {eq:11.3e} "
              f"{rtxt:>11} {100*A['correct'][:,j].mean():9.0f}%")

    for tag in ("A", "B"):
        s = store[tag]
        frac_wrong = 100 * (1 - s["correct"].mean())
        print(f"\nbranch {tag}: polish lands on the WRONG branch at "
              f"{frac_wrong:.2f}% of (seed, momentum) pairs")
        # closest-point behaviour, the honest worst case
        ep = np.median(s["err_pinn"][:, -1])
        eq = np.median(s["err_pol"][:, -1])
        tail = "(polished at machine precision)" if eq <= FLOOR else \
               f"({ep/eq:.1f}x)"
        print(f"  at closest approach (p_c - p = {dist[-1]:.2e}): "
              f"PINN {ep:.3e} -> polished {eq:.3e}  {tail}")
        med_ratio = np.median(s["err_pinn"] / np.maximum(s["err_pol"], FLOOR))
        frac_floor = 100 * (s["err_pol"] <= FLOOR).mean()
        print(f"  median improvement (polished error floored at {FLOOR:.0e}): "
              f"{med_ratio:.3g}x")
        print(f"  polished error at the double-precision floor in "
              f"{frac_floor:.1f}% of (seed, momentum) pairs")

    np.savez(f"{OUT}/polish_{case}.npz", p_par=P, pc=pc, gap=gap, dist=dist,
             n_seeds=N_SEEDS, n_steps=N_STEPS, epochs=EPOCHS,
             **{f"{t}_{k}": store[t][k] for t in store for k in store[t]})
    print(f"\nwrote {OUT}/polish_{case}.npz")


if __name__ == "__main__":
    case = sys.argv[1]
    if case == "mono":
        import os
        if not os.path.exists(f"{GT}/gt_mono_fine.npz"):
            build_mono_fine()
    run(case)
