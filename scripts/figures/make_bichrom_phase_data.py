import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
import bichromatic as B
from pinn_common import PINN, train_bichrom

torch.set_default_dtype(torch.float64)
OUT = out("bichromatic")
GT = out("ground_truth")

I_P = 0.5
RATIO = 0.3
EPOCHS = 2500
N_SEEDS = 5
P_LO_FRAC, P_HI_FRAC = 0.17, 1.16
N_POINTS = 220

PHASES = {"0.5pi": 0.5 * np.pi, "0.6pi": 0.6 * np.pi, "0.9pi": 0.9 * np.pi}

# monochromatic pair A caustic, the starting point of every adiabatic track
V0_MONO = [4.96186, 0.96739, 9.26187, -0.01340, 2.197280, 0.017697]


def locate_caustic(phi, verbose=True):
    """Adiabatically continue the monochromatic pair A caustic in the
    harmonic ratio 0 -> RATIO at this fixed phi, so the located caustic is
    provably the descendant of pair A rather than an unrelated one. Repeated
    per phase, since the orbit splitting differs with phi."""
    B.RATIO = 0.0
    v, ok = B.caustic_point(V0_MONO, 0.0, phi)
    if not ok:
        raise RuntimeError("monochromatic caustic solve failed")
    for r in np.linspace(0.0, RATIO, 31)[1:]:
        B.RATIO = float(r)
        v, ok = B.caustic_point(v, 0.0, phi)
        if not ok:
            raise RuntimeError(f"caustic solve failed at ratio={r}, phi={phi}")
    B.RATIO = RATIO
    kappa = B.cond_fixed_p(v[:4], v[4], 0.0, phi)
    if verbose:
        print(f"  p_c = {v[4]:.6f} {v[5]:+.4e}j   |Im p_c| = {abs(v[5]):.4e}   "
              f"kappa = {kappa:.4e}   delay = {v[2]-v[0]:.4f}")
    return np.array(v), kappa


def build_reference(tag, phi):
    print(f"\n=== phi = {tag} ({phi:.6f} rad): continuation reference ===")
    sol, kappa = locate_caustic(phi)
    pc_re, pc_im = sol[4], sol[5]

    p_grid = np.linspace(P_LO_FRAC * pc_re, P_HI_FRAC * pc_re, N_POINTS)
    B.RATIO = RATIO
    seeds = B.find_orbits(p_grid[0], 0.0, phi)

    tracks = []
    for s in seeds:
        x = np.array(s, float)
        row = []
        for p in p_grid:
            y, ok = B.newton(x, p, 0.0, phi)
            if ok:
                x = y
            row.append(x.copy())
        tracks.append(np.array(row))

    j_c = int(np.argmin(np.abs(p_grid - pc_re)))
    th_c = sol[0] + 1j * sol[1]
    dist = [abs((t[j_c, 0] + 1j * t[j_c, 1]) - th_c) for t in tracks]
    order = np.argsort(dist)
    iA, iB = order[0], order[1]
    print(f"  pair A = families {iA}, {iB} "
          f"(|d theta_s| {dist[iA]:.3e}, {dist[iB]:.3e}; "
          f"next nearest {dist[order[2]]:.3e})")

    # order by return delay so short/long mean the same thing at every phase
    dA = np.mean(tracks[iA][:, 2] - tracks[iA][:, 0])
    dB = np.mean(tracks[iB][:, 2] - tracks[iB][:, 0])
    if dA > dB:
        iA, iB = iB, iA
        dA, dB = dB, dA
    print(f"  short = family {iA} (delay {dA:.3f}), long = family {iB} (delay {dB:.3f})")

    XA, XB = tracks[iA], tracks[iB]
    sep = np.abs((XA[:, 0] + 1j * XA[:, 1]) - (XB[:, 0] + 1j * XB[:, 1]))
    print(f"  branch separation: min {sep.min():.4e} at p = "
          f"{p_grid[np.argmin(sep)]:.4f} (Re p_c = {pc_re:.4f})")
    if sep.min() < 1e-8:
        raise RuntimeError("branches merged numerically - tracking jumped")

    res = max(max(np.linalg.norm(B.F_real(x, p, 0.0, phi)) for x, p in zip(XA, p_grid)),
              max(np.linalg.norm(B.F_real(x, p, 0.0, phi)) for x, p in zip(XB, p_grid)))
    print(f"  continuation residual: max |F| = {res:.3e}")

    np.savez(f"{GT}/gt_bichrom_{tag}.npz", p_par=p_grid, XA=XA, XB=XB, sep=sep,
             phi=phi, ratio=RATIO, pc_re=pc_re, pc_im=pc_im, kappa=kappa, sol=sol)
    print(f"  wrote {GT}/gt_bichrom_{tag}.npz")


def box(vals, pad=3.0, floor=1e-3):
    lo, hi = vals.min(0), vals.max(0)
    c = 0.5 * (lo + hi)
    h = np.maximum(0.5 * (hi - lo) * pad, floor)
    return list(c), list(h)


def train_all(tag, phi):
    g = np.load(f"{GT}/gt_bichrom_{tag}.npz")
    p_grid, XA, XB = g["p_par"], g["XA"], g["XB"]
    B.RATIO = RATIO
    A_1, A_2 = B.A_1, B.A_2()

    p_t = torch.tensor(p_grid)
    pp0 = torch.zeros_like(p_t)
    z = ((p_t - p_grid.mean()) / p_grid.std()).unsqueeze(1)

    print(f"\n=== phi = {tag}: training, {N_SEEDS} seeds per branch ===")
    d = dict(p_par=p_grid, pc_re=float(g["pc_re"]), pc_im=float(g["pc_im"]),
             kappa=float(g["kappa"]), phi=phi, ratio=RATIO, epochs=EPOCHS,
             n_seeds=N_SEEDS)
    for tagb, Xg in (("short", XA), ("long", XB)):
        c, w = box(Xg, pad=3.0)
        cont_s = Xg[:, 0] + 1j * Xg[:, 1]
        cont_r = Xg[:, 2] + 1j * Xg[:, 3]
        S = []
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            m = PINN(c, w, input_dim=1)
            loss = train_bichrom(m, z, p_t, pp0, A_1, A_2, phi, I_P, EPOCHS)
            with torch.no_grad():
                ths, thr = m(z)
            ths, thr = ths.numpy(), thr.numpy()
            res = np.array([
                np.linalg.norm(B.F_real([a.real, a.imag, b.real, b.imag], p, 0.0, phi))
                for a, b, p in zip(ths, thr, p_grid)])
            err = np.abs(ths - cont_s)
            S.append((ths, thr, res, err, loss))
            print(f"  {tagb:5s} seed {seed}: loss {loss:.3e}  "
                  f"median |F| {np.median(res):.3e}  median |d th_s| {np.median(err):.3e}")
        mr = np.array([np.median(s[2]) for s in S])
        me = np.array([np.median(s[3]) for s in S])
        print(f"  {tagb:5s} ACROSS SEEDS: |F| {np.median(mr):.3e} "
              f"[{mr.min():.2e}, {mr.max():.2e}]   "
              f"|d th_s| {np.median(me):.3e} [{me.min():.2e}, {me.max():.2e}]")
        d[f"{tagb}_cont_s"] = cont_s
        d[f"{tagb}_cont_r"] = cont_r
        for i, s in enumerate(S):
            d[f"{tagb}_ths_{i}"], d[f"{tagb}_thr_{i}"] = s[0], s[1]
            d[f"{tagb}_res_{i}"], d[f"{tagb}_err_{i}"] = s[2], s[3]
        d[f"{tagb}_loss"] = np.array([s[4] for s in S])
    np.savez(f"{OUT}/bichrom_{tag}_pinn.npz", **d)
    print(f"  wrote {OUT}/bichrom_{tag}_pinn.npz")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    tags = list(PHASES) if which == "all" else [which]
    for tag in tags:
        phi = PHASES[tag]
        if stage in ("gt", "all"):
            build_reference(tag, phi)
        if stage in ("train", "all"):
            train_all(tag, phi)
