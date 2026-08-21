import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
import continuation as C

OUT = out("ground_truth")
SEED_P = 0.40


def family_track(family_index, p_grid, p_perp):
    seeds = C.find_orbits(SEED_P, 0.0)
    seed = seeds[family_index]
    y, ok = C.newton(seed, SEED_P, p_perp)
    if ok:
        seed = y

    samples = {}
    for d in (+1.0, -1.0):
        path, _, _ = C.continue_branch(
            seed, SEED_P, p_perp,
            p_lo=p_grid[0] - 0.02, p_hi=p_grid[-1] + 0.02,
            ds=2e-3, ds_max=1e-2, direction=d
        )
        for j, xs in C.sample_path(path, p_grid, p_perp).items():
            if j not in samples:
                samples[j] = xs[0]

    idx = sorted(samples.keys())
    X = np.array([samples[j] for j in idx])
    return p_grid[idx], X


def main():
    p_grid = np.linspace(0.35, 2.55, 400)

    # families 1 and 2 are the short and long branches of orbit pair A
    for fam, branch in ((1, "short"), (2, "long")):
        p, X = family_track(fam, p_grid, 0.0)
        res = np.array([np.linalg.norm(C.F_real(x, pp, 0.0)) for x, pp in zip(X, p)])
        k = np.array([C.k_of(x[0] + 1j * x[1], x[2] + 1j * x[3]) for x in X])

        print(f"pair A {branch} (family {fam}): {len(p)} points, p in [{p.min():.3f}, {p.max():.3f}]")
        print(f"  max |F|      : {res.max():.3e}")
        print(f"  Re th_s      : [{X[:,0].min():.4f}, {X[:,0].max():.4f}]")
        print(f"  Im th_s      : [{X[:,1].min():.4f}, {X[:,1].max():.4f}]")
        print(f"  Re th_r      : [{X[:,2].min():.4f}, {X[:,2].max():.4f}]")
        print(f"  Im th_r      : [{X[:,3].min():.4f}, {X[:,3].max():.4f}]")
        print(f"  Re k         : [{k.real.min():.4f}, {k.real.max():.4f}]")
        print(f"  Im k         : [{k.imag.min():.4f}, {k.imag.max():.4f}]")

        np.savez(f"{OUT}/gt_pair_a_{branch}.npz", p_par=p, X=X, k=k, residual=res)


if __name__ == "__main__":
    main()
