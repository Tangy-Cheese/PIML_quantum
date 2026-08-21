import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paths import out, figure_dir
import continuation as C

OUT = out("diagnostics")


def caustic_at(Ip, v0, p_perp=0.0):
    C.I_P = Ip
    return C.caustic_point(v0, p_perp)


def newton_sweep(Ip, p_grid, x0, p_perp=0.0):
    """Warm-started fixed-p Newton across the momentum grid. Records the
    condition number and flags branch jumps.
    """
    C.I_P = Ip
    x = np.array(x0, float)
    conds, delays, ok_flags = [], [], []
    for p in p_grid:
        y, ok = C.newton(x, p, p_perp)
        if ok:
            x = y
        conds.append(C.cond_fixed_p(x, p, p_perp))
        delays.append(x[2] - x[0])
        ok_flags.append(ok)
    return np.array(conds), np.array(delays), np.array(ok_flags)


def main():
    Ip_list = np.array([0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05,
                        0.03, 0.02, 0.01, 0.005, 0.002])

    v = [4.96186, 0.96739, 9.26187, -0.01340, 2.197280, 0.017697]
    rows = []
    print(" I_p      Re p_c     Im p_c      gamma    Im(th_s)")
    for Ip in Ip_list:
        sol, ok = caustic_at(Ip, v)
        if not ok:
            print(f"{Ip:6.3f}   no convergence")
            continue
        v = sol
        Up = C.U_P
        gamma = np.sqrt(Ip / (2 * Up))
        rows.append((Ip, sol[4], sol[5], gamma, sol[1]))
        print(f"{Ip:6.3f} {sol[4]:10.5f} {sol[5]:+11.6f} {gamma:9.4f} {sol[1]:10.5f}")

    R = np.array(rows)
    Ip_a, re_c, im_c, gam, im_ths = R.T

    mask = Ip_a <= 0.2
    q = np.polyfit(np.log(Ip_a[mask]), np.log(np.abs(im_c[mask])), 1)
    print(f"\nscaling of |Im p_c| with I_p over the small-I_p range:")
    print(f"  |Im p_c| ~ I_p^{q[0]:.3f}")

    print("\nNewton conditioning on the real axis at selected I_p:")
    cond_max = []
    for Ip in Ip_list:
        idx = np.argmin(np.abs(Ip_a - Ip))
        if np.abs(Ip_a[idx] - Ip) > 1e-12:
            continue
        pc = re_c[idx]
        p_grid = np.linspace(max(0.2, pc - 0.5), pc + 0.5, 600)
        C.I_P = Ip
        orbs = C.find_orbits(max(0.2, pc - 0.5) + 0.05, 0.0)
        if len(orbs) < 2:
            cond_max.append(np.nan)
            continue
        seed = min(orbs, key=lambda x: abs((x[2] - x[0]) - 4.3))
        conds, delays, oks = newton_sweep(Ip, p_grid, seed)
        cond_max.append(conds.max())
        jump = np.abs(np.diff(delays)).max()
        print(f"  I_p={Ip:6.3f}  max cond {conds.max():.3e}  "
              f"largest delay jump {jump:.3f}  failures {(~oks).sum()}")

    C.I_P = 0.5

    np.savez(f"{OUT}/ip_scan.npz", Ip=Ip_a, re_pc=re_c, im_pc=im_c,
             gamma=gam, im_ths=im_ths, cond_max=np.array(cond_max),
             slope=q[0])

    plt.figure(figsize=(7.5, 4.5))
    plt.loglog(Ip_a, np.abs(im_c), "o-")
    plt.loglog(Ip_a, np.abs(im_c[0]) * (Ip_a / Ip_a[0]) ** q[0], "--",
               label=rf"$\propto I_p^{{{q[0]:.2f}}}$")
    plt.xlabel(r"$I_p$ (a.u.)")
    plt.ylabel(r"$|\mathrm{Im}\,p_c|$ (a.u.)")
    plt.legend()
    plt.title("Distance of the caustic from the real momentum axis")
    plt.savefig(f"{OUT}/ip_scan_imag.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    plt.semilogx(Ip_a, re_c, "o-")
    plt.xlabel(r"$I_p$ (a.u.)")
    plt.ylabel(r"$\mathrm{Re}\,p_c$ (a.u.)")
    plt.title("Caustic momentum vs ionisation potential")
    plt.savefig(f"{OUT}/ip_scan_real.png", dpi=150, bbox_inches="tight")
    plt.close()

    cm = np.array(cond_max)
    good = np.isfinite(cm)
    plt.figure(figsize=(7.5, 4.5))
    plt.loglog(Ip_a[good], cm[good], "o-")
    plt.xlabel(r"$I_p$ (a.u.)")
    plt.ylabel("max condition number on the real axis")
    plt.title("Conditioning of the fixed-$p$ Newton system")
    plt.savefig(f"{OUT}/ip_scan_cond.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    plt.loglog(gam, np.abs(im_c), "o-")
    plt.xlabel(r"Keldysh parameter $\gamma$")
    plt.ylabel(r"$|\mathrm{Im}\,p_c|$ (a.u.)")
    plt.title("Caustic offset vs Keldysh parameter")
    plt.savefig(f"{OUT}/ip_scan_gamma.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
