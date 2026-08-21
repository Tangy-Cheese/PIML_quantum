"""Produces: cutoff_law.png

Fits the rescattered cutoff law E_c = b*Ip + a*Up from caustic locations
found by the discriminant system, and compares to the published values
10.007 Up + 0.538 Ip.

Slide reference: "Validation: cutoff law"
Requires: ip_scan.npz (run ip_scan.py first)
Runtime: <1 minute (uses saved data, no resolving)
"""

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


def main():
    d = np.load(f"{OUT}/ip_scan.npz")
    Ip, re_pc = d["Ip"], d["re_pc"]
    Ec = re_pc ** 2 / 2

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(Ip, Ec / C.U_P, "o", ms=6, color="#241040",
             label="caustic from discriminant system")

    xs = np.linspace(0, 0.52, 100)
    plt.plot(xs, (0.538 * xs + 10.007 * C.U_P) / C.U_P, "-", color="#d05070", lw=1.6,
             label=r"literature $10.007U_p+0.538I_p$")

    mask = Ip <= 0.01
    q = np.polyfit(Ip[mask], Ec[mask], 1)
    plt.plot(xs, (q[0] * xs + q[1]) / C.U_P, "--", color="#9b7fd4", lw=1.6,
             label=rf"our fit ($I_p\leq0.01$): ${q[0]:.4f}I_p+{q[1]/C.U_P:.4f}U_p$")

    plt.xlabel(r"$I_p$ (a.u.)")
    plt.ylabel(r"caustic energy ($U_p$)")
    plt.legend(fontsize=8)
    plt.title("Rescattered cutoff law vs the published formula")
    plt.savefig(f"{OUT}/cutoff_law.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(f"fit (Ip <= 0.01): b = {q[0]:.4f}  (literature 0.538)")
    print(f"                  a = {q[1]/C.U_P:.4f}  (literature 10.007)")
    print("figure written: cutoff_law.png")


if __name__ == "__main__":
    main()
