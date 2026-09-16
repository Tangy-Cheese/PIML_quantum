# src

Core library modules shared across the scripts. Classical/semiclassical
solvers (monochromatic and bichromatic fields) live under
[classical_solvers/](classical_solvers/README.md).

## paths.py

Canonical filesystem locations for the project.

Every path is resolved from this file's own location, not from the current
working directory, so scripts may be run from anywhere:

    python scripts/figures/make_error_maps_2d.py
    cd scripts/figures && python make_error_maps_2d.py

Both forms write to the same place.

## pinn_common.py

The k-eliminated PINN and its trainer -- shared by every published result.

k is not a network output. It is reconstructed in closed form from
(theta_s, theta_r) via the return condition (report.tex Eq. k-elimination),
so the return condition is satisfied identically and only two residuals --
tunnelling and rescattering -- enter the loss. This is the architecture of
Fig. 6 and Eq. loss-k-eliminated-full, and the loss report.tex states was
"used to produce every result in Secs. 5.2-6".

An earlier generation of scripts instead emitted k as a free sixth network
output and trained on three residuals (the unreduced system,
Eq. loss-unreduced-full). Those scripts have been removed; everything under
scripts/figures/ uses the module below, so the code matches the claim.
