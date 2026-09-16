# src

Core library modules shared across the scripts.

## bichromatic.py

Rescattered ATI saddle points in a bichromatic linearly polarised field.

    A(theta) = A_1 cos(theta) + A_2 cos(2 theta + phi)

Adding the second harmonic breaks the half-cycle symmetry E(t + T/2) = -E(t),
so orbit pairs split and the relative phase phi becomes a third real control
parameter alongside (p_par, p_perp).

The analytic elimination of k via the return condition carries over unchanged.
Writing I(theta) for the antiderivative of A, the return condition gives

    k = -[I(theta_r) - I(theta_s)] / (theta_r - theta_s)

and differentiating, using only I' = A,

    dk/dtheta_s = +(A(theta_s) + k) / (theta_r - theta_s)
    dk/dtheta_r = -(A(theta_r) + k) / (theta_r - theta_s)

which is independent of the field. Only A, dA and the antiderivatives change.

## continuation.py

Pseudo-arclength continuation for the rescattered ATI saddle-point system.

Phase units theta = omega * t, matching rescattered.py. The intermediate
momentum k is eliminated analytically via the return condition, leaving two
complex equations in two complex unknowns (four real). The momentum p_par is
promoted to an unknown and an arclength normalisation row is appended, so the
augmented 5x5 Jacobian stays nonsingular at a simple fold and the branch is
traced straight through the coalescence.

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
