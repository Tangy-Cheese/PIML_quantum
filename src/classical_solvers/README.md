# src/classical_solvers

Classical/semiclassical saddle-point solvers: root finding, continuation,
caustic detection, and semiclassical amplitude construction, for two field
shapes.

## monochromatic.py

Pseudo-arclength continuation for the rescattered ATI saddle-point system.

Phase units theta = omega * t, matching rescattered.py. The intermediate
momentum k is eliminated analytically via the return condition, leaving two
complex equations in two complex unknowns (four real). The momentum p_par is
promoted to an unknown and an arclength normalisation row is appended, so the
augmented 5x5 Jacobian stays nonsingular at a simple fold and the branch is
traced straight through the coalescence.

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
