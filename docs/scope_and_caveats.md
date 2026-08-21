# Scope and caveats

What this code establishes, and — as importantly — what it does not. Everything
here is stated in the dissertation too; this file collects it in one place so
that no result is quoted from the repository without its qualification.

## Established and verified

* **Reference solver.** Pseudo-arclength continuation reaches
  $\lVert F\rVert \le 10^{-12}$ and survives folds by construction (the
  augmented Jacobian stays nonsingular by a transversality argument — see the
  `continuation.py` module docstring). Validated externally: the
  caustic-derived cutoff law matches the published $10.007\,U_p + 0.538\,I_p$
  of Busuladžić *et al.* to 0.006% at $I_p \le 0.01$.
* **The caustic is found, not guessed.** It comes from the discriminant system
  ($F = 0$, $\det J = 0$, with $p_\parallel$ promoted to an unknown), giving
  $p_c = 2.197280 + 0.017697i$ for pair A at $p_\perp = 0$.
* **Newton's real failure mode is misidentification, not ill-conditioning.**
  $\kappa \le 33$ across four decades of $I_p$ generically, because
  $\mathrm{Im}\,p_c \neq 0$ keeps real momenta away from the branch point. The
  actual failure is silent: warm-started Newton at $p_\perp \ge 1.0$ returns a
  different, still-valid family with a return delay of $-4.37$ — the electron
  "returns" before it ionises — at a residual of $10^{-15}$. A wrong branch is
  still a root. A continuous network cannot make that specific error.
* **$\phi^\ast$ is constructed, not lucky.** $\mathrm{Im}\,p_c(\phi)$ is a real
  function of one real variable, so a zero crossing is generic;
  $\phi^\ast = 0.2489\pi$ gives $\kappa \approx 1.9\times10^{15}$ — the one
  configuration where the textbook ill-conditioning claim is literally true.
* **PINN across all eight paired families.** Median
  $\lvert\Delta\theta_s\rvert = 5.34\times10^{-4}$, with no degradation with
  return order: families 7/8, the latest returns and the furthest from any
  classical seed, are the *most* accurate, because the PINN needs no seed.
* **Error floor at coalescence.** A smooth, bounded-Lipschitz network cannot
  represent the fold's $d\theta/dp \to \infty$, so error saturates rather than
  degrading gradually. The universal approximation theorem guarantees
  value-closeness and says nothing about derivative-closeness.

## Qualifications that must travel with the results

* **Three distinct senses of "robustness"** — do not conflate them:
  1. *conditioning* ($\kappa$ small — true generically, false only at $\phi^\ast$);
  2. *branch continuity* (the PINN's real structural claim, true everywhere by
     construction);
  3. *seed and hyperparameter robustness* (only partially shown — the window
     ablation, the case comparison and the Appendix D results are five-seed;
     several other results are single-seed).
* **The window ablation ran at a reduced budget** (1500 epochs). At full
  training $1\times$ and $3\times$ tie. The load-bearing claim is the ordering,
  not the specific numbers.
* **Window calibration is mildly circular.** Windows are centred on the
  continuation solver's already-known range, so "the PINN finds families 7/8
  that naively-seeded Newton misses" is not yet fully apples-to-apples. The
  untried fix is to seed windows from classical simple-man estimates, which are
  closed-form and available before solving anything.
* **"Nine families at 2.5 field cycles" is a pragmatic search radius**, not a
  standard convention. The nearest comparable practice in the literature is
  orbit-count truncation (~6 shortest-travel-time trajectories dominate most
  spectral regions).
* **Only pair A's caustic has been confirmed by the discriminant system.**
  Pairs B/C/D show error peaking sharply at their own distinct momenta
  ($p \approx 1.699$, $1.997$, $1.776$), which is suggestive of their own
  caustics, but they have not been run through `caustic_point` directly. Cheap
  to check; not done.
* **$\phi^\ast$ symmetry under $p \to -p$ is unchecked** — the second harmonic
  can break the half-cycle symmetry that would otherwise guarantee it.
* **The rescattered caustic has no closed form.** After eliminating $k$ the
  system is transcendental, with no identity analogous to $\cos$ being
  two-to-one in the direct case. The discriminant *method* is exact and
  general; the specific number is necessarily numerical.
* **The intermediate momentum $k$ is not where the caustic lives.** $k$ is
  complex, but $p_c$ is a final-momentum-space object; $k$ merges to a shared
  value as a consequence of $(\theta_s, \theta_r)$ merging, not independently.

## Not claimed: the amplitude / PMD stage

`continuation.py` contains most of the hard amplitude numerics already — the
semiclassical action with tunnelling phase, the Hessian-determinant
normalisation, exponentially-scaled Airy evaluation that avoids $\sim10^{84}$
intermediate overflow, phase unwrapping and branch anchoring, and a working
two-orbit Stokes filter in `pair_amplitude()`. Three things nevertheless block
an amplitude-level result:

1. **`uniform_anchored` does not reduce correctly away from the caustic.** It
   was calibrated at $\phi^\ast$, where $\mathrm{Im}\,p_c \approx 0$ and the two
   saddles stay close; in the generic monochromatic case it departs from the
   trusted ordinary formula by up to a factor of 40 at low momentum. The likely
   locus is the branch choice fixing the sign of $\sqrt\zeta$ — equivalently,
   which saddle is labelled $\tau_a$ — which flips the sign of the odd term
   without affecting $\zeta$ itself. A second candidate, not excluded, is the
   $\zeta^{1/2}$ compensation in $C_-$.
2. **The dipole and elastic-scattering matrix elements are not implemented.**
   `orbit_amplitude()` sets $V_{pk}$ and $d$ to unity. Phase and normalisation
   are physical; absolute intensity and fine structure are not.
3. **Only one coalescing pair is handled.** `pair_amplitude()` is built for two
   orbits; summing all four pairs coherently is unattempted.

Consequently no momentum distribution appears in the dissertation. This is a
scoping decision, not a silent omission: the trajectory results are validated by
physics residual and by agreement with the continuation solver, neither of which
passes through the amplitude routine.
