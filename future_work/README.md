# Future work

Code kept deliberately, but **not** part of any result in the dissertation.
Nothing here backs a published figure, table or number. It is retained because
it is the concrete starting point for the next step, not because it is finished.

## `pmd/` — photoelectron momentum distributions, pair A only

The intended chain, in run order:

| script | purpose |
|---|---|
| `make_pair_a_theta_1d.py` | 1D: $\mathrm{Re}/\mathrm{Im}(\theta_s,\theta_r)$ vs. $p$, continuation and PINN, both branches of pair A, plus the PINN's own $\lVert F\rVert$ residual |
| `make_pair_a_theta_2d.py` | the same on the full $(p_\parallel, p_\perp)$ grid — the PMD input |
| `make_pair_a_pmd.py` | assembles the PMD row by row in $p_\perp$ using `continuation.pair_amplitude()` |
| `plot_pair_a_complex.py` | complex-plane view of the pair A trajectories |

Outputs land in `output/future_work_pmd/`.

### Status — read before running

* **Blocked on a known bug, not on missing code.** `uniform_anchored` in
  `src/continuation.py` was calibrated at $\phi^\ast$ and does not reduce
  correctly to the ordinary stationary-phase amplitude far from the caustic
  (off by up to a factor of 40 at low momentum). Fixing that is the first task;
  the two candidate causes are narrowed down in
  [`../docs/scope_and_caveats.md`](../docs/scope_and_caveats.md).
* **These scripts predate the report pipeline.** They were written against the
  earlier free-$k$ PINN — three network outputs including $k$, and a
  `train(model, z, p, pp)` signature — rather than the closed-form-$k$,
  two-residual `src/pinn_common.py` used for every published result. Those two
  earlier modules are vendored in `pmd/legacy/` so the scripts remain runnable
  as written; **porting them onto `pinn_common.py` is part of the job**, since
  results produced with the legacy PINN would not match the dissertation's
  stated loss.
* **Never executed end to end.** They were syntax-checked only. Expect to debug
  shape mismatches and interface assumptions against the real runtime.
* **Scope any result as "pair A only".** `pair_amplitude()` handles one
  coalescing pair; summing all four pairs coherently is unattempted.

### `pmd/legacy/`

`all_families.py` and `error_maps.py` as they were before the closed-form-$k$
rewrite, kept solely as dependencies of the scripts above. They are superseded
by `scripts/figures/make_all_families_1d.py` and
`scripts/figures/make_error_maps_2d.py` for every purpose that touches a published
result, and their `main()` entry points write to `output/future_work_pmd/` so a
stray run cannot overwrite report data.

## Other threads worth taking up

* **Independent validation against Sufia's solver.** Compare against
  *continuation* first — both claim to be exact, so a disagreement there is the
  priority bug, before the PINN is touched at all. Then compare the PINN the
  same way it is currently compared against continuation (same
  $\lvert\Delta\theta_s\rvert$ metric, same per-family breakdown). Watch
  specifically for agreement on the *pairing* — which families merge with which
  — and on caustic locations, not just $\theta$ values at scattered points:
  those are the claims most exposed to a subtly wrong equation that still gives
  small residuals. Keep the continuation solver either way; the point is two
  independent references agreeing, not replacing one with the other.
* **Confirm the caustics of pairs B, C and D** through `caustic_point`, as was
  done for pair A. Cheap.
* **Window parameterisation beyond width.** Current work varies only the
  half-width around a centre taken from continuation's known range. Open: take
  the centre from classical simple-man estimates instead (this closes the
  circularity noted in the caveats); whether window *shape* matters; whether a
  field-symmetry prior can generate a mirrored window for a symmetric partner
  family for free.
